from typing import List, Union

from spn.algorithms import add_node_mpe
from spn.algorithms.Inference import add_node_likelihood

import numpy as np
import tensorflow as tf

from spn.gpu.TensorFlow import add_node_to_tf_graph, add_tf_graph_to_node
from spn.structure.Base import Sum, Node, get_nodes_by_type, Product, assign_ids, rebuild_scopes_bottom_up, Leaf
import logging

logger = logging.getLogger(__name__)


class Maxout(Node):
    """
    A Maxout node that behaves equivalent to Maxout Networks (https://arxiv.org/pdf/1302.4389.pdf).
    Each Maxout node simulates `k` internal Sum nodes and only outputs the
    maximum result of the list of internal sum node results.

    Maxout(x, k) = max_k{ Sum_k(x) }

    """

    def __init__(self, internal: Sum = None, k: int = 2, children: List[Node] = None):
        """
        Create a Maxout node from a given sum node. Contains k internal pseudo-sum
        nodes represented in the weight matrix.

        Parameters
        ----------
        internal : Sum
            Sum node to augment.
        k : int
            Number of internal sum nodes.
        children : List[Node]
            List of children.
        """
        Node.__init__(self)

        # Check arguments
        if internal is None and children is None:
            raise Exception("Maxout node needs either an internal sum node or explicit children but both were None.")
        if internal is not None and children is not None:
            raise Exception(
                "Maxout node can only be instantiated either from a sum "
                "node or from explicit children but both were given."
            )

        # Create from internal
        if internal is not None:
            self.scope = internal.scope
            self.children = internal.children
            self.k = k
            self.weights = np.zeros((len(self.children), k))
            for i in range(k):
                self.weights[:, i] = np.array(internal.weights)

        # Create with explicit children
        if children is not None:
            self.children = children
            self.k = k
            self.weights = np.random.rand(len(children), k)
            renormalize_weights(self)


def renormalize_weights(node: Union[Sum, Maxout]):
    """
    Renormalize a nodes weights such that they are all positive and sum to one.

    Parameters
    ----------
    node : Union[Sum, Maxout]
        Node that contains weights. Can be either a sum or a maxout node.
    """
    node.weights /= np.sum(node.weights, axis=0)


def assert_normalized_weights(node: Union[Sum, Maxout]):
    """
    Assert, that the weights of node are normalized.
    
    Parameters
    ----------
    node : Union[Sum, Maxout]
        Node whose weights are to be checked
    """
    assert np.isclose(np.sum(node.weights, axis=0), 1.0).all(), "Unnormalized weights {} for node {}".format(
        node.weights, node
    )


def maxout_log_likelihood(node: Maxout, children, data=None, dtype=np.float64) -> np.ndarray:
    """
    Compute the log likelihood of a given Maxout node based on the data and its
    children.

    Parameters
    ----------
    node : Maxout
        Maxout input node.
    children : 
        Children's log likelihood.
    data : 
        Evidence.
    dtype : 
        Data type.

    Returns
    -------
    np.ndarray
        Log likelihood of the given Maxout node.
    """
    llchildren = np.concatenate(children, axis=1)
    assert llchildren.dtype == dtype
    assert_normalized_weights(node)

    # Simulate scipy.special.logsumexp for weight matrices
    res = np.log(np.dot(np.exp(llchildren), node.weights))

    # Choose the highest log likelihood
    maxlls = np.max(res, axis=1)
    return maxlls.reshape(-1, 1)


def maxout_likelihood(node: Maxout, children, data=None, dtype=np.float64) -> np.ndarray:
    """
    Compute the likelihood of a given maxout node based on the data and its
    children.

    Parameters
    ----------
    node : Maxout
        Maxout node.
    children : 
        Children's likelihood.
    data : 
        Evidence.
    dtype : 
        Data type.

    Returns
    -------
    np.ndarray
        Likelihood of the given Maxout node.
    """
    llchildren = np.concatenate(children, axis=1)
    assert llchildren.dtype == dtype

    assert_normalized_weights(node)

    # Calculate the likelihood for each internal node
    ls = np.dot(llchildren, node.weights)

    # Choose the highest likelihood
    res = np.max(ls, axis=1)
    return res.reshape(-1, 1)


def mpe_maxout(node: Maxout, parent_result, data=None, lls_per_node: np.ndarray = None, rand_gen=None):
    """
    Compute the MPE for a given Maxout node.

    Parameters
    ----------
    node : Maxout
        Node to compute the MPE for
    parent_result : 
        Parent index ids in lls_per_node.
    data : 
        Input data.
    lls_per_node : np.ndarray
        Array of size len(data) x len(nodes).
    rand_gen : 
        Random number generator.

    Returns
    -------
    List[List[int]]
        Children row ids that had the highest log likelihood.
    """
    if len(parent_result) == 0:
        return None

    # Shape: (number of datapoints, number of children, number of internal nodes)
    num_datapoints = len(parent_result)
    num_children = len(node.children)
    num_internal_sumnodes = node.k

    # Calculate children log probabilities with appropriate weights
    w_children_log_probs = np.zeros((num_datapoints, num_children, num_internal_sumnodes))
    weights = node.weights
    for i, child in enumerate(node.children):
        lls = lls_per_node[parent_result, child.id]
        # Sum over children axis
        w_children_log_probs[:, i, :] = lls.reshape(num_datapoints, 1) + np.log(weights[i, :])

    # Reduce to sum along children axis
    child_sums = np.sum(w_children_log_probs, axis=1)
    assert child_sums.shape == (num_datapoints, num_internal_sumnodes)

    # Find argmax along internal node axis (indicates the internal node that had the
    # maximum sum of children's log likelihood)
    firing_internal_node_idxs = np.argmax(child_sums, axis=1)
    assert firing_internal_node_idxs.shape == (num_datapoints,)

    firing_internal_node_log_probs = w_children_log_probs[np.arange(num_datapoints), :, firing_internal_node_idxs]
    assert firing_internal_node_log_probs.shape == (num_datapoints, num_children)

    # Get the idx of the maximum child branch for each datapoint
    max_child_branches = np.argmax(firing_internal_node_log_probs, axis=1)
    assert max_child_branches.shape == (num_datapoints,)

    children_row_ids = []
    for i, child in enumerate(node.children):
        children_row_ids.append(parent_result[max_child_branches == i])

    return children_row_ids


def log_maxout_to_tf_graph(
    node: Maxout, children, data_placeholder=None, variable_dict=None, log_space=True, dtype=np.float32
):
    """
    Compute the log likelihood of a maxout node in a tf graph.

    Parameters
    ----------
    node : Maxout
        Maxout input node.
    children :
        Children node log likelihoods.
    data_placeholder :
        Data placeholder.
    variable_dict : Dict[Node, tf.Variable]
        Dict that maps from a SPN node to its tensor in the tf graph.
    log_space : bool
        Flag whether to compute likelihoods in log space. Must be true.
    dtype : dtype
        Data input type.

    Returns
    -------
    tf.Tensor
        Tensor that represents the Maxout log likelihood.
    """
    assert log_space

    with tf.variable_scope("%s_%s" % (node.__class__.__name__, node.id)):
        softmax_inverse = np.log(node.weights / np.max(node.weights, axis=0)).astype(dtype)
        # Apply maxout to enforce weights in (0, 1) and sum(w) = 1.0 via softmax over children axis (0)
        tfweights = tf.nn.softmax(tf.get_variable("weights", initializer=tf.constant(softmax_inverse)), axis=0)
        variable_dict[node] = tfweights
        childrenprob = tf.stack(children, axis=1)

        # Children prob is of shape N x C (num datapoints x num children)
        # To broadcast the log(weights) matrix of shape C x K (k: num internal
        # weights per child) onto the children we need to artificially extend the
        # childrenprob matrix to shape N x C x K
        childrenprob = tf.tile(tf.expand_dims(childrenprob, axis=2), multiples=(1, 1, node.k))

        # Broadcast childrenprob (N x C x K) onto weights (C x K) and reduce to the sum over the children axis
        res = tf.reduce_logsumexp(childrenprob + tf.log(tfweights), axis=1)
        # Take max along K axis
        res = tf.reduce_max(res, axis=1)
        return res


def tf_graph_to_maxout(node: Maxout, tfvar):
    """
    Transform a tensorflow variable into a SPN node

    Parameters
    ----------
    node : Maxout 
        SPN node to be updated with information from the tensorflow variable.
    tfvar : tf.Tensor 
        Tensorflow variable.
    """
    node.weights = tfvar.eval()
    assert np.all(node.weights > 0), "Maxout weights were negative after tf optimization: %s" % node.weights


def augment_spn_maxout_all(spn: Node, k: int, transform_func=None) -> Node:
    """
    Augment all Sum nodes in an SPN to Maxout nodes.

    Parameters
    ----------
    spn : Node 
        SPN root node
    k : int
        Number of internal Sum nodes per Maxout node
    transform_func : 
        Transformation function that transforms a Sum node into a Maxout node

    Returns
    -------
    Node
        Maxout augmented SPN.
    """
    sum_nodes = get_nodes_by_type(spn, Sum)
    for sn in sum_nodes:
        spn = augment_spn_maxout(spn=spn, sumnode=sn, k=k, transform_func=transform_func, reassign_ids=False)

    # Fix ids and scopes
    assign_ids(spn)
    rebuild_scopes_bottom_up(spn)
    return spn


def augment_spn_maxout_random(spn: Node, k: int, transform_func=None) -> Node:
    """
    Augment all Sum nodes in an SPN to Maxout nodes.

    Parameters
    ----------
    spn : Node
        SPN root node.
    k : int
        Number of internal Sum nodes per Maxout node.
    transform_func : 
        Transformation function that transforms a Sum node into a Maxout node.

    Returns
    -------
    Node
        Maxout augmented SPN.
    """
    sum_nodes = get_nodes_by_type(spn, Sum)
    for sn in sum_nodes:
        spn = augment_spn_maxout(spn=spn, sumnode=sn, k=k, transform_func=transform_func, reassign_ids=False)

    # Fix ids and scopes
    assign_ids(spn)
    rebuild_scopes_bottom_up(spn)
    return spn


def augment_spn_maxout(spn: Node, sumnode: Sum, k: int, transform_func=None, reassign_ids: bool = True):
    """
    Augment an spn such that the given sumnode gets replaced with a Maxout node.

    Parameters
    ----------
    spn : Node
        Root node of the SPN.
    sumnode : Sum
        Sum node that gets replaced with a Maxout node.
    k : int
        Number of internal sum nodes in the new Maxout node.
    transform_func : 
        Transformation function that transforms a Sum node into a Maxout node.

    Returns
    -------
    Node
        Root node of the SPN `spn` with the augmented Sum node.
    """

    # First case: sumnode is the rootnode of the SPN
    # Simply replace with maxout node and return
    node_is_root = spn == sumnode
    if node_is_root:
        # Parent was root node - therefore, create a new parent
        maxout = transform_func(sumnode, k)
        if reassign_ids:
            # Fix ids and scopes
            assign_ids(spn)
            rebuild_scopes_bottom_up(spn)
        return maxout

    # Second Case: sumnode is not rootnode of the SPN
    # Wrap sumnode with maxout node and find all parents of the sumnode and update their children list
    parents = get_parents(spn, sumnode)
    for parent in parents:
        # Remove sumnode from parents children
        parent.children.remove(sumnode)

        # Create maxout node from sumnode
        maxout_node = transform_func(sumnode, k)

        # Add the new maxout node to the children list
        parent.children.append(maxout_node)

    if reassign_ids:
        # Fix ids and scopes
        assign_ids(spn)
        rebuild_scopes_bottom_up(spn)
    return spn


def create_sum_to_maxout_func(weight_augmentation_factor):
    """
    Create a sum-to-maxout lambda function that takes a sum node and k parameter and turns the sum node into a maxout 
    node with k internal sum nodes. Each internal sum node's weights will be augmented with uniformly drawn random
    numbers between 0 and `weight_augmentation_factor`.

    weight_augmentation_factor : float
        Scaling factor of the uniform distribution used to augment the internal sum node weights.
    """
    def sum_to_maxout(sumnode: Sum, maxout_k: int):
        # Create maxout node from sum node
        maxout_node = Maxout(internal=sumnode, k=maxout_k)

        # Add some noise to weights
        maxout_node.weights += np.random.randn(*maxout_node.weights.shape) * weight_augmentation_factor

        # Make sure no negative weights occurred
        # Note: Do not clip into (0, 1) range as negative weights would be clipped to 0 which stays
        # 0 after renormalization below
        maxout_node.weights = np.abs(maxout_node.weights)

        # Renormalize weights
        renormalize_weights(maxout_node)
        return maxout_node

    return sum_to_maxout


def sum_to_maxout_rand(sumnode: Sum, maxout_k: int) -> Maxout:
    """
    Convert a sum node to a maxout node with randomly initialized weights.

    Parameters
    ----------
    sumnode : Sum
        Sum node to get the children from.
    k : int
        Maxout k parameter which is the number of internal sum nodes.

    Returns
    -------
    Maxout
        Maxout node.
    """
    return Maxout(children=sumnode.children, k=maxout_k)


def get_parents(spn: Node, child: Union[Node, int]) -> List[Node]:
    """
    Get parents of a certain child.

    Parameters
    ----------
    spn : Node
        Root node of the SPN.
    child : Union[Node, int]
        Either child node or child node id of which to find the parents.
    
    Returns
    -------
    List[Node]
        Parents of the child node.
    """
    nodes = get_nodes_by_type(spn)
    nodes = filter(lambda n: not isinstance(n, Leaf), nodes)

    child_id = child.id

    # Collect parents
    parents = []

    # Easiest solution: root node is child
    if spn.id == child_id:
        return parents

    # Iterate over all possible nodes
    for candidate in nodes:
        # For each candidate check if it's children contains the child node id
        if child_id in [c.id for c in candidate.children]:
            parents.append(candidate)

    return parents


def _get_id(node: Union[Node, int]):
    """
    Get the id of a node.

    Parameters
    ----------
    node : Union[Node, int]
        Node to get the id from.

    Returns
    -------
    int
        Id of the passed node.
    """
    if isinstance(node, Node):
        # If node is passed, get node id
        child_id = node.id
    elif isinstance(node, int):
        # If id is passed, do nothing
        child_id = node
    else:
        raise TypeError("Parameter <node> must be either a Node or the id of a node.")
    return child_id


def sklearn_classifier_pre_opt_maxout(maxout_k:int, transform_func):
    def hook(spn: Node):
        spn = augment_spn_maxout_all(
            spn=spn, k=maxout_k, transform_func=transform_func
        )
        return spn
    return hook
    

def print_maxout_nodes(spn):
    nodes = get_nodes_by_type(spn, Maxout)
    for n in nodes:
        logger.info("Maxout Node [{}]:".format(n.id))
        logger.info("\n{}".format(n.weights))


# Register MPE and likelihoods in SPFlow
add_node_mpe(
    node_type=Maxout, bottom_up_lambda=maxout_log_likelihood, top_down_lambda=mpe_maxout, bottom_up_lambda_is_log=True
) 
add_node_likelihood(node_type=Maxout, lambda_func=maxout_likelihood, log_lambda_func=maxout_log_likelihood)
add_node_to_tf_graph(node_type=Maxout, lambda_func=log_maxout_to_tf_graph)
add_tf_graph_to_node(node_type=Maxout, lambda_func=tf_graph_to_maxout)
