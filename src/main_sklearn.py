from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

from spn.gpu.TensorFlow import add_node_to_tf_graph
from spn.sklearn.classifier import SPNClassifier
from spn.sklearn.strategy.optimization import classification_categorical_to_tf_graph
from spn.sklearn.strategy.structure import LearnClassifierParametric
from spn.structure.leaves.parametric.Parametric import Categorical

add_node_to_tf_graph(Categorical, classification_categorical_to_tf_graph)
X, y = load_iris(return_X_y=True)
struct = LearnClassifierParametric(min_instances_slice=25)
clf = SPNClassifier(structure_learner=struct)
scores = cross_val_score(clf, X, y, cv=10, n_jobs=4)
print("Scores: ", scores)
