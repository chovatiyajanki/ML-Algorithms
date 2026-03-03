import sklearn as sk
from sklearn import datasets
from sklearn.datasets import load_iris
from sklearn import tree
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier,plot_tree

iris = load_iris()
print(iris)

X = iris.data
Y = iris.target

clf = tree.DecisionTreeClassifier(max_leaf_nodes=3,random_state= 0)
clf = clf.fit(X,Y)

tree.plot_tree(clf)
plt.show()