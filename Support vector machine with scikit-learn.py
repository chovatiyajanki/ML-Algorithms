import sklearn as sk
from sklearn import datasets
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

dataset = datasets.load_iris()
print(dataset)
print(type(dataset))
print('-------------------')

X = dataset.data
y = dataset.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=0)

svm_model = svm.SVC(kernel='linear')
svm_model.fit(X_train,y_train)

dataClass = svm_model.predict(X_test)

print('The Iris Type Is : ')
print(dataset.target_names[dataClass])
print(accuracy_score(y_test,dataClass))
print(confusion_matrix(y_test,dataClass,labels=[0,1,2]))
print('---------------program ends --------------------')