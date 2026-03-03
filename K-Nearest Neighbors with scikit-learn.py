import sklearn as sk
from sklearn import datasets
from sklearn import neighbors 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

dataset = datasets.load_iris()
print("This dataset is : ",dataset)
print("Dataset type is : ",type(dataset))

X = dataset.data
Y = dataset.target

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33,random_state=42)

model = sk.neighbors.KNeighborsClassifier(n_neighbors=2,weights="distance",metric="manhattan")
print("\nModel is : ",model)

model_fit = model.fit(x_train,y_train)
print("\nModel fit is : ",model_fit)

data_class = model.predict(x_test)
print("\nDataClass is : ",data_class)

print("\nThe Iris Type is : ")
print(dataset.target_names[data_class])

print("\nConfusion matrix is : ")
print(confusion_matrix(y_test,data_class,labels=[0,1,2]))

print("\nAccuracy Score is : ",accuracy_score(y_test,data_class))