import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def sigmoid(z):
    return (1/(1+np.exp(-z)))

def hypothesis(x,theta):
    return sigmoid(np.dot(x,theta))

def cost_function(x,y,theta):
    m=len(y)
    h=hypothesis(x,theta)
    cost=-(1/m)*np.sum(y*np.log(h)+(1-h)*np.log(1-h))
    return cost

def gradient_descent(x,y,theta,alpha,num_iteration):
    m=len(y)
    cost_history=[]
    for i in range(num_iteration):
        h=hypothesis(x,theta)
        theta-=(alpha/m)*np.dot(x.T,(h-y))
        cost_history.append(cost_function(x,y,theta))
    return theta,cost_history
def predict(x,theta):
    return np.round(hypothesis(x,theta))

data=load_breast_cancer()
X=data.data
y=data.target
scaler=StandardScaler()
X=scaler.fit_transform(X)
X=np.c_[np.ones(X.shape[0]),X]

X_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

alpha=0.01
num_iterations=1000
theta=np.zeros(X_train.shape[1])
theta,cost_history=gradient_descent(X,y,theta,alpha,num_iterations)
predictions=predict(x_test,theta)

acc=accuracy_score(y_test,predictions)

print(f'Predictions: {predictions}')
print(f'Cost history: {cost_history[-10:]}')