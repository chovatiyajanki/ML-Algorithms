from sklearn.cluster import KMeans
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import homogeneity_score

iris = datasets.load_iris()
x = iris.data
y = iris.target
err = []
for i in range(1,11):
    est = KMeans(n_clusters=i,n_init=25,init="k-means++",random_state=0)
    predict_y = est.fit_predict(x)
    err.append(est.inertia_)

plt.plot(range(1,11),err)
plt.title("elbow method")
plt.xlabel('member of cluster')
plt.ylabel('error')
plt.show()

Fest = KMeans(n_clusters=i,n_init=25,init="k-means++",max_iter=1500,random_state=0)
predict_y = Fest.fit_predict(x)
plt.scatter(x[predict_y == 0,0],x[predict_y == 0,1],s = 100, c = "red" , label = "iris-setosa")
plt.scatter(x[predict_y == 1,0],x[predict_y == 1,1],s = 100, c = "blue" , label = "iris-vericolor")
plt.scatter(x[predict_y == 2,0],x[predict_y == 2,1],s = 100, c = "green" , label = "iris-virginica")
plt.scatter(Fest.cluster_centers_[:,0],Fest.cluster_centers_[:,1],s = 100,c = "yellow")
plt.legend()
plt.show()

print("score : ",homogeneity_score(y,predict_y))