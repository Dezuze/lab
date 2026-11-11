from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
X=StandardScaler().fit_transform(load_digits().data)
results=[]
for k in [2,4,6,8,10,12]:
    m=KMeans(n_clusters=k,n_init=10,random_state=42).fit(X)
    silhouette=silhouette_score(X,m.labels_)
    results.append((k,m.inertia_,silhouette))
    print("K",k,"inertia",round(m.inertia_,2),"silhouette",round(silhouette,4))
best=max(results,key=lambda result: result[2])
print("Best K by silhouette:", best[0])
ks, inertias, silhouettes=zip(*results)
fig, axes=plt.subplots(1,2,figsize=(10,4))
axes[0].plot(ks,inertias,"o-"); axes[0].set_title("Elbow: inertia"); axes[0].set_xlabel("K")
axes[1].plot(ks,silhouettes,"o-"); axes[1].set_title("Silhouette score"); axes[1].set_xlabel("K")
plt.tight_layout(); plt.show()
