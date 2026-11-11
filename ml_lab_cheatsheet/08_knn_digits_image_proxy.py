# Uses a local Fashion-MNIST .npz when available; otherwise uses bundled
# Digits as an offline image-classification fallback.
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
fashion_path = Path("fashion-mnist.npz")
if fashion_path.exists():
    data = np.load(fashion_path)
    X = np.concatenate((data["x_train"], data["x_test"]))
    y = np.concatenate((data["y_train"], data["y_test"]))
    X=X.reshape(len(X), -1).astype("float32") / 255.0
    print("Dataset: Fashion-MNIST")
else:
    digits=load_digits()
    X=digits.images.reshape(len(digits.images), -1)
    X=X.astype("float32") / 16.0
    y=digits.target
    print("Dataset: bundled Digits fallback (add fashion-mnist.npz for Fashion-MNIST)")
a,b,c,d=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
for k in [1,3,5,7,9]:
    m=KNeighborsClassifier(n_neighbors=k).fit(a,c)
    print("K",k,"accuracy",accuracy_score(d,m.predict(b)))
