from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,log_loss
X,y=load_breast_cancer(return_X_y=True)
a,b,c,d=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
s=StandardScaler(); a=s.fit_transform(a); b=s.transform(b)
models=[("MLE approx",LogisticRegression(C=1e6,max_iter=5000)),("MAP L2",LogisticRegression(C=.1,max_iter=5000)),("MAP L1",LogisticRegression(C=.1,solver="saga",penalty="l1",max_iter=5000))]
for n,m in models:
    m.fit(a,c); p=m.predict(b)
    print(n,"accuracy",accuracy_score(d,p),
          "logloss",log_loss(d,m.predict_proba(b)),
          "parameter L2 norm", (m.coef_**2).sum()**0.5,
          "nonzero", (m.coef_!=0).sum())
