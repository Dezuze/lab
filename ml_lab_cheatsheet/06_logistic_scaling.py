# pima.csv: columns Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigree,Age,Outcome
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
df=pd.read_csv("pima.csv")
cols=["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
df[cols]=df[cols].replace(0,np.nan)
X=df.drop(columns="Outcome"); y=df.Outcome
a,b,c,d=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
imputer=SimpleImputer(strategy="median")
a=imputer.fit_transform(a); b=imputer.transform(b)
for name,scale in [("Raw",False),("Scaled",True)]:
    A,B=a.copy(),b.copy()
    if scale:
        s=StandardScaler(); A=s.fit_transform(A); B=s.transform(B)
    m=LogisticRegression(max_iter=1000).fit(A,c)
    p=m.predict(B)
    print(name, "accuracy", accuracy_score(d,p),
          "precision", precision_score(d,p,zero_division=0),
          "recall", recall_score(d,p,zero_division=0),
          "F1", f1_score(d,p,zero_division=0))
    print(classification_report(d,p,zero_division=0))
