from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,RidgeCV,LassoCV
from sklearn.metrics import mean_squared_error,r2_score
X,y=load_diabetes(return_X_y=True)
a,b,c,d=train_test_split(X,y,test_size=.2,random_state=42)
s=StandardScaler(); a=s.fit_transform(a); b=s.transform(b)
for name,m in [("Linear",LinearRegression()),("Ridge",RidgeCV(alphas=[.01,.1,1,10,100])),("Lasso",LassoCV(cv=5,max_iter=10000))]:
    m.fit(a,c); p=m.predict(b)
    print(name,"MSE",mean_squared_error(d,p),"R2",r2_score(d,p),
          "alpha",getattr(m,"alpha_", "-"),
          "nonzero coefficients", (m.coef_ != 0).sum())
