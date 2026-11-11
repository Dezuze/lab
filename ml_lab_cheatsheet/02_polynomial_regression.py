# Place mpg.csv beside this file. Required columns: mpg, displacement.
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
df=pd.read_csv("mpg.csv").dropna(subset=["mpg","displacement"])
X=df[["displacement"]]; y=df.mpg
a,b,c,e=train_test_split(X,y,test_size=.2,random_state=42)
grid=np.linspace(X.min().iloc[0],X.max().iloc[0],200).reshape(-1,1)
plt.scatter(b,e,label="test")
for degree in [1,2,3,4]:
    poly=PolynomialFeatures(degree); A=poly.fit_transform(a); B=poly.transform(b)
    model=LinearRegression().fit(A,c); p=model.predict(B)
    label = "Linear (degree 1)" if degree == 1 else f"Polynomial degree {degree}"
    print(label,"MSE",mean_squared_error(e,p),"R2",r2_score(e,p))
    plt.plot(grid,model.predict(poly.transform(grid)),label=label)
plt.legend(); plt.show()
