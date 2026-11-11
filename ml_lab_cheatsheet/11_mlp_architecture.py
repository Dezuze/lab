# winequality-red.csv (semicolon separated); change sep if your CSV is comma separated.
import pandas as pd, time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
df=pd.read_csv("winequality-red.csv",sep=";"); X=df.drop(columns="quality"); y=df.quality
a,b,c,d=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
s=StandardScaler(); a=s.fit_transform(a); b=s.transform(b)
for layers in [(10,),(100,),(64,32,16)]:
    m=MLPClassifier(hidden_layer_sizes=layers,max_iter=1000,random_state=42)
    t=time.time(); m.fit(a,c); print(layers,"accuracy",accuracy_score(d,m.predict(b)),"epochs",m.n_iter_,"sec",round(time.time()-t,2))
