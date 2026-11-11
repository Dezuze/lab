from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,f1_score
tr=fetch_20newsgroups(subset="train",remove=("headers","footers","quotes"))
te=fetch_20newsgroups(subset="test",remove=("headers","footers","quotes"))
v=CountVectorizer(stop_words="english",max_features=10000)
A=v.fit_transform(tr.data); B=v.transform(te.data)
for name,m in [("Multinomial",MultinomialNB()),("Bernoulli",BernoulliNB())]:
    m.fit(A,tr.target); p=m.predict(B)
    print(name,"accuracy",accuracy_score(te.target,p),"macro-F1",f1_score(te.target,p,average="macro"))
