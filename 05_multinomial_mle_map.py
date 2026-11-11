# Experiment 5: text -> vectorize -> MLE/MAP -> compare.
# Requires cached 20 Newsgroups on first run.
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
tr=fetch_20newsgroups(subset="train",categories=["sci.space","rec.autos","comp.graphics","talk.religion.misc"],remove=("headers","footers","quotes"))
te=fetch_20newsgroups(subset="test",categories=tr.target_names,remove=("headers","footers","quotes"))
v=CountVectorizer(stop_words="english",max_features=5000)
X=v.fit_transform(tr.data); T=v.transform(te.data)
# alpha close to zero is the maximum-likelihood estimate; positive alpha is
# Laplace smoothing, the MAP estimate under a symmetric Dirichlet prior.
for name, alpha in [("MLE", 1e-9), ("MAP", 1.0)]:
    m=MultinomialNB(alpha=alpha).fit(X,tr.target)
    p=m.predict(T)
    print(name, "alpha", alpha, "accuracy", accuracy_score(te.target,p),
          "macro-F1", f1_score(te.target,p,average="macro"),
          "mean class-word probability", m.feature_log_prob_.mean())
