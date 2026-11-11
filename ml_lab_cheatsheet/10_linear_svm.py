# Experiment 10: Linear SVM, decision boundary, margins and support vectors
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

d = load_iris(); X = d.data[:, 2:4]; y = (d.target == 0).astype(int)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42)
s = StandardScaler(); Xtr = s.fit_transform(Xtr); Xte = s.transform(Xte)
m = SVC(kernel='linear', C=1.0).fit(Xtr, ytr)
print('Accuracy:', accuracy_score(yte, m.predict(Xte)))

w, b = m.coef_[0], m.intercept_[0]
margin_width = 2 / np.linalg.norm(w)
print('Support vectors:', len(m.support_vectors_),
      'margin width:', margin_width,
      'weight vector:', w)
xx = np.linspace(Xtr[:, 0].min(), Xtr[:, 0].max(), 100)
plt.scatter(Xtr[:, 0], Xtr[:, 1], c=ytr)
plt.scatter(m.support_vectors_[:, 0], m.support_vectors_[:, 1],
            s=100, facecolors='none', edgecolors='red', label='Support vectors')
for offset, label in [(0, 'Decision boundary'), (1, 'Upper margin'), (-1, 'Lower margin')]:
    plt.plot(xx, -(w[0] * xx + b - offset) / w[1], label=label)
plt.xlabel('Scaled petal length'); plt.ylabel('Scaled petal width'); plt.legend(); plt.show()
