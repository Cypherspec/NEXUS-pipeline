"""Minimal NEXUS example on a public OpenML dataset (credit-g).

    python examples/quickstart.py
"""

from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from tabular_ensemble import NexusClassifier

X, y = fetch_openml(data_id=31, return_X_y=True, as_frame=True, parser="auto")
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

clf = NexusClassifier(n_jobs=-1, random_state=0).fit(X_tr, y_tr)
proba = clf.predict_proba(X_te)

print(f"accuracy {accuracy_score(y_te, clf.predict(X_te)):.3f}")
print(f"log loss {log_loss(y_te, proba, labels=clf.classes_):.3f}")
print(f"ROC-AUC  {roc_auc_score(y_te == clf.classes_[1], proba[:, 1]):.3f}")
print("ensemble weights:")
for name, w in sorted(clf.weights_.items(), key=lambda kv: -kv[1]):
    if w > 0:
        print(f"  {name:<16} {w:.2f}")
print(f"temperature {clf.temperature_:.2f}")
