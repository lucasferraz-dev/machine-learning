import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=1000, noise=.30, random_state=42)
X_dev, X_te, y_dev, y_te = train_test_split(
    X, y, test_size=.20, stratify=y, random_state=42)
X_tr, X_va, y_tr, y_va = train_test_split(
    X_dev, y_dev, test_size=.25, stratify=y_dev, random_state=42)

models = {
    'Tree': DecisionTreeClassifier(random_state=42),
    'Forest': RandomForestClassifier(n_estimators=100,
                                     min_samples_leaf=3, max_features=1, random_state=42)
}

validation = {}
for name, m in models.items():
    m.fit(X_tr, y_tr)
    tr = accuracy_score(y_tr, m.predict(X_tr))
    va = accuracy_score(y_va, m.predict(X_va))
    validation[name] = va
    print(f'{name}: train={tr:.1%}; validation={va:.1%}')

chosen_name = sorted(validation, key=lambda k: (-validation[k], k))[0]
print('CHOSEN:', chosen_name)

xx, yy = np.meshgrid(np.linspace(-2, 3, 180),
                     np.linspace(-1.5, 2, 150))
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (label, m) in zip(axes, models.items()):
    z = m.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    ax.contourf(xx, yy, z.reshape(xx.shape),
                levels=np.linspace(0, 1, 11), cmap='coolwarm', alpha=.5)
    ax.scatter(X_va[:, 0], X_va[:, 1], c=y_va,
               cmap='coolwarm', edgecolors='k', s=15)
    ax.set(title=label, xlabel='x1', ylabel='x2')

fig.tight_layout()
fig.savefig('ex1_boundaries.png', dpi=160)

pred = models[chosen_name].predict(X_te)
print(f'TEST of the chosen model: {accuracy_score(y_te, pred):.1%}')
plt.show()