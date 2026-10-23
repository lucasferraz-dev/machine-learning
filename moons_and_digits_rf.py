import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

D = load_digits()
X_dev, X_te, y_dev, y_te = train_test_split(
    D.data, D.target, test_size=.20, stratify=D.target, random_state=42)
X_tr, X_va, y_tr, y_va = train_test_split(
    X_dev, y_dev, test_size=.25, stratify=y_dev, random_state=42)

print('Sizes:', len(y_tr), len(y_va), len(y_te))
configs = [(6, 3), (6, 1), (None, 3), (None, 1)]

models_list, accuracies = [], []
for depth, leaf in configs:
    m = RandomForestClassifier(n_estimators=120, max_depth=depth,
                               min_samples_leaf=leaf, max_features='sqrt', random_state=42)
    m.fit(X_tr, y_tr)
    tr = accuracy_score(y_tr, m.predict(X_tr))
    va = accuracy_score(y_va, m.predict(X_va))
    models_list.append(m)
    accuracies.append(va)
    print(f'depth={str(depth):4}; leaf={leaf}; train={tr:.3f}; val={va:.3f}')

best_idx = int(np.argmax(accuracies))
best_model = models_list[best_idx]
print('CHOSEN CONFIG:', configs[best_idx])

baseline_tree = DecisionTreeClassifier(max_depth=6, random_state=42).fit(X_tr, y_tr)
for name, m in [('Baseline Tree', baseline_tree), ('Chosen Forest', best_model)]:
    print(f'TEST {name}: {accuracy_score(y_te, m.predict(X_te)):.3f}')

pred = best_model.predict(X_te)
ConfusionMatrixDisplay.from_predictions(y_te, pred,
                                        labels=np.arange(10), cmap='Blues', colorbar=False)
plt.title('Test: actual labels in rows, predicted in columns')
plt.tight_layout()
plt.savefig('rf_ex2_confusion.png', dpi=160)

errors = np.flatnonzero(pred != y_te)[:8]
fig, axes = plt.subplots(2, 4, figsize=(9, 5))
for ax in axes.ravel():
    ax.axis('off')
for ax, j in zip(axes.ravel(), errors):
    ax.imshow(X_te[j].reshape(8, 8), cmap='gray_r')
    ax.set_title(f'Actual {y_te[j]} / Pred {pred[j]}')

fig.tight_layout()
plt.savefig('rf_ex2_errors.png', dpi=160)
plt.show()