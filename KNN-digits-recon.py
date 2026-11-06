from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import (
train_test_split, StratifiedKFold, GridSearchCV)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score,
classification_report, ConfusionMatrixDisplay)

saida = Path("resultados_digitos")
saida.mkdir(exist_ok=True)
dados = load_digits()
X, y = dados.data, dados.target
Xtr, Xte, ytr, yte = train_test_split(
X, y, test_size=0.25, stratify=y, random_state=42)

pipe = Pipeline([
("escala", "passthrough"),
("knn", KNeighborsClassifier())])
grade = {
"escala": ["passthrough", StandardScaler()],
"knn__n_neighbors": [1, 3, 5, 7, 9],
"knn__weights": ["uniform", "distance"],
"knn__p": [1, 2]}
cv = StratifiedKFold(n_splits=5, shuffle=True,
random_state=42)
busca = GridSearchCV(pipe, grade, cv=cv,
scoring="accuracy", n_jobs=1, refit=True)
busca.fit(Xtr, ytr)
print("Configuracao:", busca.best_params_)
print("Acuracia media na CV:", busca.best_score_)

melhor = busca.best_estimator_
pred = melhor.predict(Xte)
base = DummyClassifier(strategy="most_frequent")
base.fit(Xtr, ytr)
print("Acuracia baseline:", base.score(Xte, yte))
print("Acuracia KNN:", accuracy_score(yte, pred))
print(classification_report(yte, pred, digits=3, zero_division=0))
ConfusionMatrixDisplay.from_predictions(yte, pred, cmap="Purples", colorbar=False)
plt.tight_layout()
plt.savefig(saida / "confusao.png", dpi=160)
plt.show()

erros = np.flatnonzero(pred != yte)[:8]
fig, axs = plt.subplots(2, 4, figsize=(9, 5))
for ax in axs.ravel():
  ax.axis("off")
for ax, i in zip(axs.ravel(), erros):
  ax.imshow(Xte[i].reshape(8, 8), cmap="gray_r",
  vmin=0, vmax=16)
  ax.set_title(f"Real {yte[i]} / previsto {pred[i]}")
fig.tight_layout()
fig.savefig(saida / "erros.png", dpi=160)
plt.close(fig)

q = np.array([
[0, 0, 8, 13, 14, 5, 0, 0],
[0, 3, 16, 7, 10, 15, 0, 0],
[0, 0, 2, 0, 10, 13, 0, 0],
[0, 0, 0, 8, 16, 3, 0, 0],
[0, 0, 0, 10, 15, 3, 0, 0],
[0, 0, 0, 0, 8, 14, 0, 0],
[0, 0, 9, 6, 12, 15, 0, 0],
[0, 0, 10, 16, 13, 5, 0, 0]], dtype=float)
print("Consulta desenhada:", melhor.predict(q.reshape(1, -1)))

escala = melhor.named_steps["escala"]
qv = q.reshape(1, -1)
if escala != "passthrough":
  qv = escala.transform(qv)
dist, idx = melhor.named_steps["knn"].kneighbors(qv)
print("Rotulos vizinhos:", ytr[idx[0]])
print("Distancias:", np.round(dist[0], 3))

fig, axs = plt.subplots(1, len(idx[0]) + 1,
figsize=(2 * (len(idx[0]) + 1), 2.6))
axs[0].imshow(q, cmap="gray_r", vmin=0, vmax=16)
axs[0].set_title("Consulta")
for ax, i, d in zip(axs[1:], idx[0], dist[0]):
  ax.imshow(Xtr[i].reshape(8, 8), cmap="gray_r",
  vmin=0, vmax=16)
  ax.set_title(f"Classe {ytr[i]}\nd={d:.2f}")
for ax in axs:
  ax.axis("off")
fig.tight_layout()
fig.savefig(saida / "vizinhos.png", dpi=160)
plt.close(fig)