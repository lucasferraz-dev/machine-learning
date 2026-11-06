import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
nomes = np.array(["Aurora", "Brisa", "Calma", "Duna",
"Eco", "Festa", "Giro", "Horizonte"])
X = np.array([[2, 70], [3, 80], [4, 90], [5, 95],
[6, 110], [7, 125], [8, 140], [9, 150]])
y = np.array([0, 0, 0, 1, 0, 1, 1, 1])
consulta = np.array([[6.2, 100]])

modelo = make_pipeline(
  StandardScaler(),
  KNeighborsClassifier(n_neighbors=3))
modelo.fit(X, y)
rotulos = np.array(["relaxar", "treinar"])
print("Previsao:", rotulos[modelo.predict(consulta)[0]])
print("Votos relativos:", modelo.predict_proba(consulta)[0])

z = modelo[0].transform(consulta)
dist, idx = modelo[1].kneighbors(z)
for d, i in zip(dist[0], idx[0]):
  print(nomes[i], rotulos[y[i]], round(float(d), 3))

for k in [1, 3, 5, 7]:
  bruto = KNeighborsClassifier(n_neighbors=k).fit(X, y)
  padrao = make_pipeline(StandardScaler(),
  KNeighborsClassifier(n_neighbors=k)).fit(X, y)
  a = rotulos[bruto.predict(consulta)[0]]
  b = rotulos[padrao.predict(consulta)[0]]
  print(f"K={k}: sem escala={a}; com escala={b}")

  plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm",
edgecolors="black", s=85, vmin=0, vmax=1)
for nome, ponto in zip(nomes, X):
  plt.annotate(nome, ponto, xytext=(4, 4),
textcoords="offset points", fontsize=8)
plt.scatter(*consulta[0], marker="*", s=240,
c="gold", edgecolors="black", label="Consulta")
plt.xlabel("Energia (0-10)")
plt.ylabel("Andamento (batidas/minuto)")
plt.legend()
plt.tight_layout()
plt.savefig("musicas.png", dpi=160)
plt.show()