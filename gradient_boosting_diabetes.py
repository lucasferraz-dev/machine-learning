import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
D = load_diabetes(scaled=False)
X_dev, X_te, y_dev, y_te = train_test_split(
D.data, D.target, test_size=.20, random_state=42)
X_tr, X_va, y_tr, y_va = train_test_split(
X_dev, y_dev, test_size=.25, random_state=42)
print('Tamanhos:', len(y_tr), len(y_va), len(y_te))
# Ordem fixa tambem resolve empates exatos: primeiro candidato listado.
configs = [(eta, n, d) for eta in [.05, .10]
for n in [50, 150] for d in [1, 2]]

modelos, erros = [], []
for eta, n, depth in configs:
  m = GradientBoostingRegressor(learning_rate=eta, n_estimators=n,
  max_depth=depth, random_state=42)
  m.fit(X_tr, y_tr)
  tr = np.sqrt(mean_squared_error(y_tr, m.predict(X_tr)))
  va = np.sqrt(mean_squared_error(y_va, m.predict(X_va)))
  modelos.append(m); erros.append(va)
  print(f'eta={eta:.2f}; n={n:3}; d={depth}; treino={tr:.2f}; val={va:.2f}')
i = int(np.argmin(erros))
melhor = modelos[i]
print('ESCOLHIDA:', configs[i])
base = DummyRegressor(strategy='mean').fit(X_tr, y_tr)
for nome, m in [('Media', base), ('Boosting escolhido', melhor)]:
  p = m.predict(X_te)
  rmse = np.sqrt(mean_squared_error(y_te, p))
  mae = mean_absolute_error(y_te, p)
  print(f'TESTE {nome}: RMSE={rmse:.2f}; MAE={mae:.2f}')

pred = melhor.predict(X_te)
residuos = y_te - pred
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(y_te, pred, alpha=.7)
lim = [min(y_te.min(), pred.min()), max(y_te.max(), pred.max())]
axes[0].plot(lim, lim, 'r--')
axes[0].set(xlabel='Real', ylabel='Previsto', title='Teste: previsao')
axes[1].scatter(pred, residuos, alpha=.7)
axes[1].axhline(0, color='red', linestyle='--')
axes[1].set(xlabel='Previsto', ylabel='Real - previsto', title='Teste: residuos')
fig.tight_layout(); fig.savefig('gb_ex2_diagnostico.png', dpi=160)
print('Media dos residuos:', round(residuos.mean(), 2))
plt.show()