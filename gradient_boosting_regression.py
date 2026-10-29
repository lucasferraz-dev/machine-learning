import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
rng = np.random.default_rng(42)
X = rng.uniform(0, 10, size=(800, 1))
y = 20 + 3*X[:, 0] + 10*np.sin(X[:, 0]) + rng.normal(0, 3, 800)
X_dev, X_te, y_dev, y_te = train_test_split(
X, y, test_size=.20, random_state=42)
X_tr, X_va, y_tr, y_va = train_test_split(
X_dev, y_dev, test_size=.25, random_state=42)

modelos, erros = {}, {}
for n in [5, 20, 50, 100, 200]:
  m = GradientBoostingRegressor(n_estimators=n,
  learning_rate=.1, max_depth=2, random_state=42)
  m.fit(X_tr, y_tr)
  rmse_tr = np.sqrt(mean_squared_error(y_tr, m.predict(X_tr)))
  rmse_va = np.sqrt(mean_squared_error(y_va, m.predict(X_va)))
  modelos[n], erros[n] = m, rmse_va
  print(f'{n:3d} etapas: RMSE treino={rmse_tr:.2f}; val={rmse_va:.2f}')
# Menor RMSE de validacao; empate exato: menos etapas.
n_star = min(erros, key=lambda n: (erros[n], n))
melhor = modelos[n_star]
print('ETAPAS ESCOLHIDAS:', n_star)

base = DummyRegressor(strategy='mean').fit(X_tr, y_tr)
for nome, m in [('Media do treino', base), ('Boosting escolhido', melhor)]:
  rmse = np.sqrt(mean_squared_error(y_te, m.predict(X_te)))
  print(f'TESTE {nome}: RMSE={rmse:.2f}')
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(list(erros), list(erros.values()), marker='o')
axes[0].scatter(n_star, erros[n_star], c='red', s=100, zorder=5)
axes[0].set(xlabel='Etapas', ylabel='RMSE', title='Validacao')
x_grid = np.linspace(0, 10, 300).reshape(-1, 1)
axes[1].scatter(X_va[:, 0], y_va, s=12, alpha=.5, label='Validacao')
for n in [5, n_star]:
  axes[1].plot(x_grid[:, 0], modelos[n].predict(x_grid),
label=f'{n} etapas')
axes[1].set(xlabel='x', ylabel='y', title='Previsoes do boosting')
axes[1].legend(); fig.tight_layout()
fig.savefig('ex2_boosting.png', dpi=160)
plt.show()