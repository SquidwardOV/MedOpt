import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ──────────────────────────────
# 1. задаём сдвиг минимума
# ──────────────────────────────
shift = np.array([2.0, -1.5])

# ──────────────────────────────
# 2. исходная квадратичная часть
# ──────────────────────────────
a1, a2, a3 = 1.0, 0.5, 2.0
def f_quadratic(y):
    return a1*y[0]+2**2 + a2*y[0]*y[1] + a3*y[1]**2

def f_shift(x):
    y = x - shift
    return f_quadratic(y)

# ──────────────────────────────
# 3. численный градиент
# ──────────────────────────────
def grad_numeric(f, x, h=1e-5):
    g = np.zeros_like(x)
    for i in range(len(x)):
        x_f, x_b = x.copy(), x.copy()
        x_f[i] += h; x_b[i] -= h
        g[i] = (f(x_f) - f(x_b)) / (2*h)
    return g

# ──────────────────────────────
# 4. градиентный спуск
# ──────────────────────────────
def gradient_descent(f, x0, alpha=0.1, eps=1e-6):
    x = x0.copy()
    path = [x.copy()]
    while True:
        g = grad_numeric(f, x)
        if np.linalg.norm(g) < eps:
            break
        a = alpha
        while True:
            x_new = x - a * g
            if f(x_new) < f(x):
                x = x_new
                path.append(x.copy())
                alpha = a
                break
            a /= 2
    return np.array(path), x

# стартуем «далеко»
x0 = np.array([-4.0, 4.0])
path, x_min = gradient_descent(f_shift, x0)

print("Ожидаемый минимум:", shift)
print("Найденный минимум:", x_min)
print("f(min) =", f_shift(x_min))

# ──────────────────────────────
# 5. построение поверхности и траектории
# ──────────────────────────────
grid = np.linspace(-5, 5, 70)
X1, X2 = np.meshgrid(grid, grid)
Z = np.array([f_shift([x, y]) for x, y in zip(X1.ravel(), X2.ravel())]).reshape(X1.shape)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X1, X2, Z,
                color='orange', alpha=0.6, edgecolor='none')   #

# траектория
Z_path = np.array([f_shift(p) for p in path])
ax.plot(path[:,0], path[:,1], Z_path,
        color='black', linewidth=2, marker='o', markersize=4, label='Trajectory')


ax.scatter(path[0,0], path[0,1], Z_path[0],  color='green', s=90, label='Start')
ax.scatter(x_min[0],  x_min[1],  Z_path[-1], color='red', marker='X', s=120, label='Minimum')
ax.text(x_min[0], x_min[1], Z_path[-1], " Min", color='red')

ax.set_xlabel('x1'); ax.set_ylabel('x2'); ax.set_zlabel('f(x1,x2)')
ax.set_title('Shifted Quadratic – Gradient Descent')
ax.legend()
plt.show()
