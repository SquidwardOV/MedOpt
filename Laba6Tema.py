import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# --- Набор функций для выбора пользователем ---
functions = {
    1: {
        'desc': 'f(x) = x1^2 + x2^2 - x1*x2',
        'fun': lambda x: x[0]**2 + x[1]**2 - x[0]*x[1]
    },
    2: {
        'desc': 'f(x) = exp(x1 + x2)',
        'fun': lambda x: np.exp(x[0] + x[1])
    },
}

# --- Выбор функции пользователем ---
print("Выберите целевую функцию:")
for key, val in functions.items():
    print(f"  {key}: {val['desc']}")
try:
    choice = int(input("Номер функции (по умолчанию 1): ") or "1")
except ValueError:
    choice = 1
if choice not in functions:
    print("Некорректный выбор, используется функция 1.")
    choice = 1

objective = functions[choice]['fun']
print(f"\nИспользуемая функция: {functions[choice]['desc']}")

# --- Определение ограничения и штрафной функции ---
def constraint_value(x):
    """g(x) = x1 + x2 - 2 >= 0"""
    return x[0] + x[1] - 2

def penalized_function(x, penalty_coeff):
    """φ(x, r) = f(x) + r * max(0, -g(x))^2"""
    g = constraint_value(x)
    return objective(x) + penalty_coeff * max(0, -g)**2

# --- Метод штрафных функций ---
def solve_penalty(initial_x, initial_r=1.0, tol=1e-6, max_steps=50):
    x = np.array(initial_x, dtype=float)
    r = initial_r
    path = []
    f_path = []

    print("\n--- Начало метода штрафных функций ---")
    for step in range(1, max_steps+1):
        res = minimize(lambda v: penalized_function(v, r), x, method='BFGS')
        x = res.x
        g_val = constraint_value(x)
        violation = max(0, -g_val)

        path.append(x.copy())
        f_path.append(objective(x))

        print(f"Шаг {step:2d}: x = [{x[0]:.6f}, {x[1]:.6f}],  штраф r = {r:.2e},  нарушение = {violation:.6e}")

        if r * violation**2 < tol:
            print(f"  -> Остановлено: r·нарушение² = {r*violation**2:.2e} < tol ({tol})")
            break

        r *= 2

    print("--- Метод завершён ---\n")
    return np.array(path), np.array(f_path), x

# --- Запуск метода ---
initial_guess = [5, 5]
path, f_path, solution = solve_penalty(initial_guess)
sol_value = objective(solution)
print(f"Найденное решение: x = [{solution[0]:.6f}, {solution[1]:.6f}], f(x) = {sol_value:.6f}\n")

# --- Подготовка сетки для графиков ---
grid = np.linspace(-1, 5, 200)
X, Y = np.meshgrid(grid, grid)

# Векторизованное вычисление Z_obj
vec_f = np.vectorize(lambda xi, yi: objective([xi, yi]))
Z_obj = vec_f(X, Y)

# Плоскость ограничения
Z_constr = X + Y - 2

# --- Построение 3D-графика ---
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Поверхность целевой функции
ax.plot_surface(X, Y, Z_obj, alpha=0.4, cmap='viridis', edgecolor='none')
# Поверхность ограничения
ax.plot_surface(X, Y, Z_constr, alpha=0.4, cmap='plasma', edgecolor='none')

# Траектория итераций
ax.plot(
    path[:,0], path[:,1], f_path,
    '-o', color='red', linewidth=2, markersize=5
)
# Финальная точка
ax.scatter(
    solution[0], solution[1], sol_value,
    color='blue', s=80, marker='X'
)

# Настройка осей и заголовка
ax.set_xlabel('x₁')
ax.set_ylabel('x₂')
ax.set_zlabel('f(x)')
ax.set_title('Метод штрафных функций: поверхность и траектория')

# Легенда через proxy-элементы
proxy1 = mpatches.Patch(color=plt.cm.viridis(0.6), label='Целевая функция')
proxy2 = mpatches.Patch(color=plt.cm.plasma(0.6), label='Плоскость ограничения')
proxy3 = Line2D([0],[0], color='red', marker='o', label='Траектория')
proxy4 = Line2D([0],[0], color='blue', marker='X', linestyle='None', label='Решение')
ax.legend(handles=[proxy1, proxy2, proxy3, proxy4], loc='best')

plt.tight_layout()
plt.show()
