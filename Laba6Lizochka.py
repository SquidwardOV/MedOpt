import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D


def objective_circle(point):
    """
    Circular objective: f(x1, x2) = x1^2 + x2^2
    Unconstrained minimum at (0,0), violates constraint g < 0
    """
    x1, x2 = point
    return x1**2 + x2**2


def objective_shifted(point):
    """
    Shifted circular objective: f(x1, x2) = (x1 + 2)^2 + (x2 + 3)^2
    Unconstrained minimum at (-2,-3), violates constraint g < 0
    """
    x1, x2 = point
    return (x1 + 2)**2 + (x2 + 3)**2


def constraint_linear(point):
    """
    Linear constraint: g(x1, x2) = x1 + x2 - 1 >= 0
    """
    x1, x2 = point
    return x1 + x2 - 1


def penalty_term(point, coeff, constraint_fun):
    """
    Quadratic penalty: coeff * max(0, -g(x))^2
    """
    return coeff * max(0, -constraint_fun(point))**2


def penalized_objective(point, coeff, objective_fun, constraint_fun):
    """
    Combined objective with penalty term
    """
    return objective_fun(point) + penalty_term(point, coeff, constraint_fun)


def penalty_method(objective_fun, constraint_fun, initial_coeff=1.0, tol=1e-6, max_iter=100):
    """
    Quadratic penalty method:
      1. Начальное приближение [0,0], коэффициент штрафа initial_coeff.
      2. На каждой итерации минимизировать penalized_objective.
      3. Увеличивать coeff до выполнения g(x)>=0 внутри tol.
    """
    current_point = np.array([0.0, 0.0])
    coeff = initial_coeff
    history = []

    for iteration in range(1, max_iter + 1):
        result = minimize(
            lambda pt: penalized_objective(pt, coeff, objective_fun, constraint_fun),
            current_point,
            method='BFGS'
        )
        current_point = result.x
        g_val = constraint_fun(current_point)
        history.append((current_point.copy(), coeff, g_val))

        # Остановка, когда штраф небольшой
        if penalty_term(current_point, coeff, constraint_fun) < tol:
            break

        coeff *= 2

    return current_point, history


if __name__ == '__main__':
    # Предоставляем выбор функции
    print("Выберите целевую функцию:")
    print("1: x1^2 + x2^2 (Circle)")
    print("2: (x1+2)^2 + (x2+3)^2 (Shifted Circle)")
    choice = input("Введите 1 или 2: ")

    if choice.strip() == '2':
        objective_fun = objective_shifted
        label = 'Shifted Circle'
    else:
        objective_fun = objective_circle
        label = 'Circle'

    solution, history = penalty_method(objective_fun, constraint_linear)

    print("История итераций (точка, коэффициент, значение g):")
    for idx, (pt, coeff, g_val) in enumerate(history, start=1):
        print(f"Iter {idx}: Point={pt}, Coeff={coeff}, g={g_val}")

    # Визуализация
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    x_vals = [pt[0] for pt, _, _ in history]
    y_vals = [pt[1] for pt, _, _ in history]
    z_vals = [objective_fun(pt) for pt, _, _ in history]

    ax.plot(x_vals, y_vals, z_vals, marker='o', linestyle='-', color='blue', label='Path')
    ax.scatter(solution[0], solution[1], objective_fun(solution), color='red', s=50, label='Solution')

    # Граница ограничения
    x_line = np.linspace(-5, 5, 100)
    y_line = 1 - x_line
    z_line = [objective_fun((xi, yi)) for xi, yi in zip(x_line, y_line)]
    ax.plot(x_line, y_line, z_line, linestyle='--', color='green', label='Constraint')

    # Поверхность целевой функции
    X, Y = np.meshgrid(x_line, y_line)
    Z_obj = np.vectorize(lambda xi, yi: objective_fun((xi, yi)))(X, Y)
    ax.plot_surface(X, Y, Z_obj, alpha=0.3)

    # Поверхность ограничения g(x)
    Z_constr = X + Y - 1
    ax.plot_surface(X, Y, Z_constr, alpha=0.3, color='gray')

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('Objective')
    ax.set_title(f'Penalty Method for {label}')
    ax.legend()

    plt.show()
