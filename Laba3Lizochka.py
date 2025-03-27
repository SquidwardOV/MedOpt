import numpy as np
import matplotlib.pyplot as plt


def target_function(x):
    return x**6-x**2-3*x+4


def approx_first_derivative(func, x_val, delta=1e-5):
    return (func(x_val + delta) - func(x_val - delta)) / (2 * delta)


def approx_second_derivative(func, x_val, delta=1e-5):
    return (func(x_val + delta) - 2 * func(x_val) + func(x_val - delta)) / (delta ** 2)


def newton_minimization(func, init_val, tol=1e-6, max_steps=50):
    """
    Реализует метод Ньютона для поиска минимума функции (нахождения корня первой производной).
    Возвращает список значений x на каждом шаге.
    """
    x_history = [init_val]
    for _ in range(max_steps):
        current_x = x_history[-1]
        first_deriv = approx_first_derivative(func, current_x)
        second_deriv = approx_second_derivative(func, current_x)

        if abs(first_deriv) < tol:
            break

        if second_deriv == 0:
            raise ValueError(f"Вторая производная равна нулю при x={current_x:.6f}")

        next_x = current_x - first_deriv / second_deriv
        x_history.append(next_x)
    return x_history


if __name__ == '__main__':
    # Исходное приближение и требуемая точность
    start_value = 2.0
    tolerance = 1e-6

    # Запуск метода Ньютона
    iterations = newton_minimization(target_function, start_value, tol=tolerance)
    optimum_x = iterations[-1]
    optimum_y = target_function(optimum_x)

    print("Итерационные значения метода Ньютона:")
    for idx, x_val in enumerate(iterations):
        print(
            f"Шаг {idx}: x = {x_val:.6f}, f(x) = {target_function(x_val):.6f}, f'(x) = {approx_first_derivative(target_function, x_val):.6f}")

    print("\nРезультаты:")
    print(f"Минимум найден по x = {optimum_x:.6f} с f(x) = {optimum_y:.6f}")
    print(f"Количество шагов: {len(iterations) - 1}")

    # Подготовка данных для построения графика
    margin = 1
    x_lower = min(iterations) - margin
    x_upper = max(iterations) + margin
    x_vals = np.linspace(x_lower, x_upper, 400)
    func_values = [target_function(x) for x in x_vals]
    deriv_values = [approx_first_derivative(target_function, x) for x in x_vals]

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, func_values, label="f(x)", color='purple')
    plt.plot(x_vals, deriv_values, label="f'(x)", color='orange')
    plt.axhline(0, linestyle="--", linewidth=0.8, color='black')

    # Отображение итерационных точек, их касательных и подписей
    for i, x_val in enumerate(iterations):
        y_val = target_function(x_val)
        d1_val = approx_first_derivative(target_function, x_val)
        d2_val = approx_second_derivative(target_function, x_val)
        plt.plot(x_val, y_val, 'o', markersize=6, color='green')  # точка на графике target_function
        plt.plot(x_val, d1_val, 'x', markersize=6, color='red')  # точка на графике первой производной

        # Касательная к графику первой производной в точке x_val
        tangent = d1_val + d2_val * (x_vals - x_val)
        plt.plot(x_vals, tangent, linestyle=":", linewidth=0.8, color='blue')

        plt.text(x_val, y_val, f" шаг {i}", fontsize=8, color='brown')

    # Отмечаем найденный минимум и корень производной
    plt.plot(optimum_x, optimum_y, 'ro', label="Найденный минимум", markersize=8)
    plt.plot(optimum_x, approx_first_derivative(target_function, optimum_x), 'o', label="Корень производной",
             markersize=10)

    plt.title("Метод Ньютона: Поиск минимума функции")
    plt.xlabel("x")
    plt.ylabel("Значения функции и её производной")
    plt.legend()
    plt.grid(True)
    plt.show()
