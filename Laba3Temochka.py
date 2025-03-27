import numpy as np
import matplotlib.pyplot as plt



def f(x):
    #return (x - 3) ** 2 + 2
    return x**6+5*x**4+x**2-1

# --- 2. Численное вычисление производных ---
def numerical_derivative(func, x, h=1e-5):
    """Приближённое вычисление первой производной f'(x)."""
    return (func(x + h) - func(x - h)) / (2 * h)


def numerical_second_derivative(func, x, h=1e-5):
    """Приближённое вычисление второй производной f''(x)."""
    return (func(x + h) - 2 * func(x) + func(x - h)) / (h ** 2)


# --- Метод Ньютона для поиска минимума f(x) (т.е. решения f'(x) = 0) ---
def newton_minimize_numeric(func, x0, eps=1e-6, max_iter=50):
    """
    Возвращает список итерационных значений x_k, найденных методом Ньютона.
    Метод останавливается, когда |f'(x_k)| < eps или превышено max_iter.
    """
    x_values = [x0]
    for _ in range(max_iter):
        xk = x_values[-1]
        df_val = numerical_derivative(func, xk)
        ddf_val = numerical_second_derivative(func, xk)

        if abs(df_val) < eps:
            break

        if ddf_val == 0:
            raise ValueError(f"Вторая производная равна нулю в точке x={xk:.6f}.")

        # Формула метода Ньютона
        x_new = xk - df_val / ddf_val
        x_values.append(x_new)

    return x_values


if __name__ == "__main__":
    # Начальное приближение и точность
    x0 = 1.0
    eps = 1e-6

    # Выполняем метод Ньютона
    x_iters = newton_minimize_numeric(f, x0, eps=eps)
    x_min = x_iters[-1]
    f_min = f(x_min)

    # Печатаем найденные промежуточные значения
    print("Итерационные точки метода Ньютона:")
    for i, xk in enumerate(x_iters):
        fxk = f(xk)
        dfxk = numerical_derivative(f, xk)
        print(f"Итерация {i}: x = {xk:.6f}, f(x) = {fxk:.6f}, f'(x) = {dfxk:.6f}")

    print("\nНайденный минимум:")
    print(f"x* = {x_min:.6f}")
    print(f"f(x*) = {f_min:.6f}")
    print(f"Число итераций: {len(x_iters) - 1}")

    # --- Подготовка данных для построения графика ---
    # Определим диапазон x с небольшим запасом вокруг всех итерационных точек

    left_bound = min(x_iters) - 1
    right_bound = max(x_iters) + 1
    x_range = np.linspace(left_bound, right_bound, 400)

    f_vals = [f(x) for x in x_range]
    fprime_vals = [numerical_derivative(f, x) for x in x_range]

    # --- 6. Построение "всего на одном графике" ---
    plt.figure(figsize=(10, 6))

    # (1) График функции f(x)
    plt.plot(x_range, f_vals, label="f(x)")

    # (2) График производной f'(x)
    plt.plot(x_range, fprime_vals, label="f'(x)")

    # (3) Горизонтальная линия y=0
    plt.axhline(0, linestyle="-", linewidth=0.8, color='gray')

    # (4) Итерационные точки и касательные к f'(x)
    for i, xk in enumerate(x_iters):
        # Значения в точке
        fxk = f(xk)
        dfxk = numerical_derivative(f, xk)
        ddfxk = numerical_second_derivative(f, xk)

        # Отмечаем точку на f(x)
        plt.plot(xk, fxk, 'o', markersize=5)  # точка на графике f(x)

        # Отмечаем точку на f'(x)
        plt.plot(xk, dfxk, 'x', markersize=5)  # точка на графике f'(x)

        # Касательная к f'(x) в точке xk:  y = f'(xk) + f''(xk)*(x - xk)
        tangent_line = dfxk + ddfxk * (x_range - xk)
        plt.plot(x_range, tangent_line, '--', linewidth=0.8)

        # Можно подписать номер итерации возле точки:
        plt.text(xk, fxk, f"k={i}", fontsize=8, color='blue')

    # (5) Отмечаем финальную точку минимума (по f(x)) и корень (по f'(x))
    plt.plot(x_min, f_min, 'ro', label="Найденный минимум f(x)")
    plt.plot(x_min, numerical_derivative(f, x_min), 'r*', label="Корень f'(x)")

    # Настройки графика
    plt.title("Метод Ньютона: f(x), f'(x), итерационные точки и касательные (на одном графике)")
    plt.xlabel("x")
    plt.ylabel("Значения f(x) и f'(x)")
    plt.legend()
    plt.grid(True)

    # Показываем всё сразу
    plt.show()
