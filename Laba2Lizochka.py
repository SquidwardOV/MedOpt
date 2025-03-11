import math
import matplotlib.pyplot as plt
import numpy as np

# Функция цели: можно заменить на любую другую
def objective(u):
    return u ** 3 + 3 * u ** 2 - 3 * u + 1

# Функция для проверки выпуклости тройки точек (u_a, u_b, u_c)
def convex_triplet(u_a, u_b, u_c):
    J_a = objective(u_a)
    J_b = objective(u_b)
    J_c = objective(u_c)
    M = J_a - J_b
    K = J_c - J_b
    return (M >= 0) and (K >= 0) and ((M + K) > 0)

# Основной алгоритм поиска минимума с использованием параболической интерполяции
def modified_minimizer(lb, ub, delta):
    # Запрашиваем начальное приближение x0 у пользователя
    x0 = float(input("Введите начальное приближение x0: "))
    # Если введена граничная точка, корректируем x0 так, чтобы он находился строго внутри интервала
    if x0 <= lb:
        x0 = lb + delta
    elif x0 >= ub:
        x0 = ub - delta

    # Обеспечиваем условие: delta < (ub - lb)/2
    if delta >= (ub - lb) / 2:
        delta = (ub - lb) / 4

    samples = []  # Список сгенерированных кандидатов
    f0 = objective(x0)
    samples.append(x0)

    # Первоначальный кандидат x1 = x0 + delta
    x1 = x0 + delta
    f1 = objective(x1)
    samples.append(x1)

    # Определяем направление поиска на основе сравнения f(x0) и f(x1)
    if f1 <= f0:
        direction = 1
        x2 = x0 + 2 * delta
    else:
        direction = -1
        # Корректируем стартовую точку для поиска влево
        x0 = x0 + delta
        x1 = x0 - delta
        x2 = x0 - 2 * delta
        # Перезаписываем список кандидатов в возрастающем порядке
        samples = [x2, x1, x0]
    samples.append(x2)

    # Генерируем последующие точки с экспоненциальным увеличением шага
    k = 3  # номер следующего кандидата
    vertex = None  # вершина параболы, если найдётся выпуклая тройка
    while True:
        # Если имеется как минимум тройка точек, проверяем их выпуклость
        if len(samples) >= 3:
            a_, b_, c_ = samples[-3], samples[-2], samples[-1]
            if convex_triplet(a_, b_, c_):
                # Вычисляем вершину параболы по формуле:
                # w = -0.5 * ((J(b)-J(a))*c^2 + (J(a)-J(c))*b^2 + (J(c)-J(b))*a^2) /
                #          ((J(a)-J(b))*c + (J(c)-J(a))*b + (J(b)-J(c))*a)
                J_a, J_b, J_c = objective(a_), objective(b_), objective(c_)
                num = (J_b - J_a) * c_ ** 2 + (J_a - J_c) * b_ ** 2 + (J_c - J_b) * a_ ** 2
                den = (J_a - J_b) * c_ + (J_c - J_a) * b_ + (J_b - J_c) * a_
                if abs(den) < 1e-14:
                    vertex = b_
                else:
                    vertex = -0.5 * (num / den)
                break  # найдено решение, выходим из цикла
        # Если выпуклая тройка не найдена, генерируем новый кандидат
        if direction == 1:
            new_sample = samples[0] + delta * (2 ** (k - 1))
        else:
            new_sample = x0 - delta * (2 ** (k - 1))
        # Если новый кандидат выходит за пределы интервала, берем крайнее значение
        if new_sample < lb or new_sample > ub:
            vertex = ub if direction == 1 else lb
            break
        samples.append(new_sample)
        k += 1

    # Итоговый минимум – это точка с наименьшим значением функции среди всех кандидатов и вершины
    all_points = samples.copy()
    if vertex is not None:
        all_points.append(vertex)
    best_u = min(all_points, key=lambda u: objective(u))
    best_val = objective(best_u)
    return best_u, best_val, samples, vertex

if __name__ == "__main__":
    lower, upper = -1, 2
    delta = (upper - lower) / 10  # например, 0.3 при [a,b]=[-1,2]

    optimum, opt_val, candidates, parab_vertex = modified_minimizer(lower, upper, delta)

    print("Найденный минимум:")
    print(f"u* ≈ {optimum:.6f}, f(u*) ≈ {opt_val:.6f}")

    # Подготовка данных для графика
    x_grid = np.linspace(lower - 0.5, upper + 0.5, 400)
    y_grid = [objective(x) for x in x_grid]

    plt.figure(figsize=(8, 6))
    plt.plot(x_grid, y_grid, linewidth=2, label='objective(u)')

    # Отображаем кандидатов: синие крестики
    for idx, pt in enumerate(candidates):
        plt.plot(pt, objective(pt), 'kx', markersize=8)
        plt.text(pt, objective(pt), f" {idx}", fontsize=9, color='brown')

    # Итоговый минимум отображаем зелёным квадратом
    plt.plot(optimum, opt_val, 'gs', markersize=10, label='Найденная точка минимума')

    plt.xlabel("u")
    plt.ylabel("objective(u)")
    plt.title("График функции с найденной точкой минимума")
    plt.legend()
    plt.grid(True)
    plt.show()
