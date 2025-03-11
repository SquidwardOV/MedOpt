import math
import matplotlib.pyplot as plt
import numpy as np

# Тестовая функция
def f(u):
    return u ** 3 + 3 * u ** 2 - 3 * u + 1
    #return 1/(u**2+u-2)

# Проверка выпуклости тройки точек для функции f(u)
# Тройка (u1, u2, u3) выпуклая, если:
# M = f(u1) - f(u2) >= 0,  K = f(u3) - f(u2) >= 0,  и M+K > 0.
def is_convex(u1, u2, u3, f):
    J1, J2, J3 = f(u1), f(u2), f(u3)
    M = J1 - J2
    K = J3 - J2
    return (M >= 0) and (K >= 0) and ((M + K) > 0)

# Основная функция, реализующая описанный алгоритм поиска минимума
def parabolic_search(a, b, step, f):
    # Выбираем начальную точку u0 в [a,b] (теперь вводится пользователем)
    u0 = float(input("Введите начальное приближение u0: "))

    # Гарантируем, что шаг меньше (b-a)/2
    if step >= (b - a) / 2:
        step = (b - a) / 4

    # Шаг 1: задаём u1 = u0 + step (убедимся, что u1 лежит в [a,b])
    if u0 + step > b:
        u0 = a  # альтернативно, можно взять u0 = a
    u1 = u0 + step
    J0, J1 = f(u0), f(u1)

    candidates = []  # список кандидатов (точек u_i)
    direction = None  # направление поиска: +1 для вправо, -1 для влево

    # Шаг 2: определяем направление поиска
    if J1 <= J0:
        # Движемся вправо: выбираем u0, u1, u2 = u0, u0+step, u0+2*step
        direction = 1
        u2 = u0 + 2 * step
        candidates = [u0, u1, u2]
    else:
        # Движемся влево:
        # По алгоритму: обновляем u0 = u0 + step, u1 = u1 - step, u2 = u0 - 2*step.
        # Для удобства сортируем их по возрастанию.
        u0_new = u0 + step
        u1_new = u1 - step  # равен исходному u0
        u2 = u0_new - 2 * step
        candidates = [u2, u1_new, u0_new]  # гарантированно: u2 < u1_new < u0_new
        direction = -1
        # Для дальнейшей генерации новых точек в отрицательном направлении будем использовать обновлённое u0_new
        u0 = u0_new

    # Список для хранения всех точек (для поиска минимума)
    all_points = candidates.copy()

    # Шаг 3: генерируем дополнительные точки, пока они лежат в [a,b]
    # и пока не найдётся выпуклая тройка.
    i = 3  # индекс для следующей точки
    w = None  # точка, полученная по параболе
    while True:
        # Если имеем хотя бы тройку, проверяем её выпуклость:
        if len(candidates) >= 3:
            u1_c, u2_c, u3_c = candidates[-3], candidates[-2], candidates[-1]
            if is_convex(u1_c, u2_c, u3_c, f):
                # Выпуклая тройка найдена – вычисляем вершину параболы по заданной формуле:
                J1_c, J2_c, J3_c = f(u1_c), f(u2_c), f(u3_c)
                numerator = (J2_c - J1_c) * u3_c ** 2 + (J1_c - J3_c) * u2_c ** 2 + (J3_c - J2_c) * u1_c ** 2
                denominator = (J1_c - J2_c) * u3_c + (J3_c - J1_c) * u2_c + (J2_c - J3_c) * u1_c
                if abs(denominator) < 1e-14:
                    w = u2_c  # если знаменатель близок к 0, используем u2 как приближение
                else:
                    w = -0.5 * (numerator / denominator)
                break  # завершаем цикл, так как получили w

        # Если выпуклая тройка ещё не найдена, генерируем следующего кандидата
        if direction == 1:
            # В правом направлении согласно алгоритму:
            u_new = candidates[0] + step * (2 ** (i - 1))
        else:
            # В левом направлении:
            u_new = u0 - step * (2 ** (i - 1))
        # Проверяем, что u_new принадлежит [a,b]
        if u_new < a or u_new > b:
            # Если новая точка выходит за пределы интервала,
            # то по алгоритму w принимается равным соответствующей границе.
            w = b if direction == 1 else a
            break
        candidates.append(u_new)
        all_points.append(u_new)
        i += 1

    # Итог: выбираем точку с минимальным значением функции среди кандидатов и, если вычислено, точки w
    if w is not None:
        all_points.append(w)
    best_point = min(all_points, key=lambda u: f(u))
    best_value = f(best_point)
    return best_point, best_value, candidates, w


if __name__ == "__main__":
    # Задаём границы интервала [a,b] и шаг (шаг должен быть меньше (b-a)/2)
    a, b = -10, 10
    step = (b - a) / 10  # например, 0.3 если [a,b]=[-1,2]

    x_min, f_min, candidates, w = parabolic_search(a, b, step, f)
    print("Приближённый минимум найден:")
    print(f"u* ≈ {x_min:.6f}, f(u*) ≈ {f_min:.6f}")

    # Построение графика
    # Формируем сетку значений для отображения функции на расширенном интервале
    x_vals = np.linspace(a - 0.5, b + 0.5, 400)
    y_vals = [f(x) for x in x_vals]

    plt.figure(figsize=(8, 6))
    plt.plot(x_vals, y_vals, label="f(u)", color="blue")

    # Отображаем все кандидаты (точки, которые генерировались в процессе)
    for i, u_val in enumerate(candidates):
        plt.scatter(u_val, f(u_val), color="orange", s=50, zorder=5)
        plt.text(u_val, f(u_val), f" {i}", fontsize=8, color="green")

    # Если w получено, выделяем его красной точкой
    if w is not None:
        plt.scatter([w], [f(w)], color="red", s=100, zorder=10, label="w (вершина параболы)")

    # Отмечаем итоговый минимум, если он отличается от w
    plt.scatter([x_min], [f_min], color="magenta", s=100, zorder=10, label="Минимум f(u)")

    plt.xlabel("u")
    plt.ylabel("f(u)")
    plt.title("Поиск минимума функции по модифицированному алгоритму")
    plt.legend()
    plt.grid(True)
    plt.show()
