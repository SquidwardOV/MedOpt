import math
import numpy as np
import matplotlib.pyplot as plt
from time import time


def J(x):

    return np.cos(x)


def bisection_method(a, b, epsilon, delta):

    iteration = 0
    history = []  # для хранения истории интервалов
    while (b - a) > epsilon:
        history.append((a, b))
        u1 = (a + b - delta) / 2
        u2 = (a + b + delta) / 2
        print(f"Итерация {iteration}: a = {a:.6f}, b = {b:.6f}")
        print(f"    u1 = {u1:.6f}, J(u1) = {J(u1):.6f}")
        print(f"    u2 = {u2:.6f}, J(u2) = {J(u2):.6f}")

        if J(u1) > J(u2):
            a = u1
        elif J(u1) < J(u2):
            b = u2
        else:
            # Если значения равны, обновляем интервал симметрично
            a = u1
            b = u2
        iteration += 1

    # Добавляем финальный интервал в историю
    history.append((a, b))

    # Проверяем итоговую длину интервала и вычисляем результат
    if (b - a) >= epsilon:
        u1 = (a + b - delta) / 2
        u2 = (a + b + delta) / 2
        print("\nИтоговый интервал [a, b] =", (a, b), f"с длиной {b - a:.6f} >= epsilon")
        result = {"u1": u1, "J(u1)": J(u1),
                  "u2": u2, "J(u2)": J(u2)}
        final_point = None  # не выделяем единую точку минимума
    else:
        u_star = (a + b) / 2
        print("\nИтоговый интервал [a, b] =", (a, b), f"с длиной {b - a:.6f} < epsilon")
        result = {"u*": u_star, "J(u*)": J(u_star)}
        final_point = u_star
    return result, history


def golden_section_search(a, b, epsilon, targ):
    """
    Метод золотого сечения с возможностью поиска минимума (targ=False)
    или максимума (targ=True).

    Параметры:
      a, b   - границы интервала,
      epsilon - требуемая точность,
      targ    - тип задачи: False - минимум, True - максимум.

    Используются формулы:
      x1 = a + ((3 - sqrt(5)) / 2) * (b - a)
      x2 = a + ((sqrt(5) - 1) / 2) * (b - a)

    Возвращает:
      method   - название метода,
      x        - искомая точка (среднее окончательного интервала),
      y        - значение функции J(x),
      fnt      - время работы алгоритма,
      kol      - число итераций,
      history  - список интервалов, выбранных на итерациях (для построения графика).
    """
    stt = time()

    # Вычисление начальных точек
    x1 = a + ((3 - math.sqrt(5)) / 2) * (b - a)
    x2 = a + ((math.sqrt(5) - 1) / 2) * (b - a)
    fx1, fx2 = J(x1), J(x2)
    history = []
    kol = 0

    while (b - a) > epsilon:
        history.append((a, b))
        print(f"Итерация {kol}: a = {a:.6f}, b = {b:.6f}")
        print(f"    x1 = {x1:.6f}, J(x1) = {fx1:.6f}")
        print(f"    x2 = {x2:.6f}, J(x2) = {fx2:.6f}")

        if not targ:  # поиск минимума
            if fx1 >= fx2:
                a = x1
                x1 = x2
                fx1 = fx2
                x2 = a + ((math.sqrt(5) - 1) / 2) * (b - a)
                fx2 = J(x2)
            else:
                b = x2
                x2 = x1
                fx2 = fx1
                x1 = a + ((3 - math.sqrt(5)) / 2) * (b - a)
                fx1 = J(x1)
        else:  # поиск максимума
            if fx1 < fx2:
                a = x1
                x1 = x2
                fx1 = fx2
                x2 = a + ((math.sqrt(5) - 1) / 2) * (b - a)
                fx2 = J(x2)
            else:
                b = x2
                x2 = x1
                fx2 = fx1
                x1 = a + ((3 - math.sqrt(5)) / 2) * (b - a)
                fx1 = J(x1)
        kol += 1

    fnt = time() - stt
    x = round((a + b) / 2, 4)
    y = round(J(x), 4)
    method_name = 'золотое сечение'
    return method_name, x, y, fnt, kol, history


def plot_search(history, final_point, a0, b0, method_name="Метод"):
    """
    Строит график функции J(x) на интервале [a0, b0] с отметками всех промежуточных интервалов.
    Если задана итоговая точка (final_point), она выделяется на графике.
    """
    margin = 0.1 * (b0 - a0)
    x_min = a0 - margin
    x_max = b0 + margin
    x_values = np.linspace(x_min, x_max, 500)
    y_values = [J(x) for x in x_values]

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label="J(x)", color="blue")

    # Отмечаем интервалы из истории
    for (a, b) in history:
        plt.axvline(a, color='grey', linestyle='--', alpha=0.3)
        plt.axvline(b, color='grey', linestyle='--', alpha=0.3)

    # Отмечаем итоговую точку
    if final_point is not None:
        plt.axvline(final_point, color='red', linestyle='-', label="Итоговая точка")
        plt.scatter([final_point], [J(final_point)], color='red', zorder=5)

    plt.title(f"График функции J(x) ({method_name})")
    plt.xlabel("x")
    plt.ylabel("J(x)")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    print("Выберите метод:")
    print("1 - Модифицированный метод бисекции")
    print("2 - Метод золотого сечения")
    method_choice = input("Ваш выбор (1 или 2): ").strip()

    a0 = float(input("Введите значение a: "))
    b0 = float(input("Введите значение b: "))
    epsilon = float(input("Введите значение epsilon: "))
    delta = float(input("Введите значение delta: "))

    if method_choice == "1":
        print("\nЗапуск модифицированного метода бисекции...\n")
        result, history = bisection_method(a0, b0, epsilon, delta)
        print("\nРезультат метода бисекции:")
        for key, value in result.items():
            print(f"{key} = {value}")
        final_point = result.get("u*", None)
        plot_search(history, final_point, a0, b0, method_name="Модифицированный метод бисекции")
    elif method_choice == "2":
        print("\nЗапуск метода золотого сечения...\n")
        print("Выберите тип задачи: 0 - минимум, 1 - максимум")
        targ_input = input("Ваш выбор (0 или 1): ").strip()
        targ = False if targ_input == "0" else True
        method_name, x, y, fnt, kol, history = golden_section_search(a0, b0, epsilon, targ)
        print("\nРезультат метода золотого сечения:")
        print(f"Метод: {method_name}")
        print(f"x = {x}, J(x) = {y}")
        print(f"Время работы: {fnt:.6f} сек, Итераций: {kol}")
        plot_search(history, x, a0, b0, method_name="Метод золотого сечения")
    else:
        print("Неверный выбор метода!")


if __name__ == "__main__":
    main()
