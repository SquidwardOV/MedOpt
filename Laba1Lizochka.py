import time
import math
import numpy as np
import matplotlib.pyplot as plt


# Пример функции. Измените по необходимости.
def f(x):
    return (x - 2) ** 2 + 1


# Метод золотого сечения для минимизации
def golden_ratio(a, b, e):
    stt = time.time()
    # Первоначальное вычисление двух точек по золотому сечению
    x1 = a + ((3 - math.sqrt(5)) / 2) * (b - a)
    x2 = a + ((math.sqrt(5) - 1) / 2) * (b - a)
    fx1, fx2 = f(x1), f(x2)
    kol = 0
    while (b - a > e):
        if fx1 >= fx2:
            a = x1
            x1 = x2
            fx1 = fx2
            x2 = a + ((math.sqrt(5) - 1) / 2) * (b - a)
            fx2 = f(x2)
        else:
            b = x2
            x2 = x1
            fx2 = fx1
            x1 = a + ((3 - math.sqrt(5)) / 2) * (b - a)
            fx1 = f(x1)
        kol += 1
    fnt = time.time() - stt
    x_opt = round((a + b) / 2, 4)
    y_opt = round(f(x_opt), 4)
    method_name = 'золотое сечение'
    return method_name, x_opt, y_opt, fnt, kol


# Метод деления отрезка (бывшая дихотомия) для минимизации
def segment_division(a, b, e, d):
    stt = time.time()
    kol = 0
    # Итерации продолжаются, пока длина интервала больше e и (b-a)/2 > d
    while (b - a > e) and (((b - a) / 2) > d):
        x1 = (a + b - d) / 2
        x2 = (a + b + d) / 2
        fx1, fx2 = f(x1), f(x2)
        if fx1 >= fx2:
            a = x1
        else:
            b = x2
        kol += 1
    fnt = time.time() - stt
    x_opt = round((a + b) / 2, 4)
    y_opt = round(f(x_opt), 4)
    method_name = 'метод деления отрезка'
    return method_name, x_opt, y_opt, fnt, kol


# Функция для построения графика с отмеченной точкой оптимума и вертикальной линией
def plot_function(f, init_a, init_b, optimal_x, optimal_y, method_name):
    margin = (init_b - init_a) * 0.1
    x_vals = np.linspace(init_a - margin, init_b + margin, 400)
    y_vals = [f(x) for x in x_vals]

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, label='f(x)', color='blue')
    plt.plot(optimal_x, optimal_y, 'ro', markersize=8, label='Оптимум')
    # Вертикальная линия через оптимальную точку
    plt.axvline(x=optimal_x, color='red', linestyle='--', label=f"Оптимальное x = {optimal_x}")

    plt.title(f"Оптимизация методом {method_name}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()


# Основная функция для взаимодействия с пользователем
def main():
    print("Решение задачи минимизации методом золотого сечения или методом деления отрезка")
    print("Функция f(x) определена как: (x-2)^2 + 1")
    try:
        # Ввод исходных параметров
        init_a = float(input("Введите начальное значение a: "))
        init_b = float(input("Введите начальное значение b: "))
        e = float(input("Введите точность e: "))
        d = float(input("Введите параметр d (для метода деления отрезка): "))

        print("\nВыберите метод оптимизации:")
        print("1 - Золотое сечение")
        print("2 - Метод деления отрезка")
        method_choice = input("Введите номер метода: ")

        # Используем копии исходных границ для работы методов
        a, b = init_a, init_b

        if method_choice == "1":
            method, x_opt, y_opt, fnt, kol = golden_ratio(a, b, e)
        elif method_choice == "2":
            method, x_opt, y_opt, fnt, kol = segment_division(a, b, e, d)
        else:
            print("Неверный выбор метода!")
            return

        print("\nРезультаты:")
        print("Метод:", method)
        print("Найденное значение x =", x_opt)
        print("Значение функции f(x) =", y_opt)
        print("Время работы: {:.6f} сек".format(fnt))
        print("Количество итераций:", kol)

        # Построение графика
        plot_function(f, init_a, init_b, x_opt, y_opt, method)

    except Exception as ex:
        print("Ошибка ввода:", ex)


if __name__ == "__main__":
    main()
