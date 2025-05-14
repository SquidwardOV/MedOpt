import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 (нужен для 3-D проекции)

# ---------------------------- численные производные ----------------------------
def numerical_gradient(f, x, eps=1e-8):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x1 = x.copy(); x2 = x.copy()
        x1[i] += eps
        x2[i] -= eps
        grad[i] = (f(x1) - f(x2)) / (2 * eps)
    return grad


def numerical_hessian(f, x, eps=1e-5):
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            x_ijp   = x.copy(); x_ijm   = x.copy()
            x_ipj_m = x.copy(); x_imj_p = x.copy()

            x_ijp[i]   += eps; x_ijp[j]   += eps
            x_ijm[i]   -= eps; x_ijm[j]   -= eps
            x_ipj_m[i] += eps; x_ipj_m[j] -= eps
            x_imj_p[i] -= eps; x_imj_p[j] += eps

            H[i, j] = (f(x_ijp) - f(x_ipj_m) - f(x_imj_p) + f(x_ijm)) / (4 * eps**2)
            H[j, i] = H[i, j]
    return H


# ------------------------------ метод Ньютона ---------------------------------
def newton_minimize(f, x0, tol=1e-6, max_iter=50, alpha0=1.0, beta=0.5, c1=1e-4):
    x = x0.copy().astype(float)
    history = [x.copy()]
    grad_norms = []

    for k in range(max_iter):
        g = numerical_gradient(f, x)
        H = numerical_hessian(f, x)

        grad_norms.append(np.linalg.norm(g))
        if grad_norms[-1] < tol:
            break

        try:
            p = -np.linalg.solve(H, g)          # направление Ньютона
        except np.linalg.LinAlgError:            # вырожденный гессиан
            p = -g / grad_norms[-1]

        # backtracking-линия поиска (условие Армишоу)
        alpha = alpha0
        while f(x + alpha * p) > f(x) + c1 * alpha * np.dot(g, p):
            alpha *= beta

        x = x + alpha * p
        history.append(x.copy())

    return np.array(history), np.array(grad_norms)


# ------------------------------ функции графиков ------------------------------
def plot_contour_and_path(f, history, xlim=(-5, 5), ylim=(-5, 5)):
    x1_vals = np.linspace(*xlim, 400)
    x2_vals = np.linspace(*ylim, 400)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = f(np.vstack([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)

    plt.figure(figsize=(6, 5))
    cs = plt.contour(X1, X2, Z, levels=30, cmap='viridis')
    plt.clabel(cs, inline=1, fontsize=8)

    # линия траектории
    plt.plot(history[:, 0], history[:, 1], '-', color='red', lw=1.2, label='Траектория')

    # все промежуточные точки + подписи-номера
    for i, (x1, x2) in enumerate(history):
        plt.scatter(x1, x2, c='red', s=40, zorder=3)
        plt.text(x1 + 0.1, x2 + 0.1, str(i), fontsize=8, color='red')

    # финиш
    plt.scatter(history[-1, 0], history[-1, 1], c='blue', s=70, zorder=4, label='Минимум')

    plt.title('Контур и все итерации Ньютона')
    plt.xlabel('$x_1$'); plt.ylabel('$x_2$')
    plt.legend(); plt.grid(True)


def plot_surface_and_path(f, history, xlim=(-5, 5), ylim=(-5, 5)):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    x1_vals = np.linspace(*xlim, 100)
    x2_vals = np.linspace(*ylim, 100)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)
    Z = f(np.vstack([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)
    ax.plot_surface(X1, X2, Z, rstride=5, cstride=5, alpha=0.6, cmap='viridis')

    # линия-траектория
    path_z = f(history)
    ax.plot(history[:, 0], history[:, 1], path_z, '-', color='red', lw=1.2, label='Траектория')

    # все точки и подписи
    for i, (x1, x2, z) in enumerate(zip(history[:, 0], history[:, 1], path_z)):
        ax.scatter(x1, x2, z, c='red', s=40)
        ax.text(x1, x2, z, f'{i}', fontsize=8, color='red')

    ax.scatter(history[-1, 0], history[-1, 1], path_z[-1],
               c='blue', s=70, label='Минимум')

    ax.set_title('3-D поверхность и все итерации Ньютона')
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$'); ax.set_zlabel('$f(x)$')
    ax.legend(); ax.grid(True)


def plot_grad_convergence(grad_norms):
    plt.figure(figsize=(5, 4))
    plt.semilogy(grad_norms, 'o-')
    plt.title('Сходимость $\\|\\nabla f\\|$')
    plt.xlabel('Итерация')
    plt.ylabel('$$\\|\\nabla f\\|$$')
    plt.grid(True)


# ------------------------------ main-пример -----------------------------------
if __name__ == '__main__':
    # --------------------------------------------
    # Используем функцию Розенброка
    # f(x) = 100*(x2-x1^2)^2 + (1-x1)^2
    # --------------------------------------------
    def f(x):
        x1, x2 = x[..., 0], x[..., 1]
        return 100.0 * (x2 - x1 ** 2) ** 2 + (1.0 - x1) ** 2

    x0 = np.array([-1.2, 1.0])      # стартовое приближение
    history, grad_norms = newton_minimize(f, x0, max_iter=100)

    x_opt = history[-1]
    f_opt = f(x_opt)
    H_opt = numerical_hessian(f, x_opt)

    print('================= Результаты =================')
    print(f'Минимум x* = {x_opt}')
    print(f'f(x*) = {f_opt:.6g}')
    print('Гессиан в точке минимума:')
    print(H_opt)

    # Визуализация
    plot_contour_and_path(f, history, xlim=(-2, 2), ylim=(-1, 3))
    plot_surface_and_path(f, history, xlim=(-2, 2), ylim=(-1, 3))
    plt.show()
