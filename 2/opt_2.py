import math
import numpy as np
import pandas as pd

alpha = 0
y0, z0 = 1, 4
x_start = np.array([1.0, 2.0, 5.0])

def rotated_coords(y, z):
    """Преобразование координат с поворотом и смещением"""
    y_p = (y - y0) * math.cos(alpha) + (z - z0) * math.sin(alpha)
    z_p = -(y - y0) * math.sin(alpha) + (z - z0) * math.cos(alpha)
    return y_p, z_p

def f3(x, y, z):
    """Целевая функция для минимизации"""
    y_p, z_p = rotated_coords(y, z)
    return 2 * x**4 + 2 * y_p**4 + z_p**2

def grad_f3(x, y, z):
    """Вычисление градиента целевой функции"""
    y_p, z_p = rotated_coords(y, z)
    dfdx = 8 * x**3
    dfdy_p = 8 * y_p**3
    dfdz_p = 2 * z_p
    dfdy = dfdy_p * math.cos(alpha) - dfdz_p * math.sin(alpha)
    dfdz = dfdy_p * math.sin(alpha) + dfdz_p * math.cos(alpha)
    return np.array([dfdx, dfdy, dfdz], dtype=float)

def numerical_hessian(f, x, eps=1e-6):
    """Численное вычисление матрицы Гессе для любой функции в точке x"""
    n = len(x)
    H = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # Вторая производная по i и j переменным
            x_pp = x.copy(); x_pp[i] += eps; x_pp[j] += eps
            x_pm = x.copy(); x_pm[i] += eps; x_pm[j] -= eps
            x_mp = x.copy(); x_mp[i] -= eps; x_mp[j] += eps
            x_mm = x.copy(); x_mm[i] -= eps; x_mm[j] -= eps
            
            H[i, j] = (f(*x_pp) - f(*x_pm) - f(*x_mp) + f(*x_mm)) / (4 * eps**2)
    
    return H

def hessian_f3(x, y, z):
    """Вычисление матрицы Гессе (вторых производных) численным методом"""
    point = np.array([x, y, z])
    return numerical_hessian(f3, point)

def golden_section(phi, a=-1, b=1, tol=1e-10):
    """Одномерная минимизация методом золотого сечения"""
    multiplier = (math.sqrt(5) - 1) / 2
    x1 = b - (b - a) * multiplier
    x2 = a + (b - a) * multiplier
    f1 = phi(x1)
    f2 = phi(x2)
    iterations = 0
    max_iter = 100
    while abs(b - a) > tol and iterations < max_iter:
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - multiplier * (b - a)
            f1 = phi(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + multiplier * (b - a)
            f2 = phi(x2)
        iterations += 1
    return (a + b) / 2

def coordinate_descent(x0, eps=1e-8, max_iter=1000):
    """Минимизация методом координатного спуска"""
    x = np.array(x0, dtype=float)
    for _ in range(max_iter):
        x_old = x.copy()
        # Поочередная оптимизация по каждой координате
        for i in range(3):
            def phi(alpha_val):
                x_temp = x.copy()
                x_temp[i] += alpha_val
                return f3(*x_temp)
            alpha_opt = golden_section(phi, -0.5, 0.5, tol=eps/10)
            x[i] += alpha_opt
        # Проверка условия сходимости
        if np.linalg.norm(x - x_old) < eps:
            break
    return x

def steepest_descent(x0, eps=1e-8, max_iter=1000):
    """Минимизация методом наискорейшего спуска"""
    x = np.array(x0, dtype=float)
    for _ in range(max_iter):
        grad = grad_f3(*x)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < eps:
            break
        # Направление антиградиента
        p = -grad / grad_norm
        def phi(alpha_val):
            return f3(*(x + alpha_val * p))
        max_step = min(1.0, 0.1 / (grad_norm + 1e-10))
        alpha_opt = golden_section(phi, 0, max_step, tol=eps/10)
        x = x + alpha_opt * p
    return x

def newton_method(x0, eps=1e-10, max_iter=1000):
    """Минимизация методом Ньютона"""
    x = np.array(x0, dtype=float)
    for _ in range(max_iter):
        grad = grad_f3(*x)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < eps:
            break
        H = hessian_f3(*x)
        # Проверка вырожденности матрицы Гессе
        det_H = np.linalg.det(H)
        if abs(det_H) < 1e-12:
            dx = -np.linalg.pinv(H) @ grad
        else:
            dx = -np.linalg.solve(H, grad)
        # Ограничение максимального шага
        step_norm = np.linalg.norm(dx)
        if step_norm > 10:
            dx = dx / step_norm * 10
        x = x + dx
    return x

coord_res = coordinate_descent(x_start)
steep_res = steepest_descent(x_start)
newton_res = newton_method(x_start)

data = []
methods = [
    ("Координатный спуск", coord_res),
    ("Наискорейший спуск", steep_res),
    ("Метод Ньютона", newton_res),
    ("Аналитический минимум", np.array([0.0, 1.0, 4.0]))
]

for name, res in methods:
    data.append({
        "Метод": name,
        "x": res[0],
        "y": res[1], 
        "z": res[2],
        "f(x,y,z)": f3(*res)
    })

df = pd.DataFrame(data)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 60)

print(df)