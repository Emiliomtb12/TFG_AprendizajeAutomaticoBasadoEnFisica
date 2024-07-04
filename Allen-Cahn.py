import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import deepxde as dde
from deepxde.backend import torch

# Parámetros para solve_ivp
L = 2.0  # longitud del dominio espacial
T = 1.0  # tiempo total
Nx = 50  # número de puntos espaciales
Nt = 50  # número de puntos temporales
alpha = 0.0001
dx = L / (Nx - 1)
x = np.linspace(-L/2, L/2, Nx)

# Condición inicial
u0 = x**2 * np.cos(np.pi * x)

# Función de la no linealidad
def nonlinearity(u):
    return 5 * u**3 - 5 * u

# Función para resolver la EDP
def edp(t, u):
    dudt = np.zeros_like(u)
    dudt[1:-1] = alpha * (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2 - nonlinearity(u[1:-1])
    # Condiciones de contorno periódicas
    dudt[0] = alpha * (u[1] - 2 * u[0] + u[-1]) / dx**2 - nonlinearity(u[0])
    dudt[-1] = alpha * (u[0] - 2 * u[-1] + u[-2]) / dx**2 - nonlinearity(u[-1])
    return dudt

# Resolver la EDP
solution = solve_ivp(edp, [0, T], u0, t_eval=np.linspace(0, T, Nt), method='RK45')

# Extraer la solución
u = solution.y
t = solution.t
X_sol, T_sol = np.meshgrid(x, t)
XT_sol = np.hstack((X_sol.flatten()[:, None], T_sol.flatten()[:, None]))
u_sol = u.T.flatten()[:, None]

# Dominio espacial y temporal para DeepXDE
geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Definir la PDE para DeepXDE
def pde(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t - 0.0001 * dy_xx + 5 * y ** 3 - 5 * y

# Condición inicial
def initial_condition(x):
    return x[:, 0:1] ** 2 * np.cos(np.pi * x[:, 0:1])

ic = dde.icbc.IC(geomtime, initial_condition, lambda x, on_initial: on_initial)

# Condiciones de contorno periódicas
def boundary_left(x, on_boundary):
    return on_boundary and np.isclose(x[0], -1)

def boundary_right(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)

bc1 = dde.icbc.PeriodicBC(geomtime, 0, boundary_left, component=0)
bc2 = dde.icbc.PeriodicBC(geomtime, 0, boundary_right, component=0)

# Condiciones de contorno periódicas con el subconjunto
observe_u = dde.icbc.PointSetBC(XT_sol, u_sol, component=0)

# Compilar y entrenar el modelo con los datos adicionales
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc1, bc2, ic, observe_u],
    num_domain=10000,
    num_boundary=80,
    num_initial=160,
)

net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
losshistory, train_state = model.train(iterations=50000)

# Predicciones
x = geomtime.random_points(1000)
y = model.predict(x)



