import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


# function
def integrand(z, x):
    return 1.0 / (np.exp(1.9 * z**2) + x)

# f(x) integral
def f(x):
    val, err = quad(integrand, 0, 1, args=(x,))
    return val, err

# interpolation nodes
x_nodes = np.arange(0, 3.1, 0.5)
y_nodes = []
err_nodes = []
for x in x_nodes:
    val, err = f(x)
    y_nodes.append(val)
    err_nodes.append(err)

cs = CubicSpline(x_nodes, y_nodes)

# polinom lagrange
def lagrange(x, xp, yp):
    n = len(xp)
    result = 0.0
    for i in range(n):
        L_i = 1.0
        for j in range(n):
            if i != j:
                L_i *= (x - xp[j]) / (xp[i] - xp[j])
        result += yp[i] * L_i
    return result

# comp points
xk = 0.25 + 0.5 * np.arange(6)

# exact values in comp points
f_xk = [f(x) for x in xk]
f_xk_vals = np.array([v for v, _ in f_xk])
f_xk_errs = [e for _, e in f_xk]

# spline and polinom in values
spline_vals = cs(xk)
lagrange_vals = [lagrange(x, x_nodes, y_nodes) for x in xk]

print("="*70)
print("calc integrals")
print("="*70)
print(f"func: f(x) = ∫₀¹ dz / (exp(1.9·z²) + x)")
print("-"*70)
print(f"{'x':>6} | {'f(x)':>18} | {'error':>15}")
print("-"*70)
for xi, yi, erri in zip(x_nodes, y_nodes, err_nodes):
    print(f"{xi:6.2f} | {yi:18.10f} | {erri:15.2e}")
print("="*70)

print("\n" + "="*90)
print("comp aproximations in points xk = 0.25 + 0.5·k (k = 0..5)")
print("="*90)
print(f"{'k':>3} | {'xk':>6} | {'f(xk)':>12} | {'L(xk)':>12} | {'S(xk)':>12} | {'L - f':>12} | {'S - f':>12}")
print("-"*90)
for k, x, fv, lv, sv in zip(range(6), xk, f_xk_vals, lagrange_vals, spline_vals):
    print(f"{k:3d} | {x:6.3f} | {fv:12.6f} | {lv:12.6f} | {sv:12.6f} | {lv-fv:12.3e} | {sv-fv:12.3e}")
print("="*90)

print("- max bias spline: {:.2e}".format(np.max(np.abs(spline_vals - f_xk_vals))))
print("- max bias polinom: {:.2e}".format(np.max(np.abs(lagrange_vals - f_xk_vals))))


x_plot = np.linspace(0, 3, 200)

y_lagrange_plot = np.array([lagrange(x, x_nodes, y_nodes) for x in x_plot])
y_spline_plot = cs(x_plot)

print("calc f(x) in points")
y_func_plot = np.array([f(x)[0] for x in x_plot])

plt.style.use('seaborn-v0_8')

# graph 1: f(x) and polynom
plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_func_plot, 'b-', linewidth=2, label='f(x) (exact)')
plt.plot(x_plot, y_lagrange_plot, 'r--', linewidth=2, label='polynom')
plt.scatter(x_nodes, y_nodes, color='black', zorder=5, label='nodes interp')
plt.scatter(xk, f_xk_vals, color='green', marker='s', zorder=5, label='pints comp xk')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('comp')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# graph 2: polynom and spline
plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_lagrange_plot, 'r--', linewidth=2, label='polynom')
plt.plot(x_plot, y_spline_plot, 'g-', linewidth=2, label='cubic spline')
plt.scatter(x_nodes, y_nodes, color='black', zorder=5, label='nodes interp')
plt.scatter(xk, f_xk_vals, color='green', marker='s', zorder=5, label='cpoints comp xk')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('comp second')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
