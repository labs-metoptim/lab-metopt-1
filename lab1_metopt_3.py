import numpy as np
from sympy.abc import a, b
from sympy import cos
from sympy.utilities.lambdify import lambdify
EPS = 1e-6

def f(x, y):
    return np.cos(x) + y**2

def grad(x, y):
    return [-np.sin(x), 2*y]

def distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def dichotomy(f):
    delta = 1e-4
    a = -20
    b = 20
    while(b - a >= EPS):
        x = (a + b)/2
        if f(x-delta) <= f(x+delta):
            b = x
        else:
            a = x   
    return x, f(x)
    
def gradient_descent(x, expected_ans):
     #x_(k+1) = x_k - a*grad(x_k)
     # where a is minimum of phi(a) = f(x_k - a*grad(x_k))
    epoch = 0
    print("--------------------------")    
    print("Function is: cos(x) + y**2")
    print("start point:", x[0], ";", x[1]) 
    while(distance(x, expected_ans) >= EPS):
        expr = cos(a) + b**2
        grad_ = grad(x[0], x[1])
        expr2 = expr.subs(a, x[0] - a * grad_[0]) \
        .subs(b, x[1] - a * grad_[1]) 
        f = lambdify(a, expr2)
        x = x - dichotomy(f)[0] * np.array(grad_) 
        epoch += 1   
    print("found minimum: ", x[0], "; ", x[1],  "\nexpected minimum: ", \
            expected_ans[0], "; ", expected_ans[1], \
            "\nnumber of iterations for accuracy", EPS, ": ", epoch)
    print("--------------------------\n")     

if __name__ == '__main__':    
    print("Dichotomy method")
    print("--------------------\n")
    gradient_descent(np.array([-20, -20]), np.array([-9*np.pi, 0.0])) 
    gradient_descent(np.array([1, 1]), np.array([np.pi, 0.0])) 