from re import S
import numpy as np
import matplotlib.pyplot as plt
import colorama
from colorama import Fore
EPS = 1e-6
def f(x, y):
    return np.cos(x) + y**2
def grad(x, y):
    return [-np.sin(x), 2*y]

def distance(point1, point2):
    return np.linalg.norm(point1 - point2)
def gradient_descent(x, lr, expected_ans, points, is_print):
    colorama.init()
    global epoch
    global best_lr
    global best_epoch
    epoch = 0
    if is_print:
        print(Fore.RED + '--------------------------------')
        print("function is: cos(x) + y**2")
        print("start point is", x[0], ";", x[1])
    while(distance(x, expected_ans) >= EPS):
        x = x - lr * np.array(grad(x[0], x[1]))
        points = np.append(points, x, axis = 0)
        epoch += 1
    if is_print:    
        print("learning rate is", lr)
        print("found minimum: ", x[0], "; ", x[1],  "\nexpected minimum: ", \
            expected_ans[0], "; ", expected_ans[1], \
            "\nnumber of iterations for accuracy", EPS, ": ", epoch)
        print(Fore.RED + '--------------------------------\n')  
    if (best_epoch > epoch):
        best_epoch = epoch
        best_lr = lr       
    return points  

def gradient_descent_with_graph(x, lr, expected_ans, is_print):
    plt.rcParams["figure.figsize"] = (20, 10)
    t = np.linspace(-30, 30, 1000)
    X, Y = np.meshgrid(t, t)
    figure = plt.figure()
    ax = figure.add_subplot(projection='3d')
    ax.plot_surface(X, Y, f(X, Y))
    points = np.array(x)  
    points = gradient_descent(x, lr, expected_ans, points, is_print).reshape(epoch+1, 2)
    plt.plot(points[:, 0], points[:, 1], 'o-')
    plt.contour(X, Y, f(X, Y), levels=sorted([f(p[0], p[1]) for p in points]))
    plt.grid(linestyle = '--')
    plt.show()
    return epoch

if __name__ == '__main__':
    best_lr = 0.5
    best_epoch = 1e9
    gradient_descent([-20, -20], 0.5, \
    np.array([-7 * np.pi, 0]), np.array([-20, -20]), True)

    gradient_descent([-20, -20], 0.1, \
    np.array([-7 * np.pi, 0]), np.array([-20, -20]), True)

    gradient_descent([-20, -20], 0.95, \
    np.array([-7 * np.pi, 0]), np.array([-20, -20]), True)

    gradient_descent([-20, -20], 0.3, \
    np.array([-7 * np.pi, 0]), np.array([-20, -20]), True)

    gradient_descent([-20, -20], 0.01, \
    np.array([-7 * np.pi, 0]), np.array([-20, -20]), True)

    gradient_descent([-20, -20], 0.99, \
    np.array([-7 * np.pi, 0]), np.array([-20, -20]), True)
    gradient_descent([-20, -20], 0.6, \
    np.array([-7 * np.pi, 0]), np.array([-20, -20]), True)
    gradient_descent([-20, -20], 0.7, \
    np.array([-7 * np.pi, 0]), np.array([-20, -20]), True)

    gradient_descent([-20, -20], 0.8, \
    np.array([-7 * np.pi, 0]), np.array([-20, -20]), True)

    print("Best lr is:", best_lr, "| Best epoch is:", best_epoch)

    gradient_descent_with_graph([-20, -20], 0.6, \
    np.array([-7 * np.pi, 0]), False)
    
