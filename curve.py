import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import random

horizon = 100


def createCurve(h):
    x, y = [0], [0]
    for i in range(h):
        x.append(i + 1)
        y.append(y[-1] + random.random() * 0.5 - 0.125)
    tck = interpolate.splrep(x, y, s=0)
    xnew = np.arange(0, h, 0.1)
    ynew = interpolate.splev(xnew, tck, der=0)
    return (xnew, ynew)


def createCurveRec(h):
    x = range(h+1)
    y = [0]
    for i in range(h):
        if i < 10 :
            y.append(random.randint(1,3) + y[-1]/2)
        else :
            y.append(random.randint(1,4) + y[-1]/2)
    smooth = h - np.sqrt(2*h)
    tck = interpolate.splrep(x, y, s=smooth)
    xnew = np.arange(0, h, 0.1)
    ynew = interpolate.splev(xnew, tck, der=0)
    return (xnew, ynew)


if __name__ == '__main__':
    (xnew, ynew) = createCurveRec(horizon)

    plt.plot(xnew, ynew - 1, xnew, ynew + 1)
    plt.axis([-0.05, horizon, -2, 8])
    plt.show()