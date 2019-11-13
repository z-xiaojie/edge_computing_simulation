import matplotlib.pyplot as plt
import numpy as np
import math

p = 0.01
b = 500 * 8000
N = math.pow(10, -9)

modevalue = np.sqrt(2 / np.pi) * math.pow(10, -3)

H = np.random.rayleigh(modevalue)
H2 = H/3

W = math.pow(10, 6)

ee= []
ee2 = []
pp= []
tt= []
while p < 1:
    r = W * math.log2(1 + p * H * H / N)
    t = b / r
    e = t * p
    ee.append(e/1)
    pp.append(p)
    tt.append(t/2)
    p += 0.01

plt.plot(pp, ee)
plt.plot(pp, tt)

plt.show()