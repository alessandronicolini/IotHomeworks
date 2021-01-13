import matplotlib.pyplot as plt
import numpy as np

theta = np.linspace(0,1, 100)

def phi(Theta, Lambda):
    return np.exp((Theta-1)*Lambda)

for Lambda in [-2, -1, -0.5, 0.5, 1, 2]:
    plt.plot(theta, phi(theta, Lambda), label=str(Lambda))
plt.grid()
plt.legend()
plt.ylim(0,1.5)
plt.savefig('ooo.png')