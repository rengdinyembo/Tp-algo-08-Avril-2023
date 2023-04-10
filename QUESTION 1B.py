import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# Durée d'intégration
t=np.linspace(0,12,5000)


# Fonctions utiles à la résolution
def f(X,t):
    z,zpoint = X
    return zpoint, - 9*z - 10*zpoint + 100*np.cos(10*t)

def sol(z0,zpoint0):
    return odeint(f,[z0,zpoint0],t)[:,0]

    
# Tracé
z0=0.01
zpoint0=0

plt.figure()
plt.plot(t,sol(z0,zpoint0))
plt.xlabel('temps en s')
plt.ylabel('z')
plt.grid()         
plt.show()
