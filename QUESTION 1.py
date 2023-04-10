import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def systeme_masse_ressort_amortisseur(x,t,m,a,k,F):
    """
    Fonction décrivant le système masse-ressort-amortisseur.
    Args:
        x (array): Vecteur d'état [position, vitesse].
        t (float): Temps.
        m (float): Masse.
        a (float): Coefficient de frottement de l'amortisseur.
        k (float): Constante de raideur du ressort.
        F (float): Force extérieure.
    Returns:
        dx/dt (array): Dérivées du vecteur d'état.
    """
    dxdt = np.zeros_like(x)
    dxdt[0] = x[1]
    dxdt[1] = (F(t) - a * x[1] - k * x[0]) / m
    return dxdt

# Paramètres du système
m = 10.0 # Masse en kg
a = 20.0 # Coefficient de frottement de l'amortisseur en Ns/m
k = 4000.0 # Constante de raideur du ressort en N/m
x0 = 0.01 # Position initiale en m
v0 = 0.0 # Vitesse initiale en m/s

# Fonction pour la force extérieure F(t)
F0 = 100.0 # Amplitude de la force en N
w = 10.0 # Fréquence de la force en rad/s
def force_exterieure(t):
    return F0 * np.cos(w * t) # Cas b) Force extérieure F(t) = F0*cos(wt)

# Intervalle de temps

t_start = 0.0
t_end = 10.0 # Temps final
dt = 0.01 # Pas de temps

# Vecteur d'état initial

x_init = np.array([x0, v0])

# Résolution de l'équation différentielle pour le cas a) Les oscillations sont libres
t_a = np.arange(t_start, t_end, dt)
x_a = odeint(systeme_masse_ressort_amortisseur, x_init, t_a, args=(m, a, k, lambda t: 0.0))

# Résolution de l'équation différentielle pour le cas b) Une force extérieure F(t) = F0*cos(wt)
t_b = np.arange(t_start, t_end, dt)
x_b = odeint(systeme_masse_ressort_amortisseur, x_init, t_b, args=(m, a, k, force_exterieure))

# Plot de la réponse du système pour les deux cas
plt.figure()
plt.subplot(2, 1, 1) 
plt.plot(t_a, x_a[:, 0], label ='Position (m)')
plt.plot(t_a, x_a[:, 1], label ='Vitesse (m/s)')
plt.xlabel('Temps (s)')
plt.ylabel('Position, Vitese')
plt.legend()
plt.title('Réponse du système masse-ressort-amortisseur (Cas a)')
plt.subplot(2, 1, 2)
plt.plot(t_b, x_b[:, 0], label ='Position (m)')
plt.plot(t_b, x_b[:, 1], label ='Vitesse (m/s)')
plt.xlabel('Temps (s)')
plt.ylabel('Position, Vitese')
plt.legend()
plt.title('Réponse du système masse-ressort-amortisseur (Cas b)')
plt.show()

# Calcul des énergies cinétique, potentielle et mécanique pour le cas a)

E_cinetique_a = 0.5 * m * x_a[:, 1]**2
E_potentielle_a = 0.5 * k * x_a[:, 0]**2
E_mecanique_a = E_cinetique_a + E_potentielle_a

# Plot de l'Ep, l'Ec et l'Em

plt.figure()
plt.plot(t_a, E_cinetique_a, label='Energie cinétique')
plt.plot(t_a, E_potentielle_a, label='Energie potentielle')
plt.plot(t_a, E_mecanique_a, label='Energie mécanique')
plt.xlabel('Temps (s)')
plt.ylabel('Ec, Ep et Em (J)')
plt.legend()
plt.title('Ec, Ep et Em à tout instant')
plt.show()
