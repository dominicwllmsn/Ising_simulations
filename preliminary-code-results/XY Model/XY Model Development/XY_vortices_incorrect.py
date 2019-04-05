import numpy as np
from numpy import pi
from numpy.random import rand
import matplotlib.pyplot as plt

def InitialState(N):
    state = np.random.random((N,N))*2*pi
    return state

def Delta(theta1, theta0):
    best = None
    for i in (-2*pi, 0, 2*pi):
      for j in (-2*pi, 0, 2*pi):
        r = (theta1+i) - (theta0+j)
        if best is None or abs(r) < abs(best):
            best = r
    return best

def Vortex(matrix, i, j):
    windingnumber = 0.
    theta = matrix[i, j]
    q = 1
    
    for (di, dj) in [(0,-1),(+1,-1),(+1,0),(0,0)]:    
        theta1 = matrix[(i+di)%N, (j+dj)%N]
        dtheta = Delta(theta1, theta)
        windingnumber += dtheta
        if q == 1:
            if dtheta > 0: bigger = True
            else: bigger = False
            q += 1
            
        if bigger is True and dtheta < 0: return 0
        elif bigger is False and dtheta > 0: return 0

        theta = theta1
    return windingnumber

def Metropolis(T, matrix):
    beta = float(1/T)
    for i in range(N):
        for j in range(N):
                a = np.random.randint(0, N)
                b = np.random.randint(0, N)
                spin =  matrix[a, b]
                for m in range(3):
                    InitialEnergy = -(np.cos(spin-matrix[(a+1)%N,b])+np.cos(spin-matrix[a,(b+1)%N])+np.cos(spin-matrix[(a-1)%N,b])+np.cos(spin-matrix[a,(b-1)%N]))
                    dTheta = pi * (np.random.random()*2-1)
                    TempSpin = spin + dTheta
                    FinalEnergy = -(np.cos(TempSpin-matrix[(a+1)%N,b])+np.cos(TempSpin-matrix[a,(b+1)%N])+np.cos(TempSpin-matrix[(a-1)%N,b])+np.cos(TempSpin-matrix[a,(b-1)%N]))
                    deltaE = FinalEnergy - InitialEnergy
                    if deltaE < 0:
                        spin += dTheta 
                    elif rand() < np.exp(-beta*deltaE):
                        spin += dTheta
                    matrix[a, b] = spin
    return matrix

def Plot(matrix, T, i = None):
    
    X, Y = np.meshgrid(np.arange(0, N), np.arange(0, N))
    U = np.cos(matrix)
    V = np.sin(matrix)
    plt.quiver(X, Y, U, V)
    
    blacklist = []
    for a in range(N):
          for b in range(N):
            if (a,b) in blacklist:
                continue
            WindingNumber = Vortex(matrix, a, b)
#            print(WindingNumber%pi)
            if abs(WindingNumber) < 0.01:
                continue
            elif WindingNumber > 0 and WindingNumber%pi < 0.01:
                print(WindingNumber%pi)
                for (di, dj) in [(0,0),(0,-1),(+1,-1),(+1,0)]:    
                    circle = plt.Circle(((a+di)%N,(b+dj)%N), 0.6, color="red", alpha=0.4)
                    ax=plt.gca()
                    ax.add_patch(circle)
                    blacklist.append(((a+di)%N,(b+dj)%N))
            elif WindingNumber < 0 and WindingNumber%pi < 0.01:
                print(WindingNumber%pi)
                for (di, dj) in [(0,0),(0,-1),(+1,-1),(+1,0)]:    
                    circle = plt.Circle(((a+di)%N,(b+dj)%N), 0.6, color="blue", alpha=0.4)
                    ax=plt.gca()
                    ax.add_patch(circle)
                    blacklist.append(((a+di)%N,(b+dj)%N))

    
    plt.xticks([])
    plt.yticks([])
    plt.axis('scaled')
    plt.axis([-1, N, -1, N])
    plt.title('T = %.2f' %T + ', Size = ' + str(N) + r'$\times$' + str(N) + ', Sweeps = %i' %i)
    plt.savefig('Sweep_'+str(i)+'.png', dpi=200) 
    plt.show()
    return

N = 20
T = 0.01
Sweeps = 20

lattice = InitialState(N)

for i in range(Sweeps+1):
    Plot(lattice, T, i)
    lattice = Metropolis(T, lattice)