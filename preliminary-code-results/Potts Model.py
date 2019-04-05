import numpy as np
from numpy import pi
from numpy.random import rand
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import scipy

def InitialState(N):
    state = np.random.choice(Angles, size=(N,N))
    return state

def Metropolis(T, matrix):
    beta = float(1/T)
    for i in range(N):
        for j in range(N):
                a = np.random.randint(0, N)
                b = np.random.randint(0, N)
                spin =  matrix[a, b]
                for m in range(3):
                    InitialEnergy = -(np.cos(spin-matrix[(a+1)%N,b])+np.cos(spin-matrix[a,(b+1)%N])+np.cos(spin-matrix[(a-1)%N,b])+np.cos(spin-matrix[a,(b-1)%N]))
                    dTheta = np.random.choice(Angles)
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
    plt.xticks([])
    plt.yticks([])
    plt.axis('scaled')
    plt.axis([-1, N, -1, N])
    plt.title('T = %.2f' %T + ', Size = ' + str(N) + r'$\times$' + str(N) + ', q = %.0f' %q +', Sweeps = %i' %i)
    plt.savefig('Sweep_'+str(i)+'.png', dpi=200) 
    plt.show()
    return

def MagnetisationCalculation(matrix):
    magCos = np.sum(np.cos(matrix))**2
    magSin = np.sum(np.sin(matrix))**2
    magnetisation = np.sqrt(magCos + magSin)
    return magnetisation

def EnergyCalculation(matrix):
    energy = 0.
    for i in range(N):
        for j in range(N):
            spin = matrix[i,j]
            energy += -(np.cos(spin-matrix[(i+1)%N,j])+np.cos(spin-matrix[i,(j+1)%N])+np.cos(spin-matrix[(i-1)%N,j])+np.cos(spin-matrix[i,(j-1)%N]))
    return energy/4

q          = 6
N          = 2**2
TempPoints = 2**6
Sweeps1    = 2**9
Sweeps2    = 2**9

Angles = []
delta = (2*pi)/q
for i in range(q):
    Angles.append(delta*(i+1))

n1 = 1/(Sweeps2*N)
n2 = 1/(Sweeps2*Sweeps2*N)

T = list(np.linspace(4, 0.01, TempPoints))

Energy = np.zeros(TempPoints)
Magnetisation = np.zeros(TempPoints)
SpecificHeat = np.zeros(TempPoints)
Susceptibility = np.zeros(TempPoints)

lattice = InitialState(N)

for m in range(len(T)):
    E1 = M1 = E2 = M2 = 0
    Beta=1/T[m]
    Beta2=Beta*Beta
    
    for i in range(Sweeps1):
        Metropolis(T[m], lattice)

    for i in range(Sweeps2):
        Metropolis(T[m], lattice)         
        Ene = EnergyCalculation(lattice)
        Mag = MagnetisationCalculation(lattice)
        
        E1 += Ene
        M1 += Mag
        M2 += Mag*Mag 
        E2 += Ene*Ene

    Energy[m]         = n1*E1
    Magnetisation[m]  = n1*M1
    SpecificHeat[m]   = (n1*E2 - n2*E1*E1)*Beta2
    Susceptibility[m] = (n1*M2 - n2*M1*M1)*Beta
    
    print('Progress: ' + str(m+1) + '/' + str(len(T)))
 
EnergyPolyfit = np.poly1d(np.polyfit(T, Energy, 11))
#yprimeprimeEnergy = EnergyPolyfit.deriv().deriv()
#x0 = 1.7
#EnergyTc = scipy.optimize.newton(yprimeprimeEnergy, x0)
#EnergyTcText = AnchoredText(r'$T_C = %4.3f$' %EnergyTc, loc=2, frameon=False)

MagnetisationPolyfit = np.poly1d(np.polyfit(T, Magnetisation, 13))
#yprimeprimeMagnetisation = MagnetisationPolyfit.deriv().deriv()
#MagnetisationTc = scipy.optimize.newton(yprimeprimeMagnetisation, x0)
#MagnetisationTcText = AnchoredText(r'$T_C = %4.3f$' %MagnetisationTc, loc=1, frameon=False)

SpecificHeatPolyfit = np.poly1d(np.polyfit(T, SpecificHeat, 13))
#yprimeSpecificHeat = SpecificHeatPolyfit.deriv()
#SpecificHeatTc = scipy.optimize.newton(yprimeSpecificHeat, x0)
SpecificHeatTc = T[np.argmax(SpecificHeat)]
SpecificHeatTcText = AnchoredText(r'$T_C = %4.3f$' %SpecificHeatTc, loc=2, frameon=False)

SusceptibilityPolyfit = np.poly1d(np.polyfit(T, Susceptibility, 13))
#yprimeSusceptibility = SusceptibilityPolyfit.deriv()
#SusceptibilityTc = scipy.optimize.newton(yprimeSusceptibility, x0)
SusceptibilityTc = T[np.argmax(Susceptibility)]
SusceptibilityTcText = AnchoredText(r'$T_C = %4.3f$' %SusceptibilityTc, loc=1, frameon=False)

f = plt.figure(figsize=(18, 10))    

sp = f.add_subplot(2, 2, 1 )
plt.plot(T, Energy, 'o', color="black");
plt.plot(np.unique(T), EnergyPolyfit(np.unique(T)), color="r");
#plt.axvline(EnergyTc, linewidth=1, color='r');
#sp.add_artist(EnergyTcText);
plt.xlabel("Temperature", fontsize=20);
plt.ylabel("Energy ", fontsize=20);

sp =  f.add_subplot(2, 2, 2 );
plt.plot(T, Magnetisation, 'o', color="black");
plt.plot(np.unique(T), MagnetisationPolyfit(np.unique(T)), color="r");
#plt.axvline(MagnetisationTc, linewidth=1, color='r');
#sp.add_artist(MagnetisationTcText);
plt.xlabel("Temperature", fontsize=20);
plt.ylabel("Magnetisation ", fontsize=20);

sp =  f.add_subplot(2, 2, 3 );
plt.plot(T, SpecificHeat, 'o', color="black");
plt.plot(np.unique(T), SpecificHeatPolyfit(np.unique(T)), color="r");
plt.axvline(SpecificHeatTc, linewidth=1, color='r');
sp.add_artist(SpecificHeatTcText);
plt.xlabel("Temperature", fontsize=20);
plt.ylabel("Specific Heat ", fontsize=20);

sp =  f.add_subplot(2, 2, 4 );
plt.plot(T, Susceptibility, 'o', color="black");
plt.plot(np.unique(T), SusceptibilityPolyfit(np.unique(T)), color="r");
plt.axvline(SusceptibilityTc, linewidth=1, color='r');
sp.add_artist(SusceptibilityTcText);
plt.xlabel("Temperature", fontsize=20);
plt.ylabel("Susceptibility", fontsize=20);

plt.savefig('Test.png')
    
    
    
    
    
    
    
    
    
    
    
    
    