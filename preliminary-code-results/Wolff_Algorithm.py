import numpy as np
import math
from numpy.random import rand
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import scipy

def InitialState(N):
    # This generates a random spin NxN configuration for the initial condition
    state = 2*np.random.randint(2, size=(N,N))-1
    return state

def Wolff(matrix, beta):
    # This is the Wolff algorithm
    # p is the probability that a bond exists between two spins
    p = 1 - np.exp(-2*J*beta)
    # Choose two random numbers between 0 and N-1
    a = np.random.randint(0, N)
    b = np.random.randint(0, N)
    # This labels the index of the spin
    SpinIndex = [(a, b)]
    # Flip the ath by bth spin
    matrix[a, b] *= -1
    while SpinIndex:
        # This removes the last item in the list 'spin'
        x, y = SpinIndex.pop()
        # x and y are the removed items,
        # and start as x=a and y=b in the first iteration
        for i in [((x-1)%N, y), ((x+1)%N, y), (x, (y-1)%N), (x, (y+1)%N)]:
        # Those are the nearest neighbour spins for the (x,y) element
            if matrix[i] == matrix[a, b] * -1 and rand() < p:
                # If the nearest neighbour spin is equal to the original ath by bth spin,
                # and a random number is less than the probability p,
                # then flip that spin
                matrix[i] *= -1
                # Add the nearest neighbour index to the list of spin indexes
                SpinIndex.append(i)
                # So it's searching for the cluster of spins with the same spin as the ath by bth spin
    return matrix

def EnergyCalculation(matrix):
    # This calculates the energy of a given configuration
    energy = 0
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            # For every spin calculate the sum of the nearest neighbours
            spin = matrix[i,j]
            neighbours = matrix[(i+1)%N, j] + matrix[i,(j+1)%N] + matrix[(i-1)%N, j] + matrix[i,(j-1)%N]
            # Then based on the sum calculate the energy for every spin in the matrix
            energy += -J*neighbours*spin
    return energy/4
    # The energy is divided by 4 to avoid quadruple counting

def MagnetisationCalculation(matrix):
    # This calculates the magnetization of a given configuration
    magnetisation = np.sum(matrix)
    # The total magnetisation is the sum of all the individual spins,
    # and hence just the sum of the matrix
    return magnetisation

# Main Code

TempPoints = 2**8         # This is the number of temperature points
N          = 2**5         # This is the size of the lattice
Sweeps1    = 2**9         # This is the number of sweeps to reach thermal equilibrium
Sweeps2    = 2**8         # This is the number of sweeps to do our calculations using the Monte Carlo method

J = 1
# J is the strength of the exchange interaction

n1 = 1/(Sweeps2*N*N)
n2 = 1/(Sweeps2*Sweeps2*N*N)
kbTc = 2*J/math.log(1+math.sqrt(2))
# This is the theoretical value of kbTc for the 2D-Ising model,
# so it seems sensible to centre our distribution around this number
T = np.random.normal(kbTc, 0.6, TempPoints)
# T is a normal (Gaussian) distribution centered on kbTc with a standard deviation of 0.3,
# and TempPoints numbers are produced
# The second argument of T is the standard deviation
T  = T[(T>1.2) & (T<3.8)]
# This gets rid of numbers in the array T thare aren't within 1.2 and 3.8
# We're doing this because we're only interested in the distribution around kbTc
TempPoints = np.size(T)
# This is now the new number of temperature points
T = np.sort(T)[::-1]
# We've sorted T in order of descending numbers

# Creates a list of zeros of length TempPoints
Energy = np.zeros(TempPoints)
Magnetisation = np.zeros(TempPoints)
SpecificHeat = np.zeros(TempPoints)
Susceptibility = np.zeros(TempPoints)

# Create the intial lattice
lattice = InitialState(N)

# This calculates m data points,
# where m is the length of T which is equal to TempPoints
# The start of the next temperature uses the final lattice of the previous...
# temperature as its initial lattice
for m in range(len(T)):
    E1 = M1 = E2 = M2 = 0
    Beta=1/T[m]
    # Beta is just 1 over the mth element of T
    Beta2=Beta*Beta
    # Beta2 is Beta squared
    
    # Do this loop Sweeps1 times
    # The purpose of this loop is to allow the system to attain thermal equilibrium
    for i in range(Sweeps1):
        # Do the metropolis algorithm to the lattice
        Wolff(lattice, Beta)

    # Do this loop Sweeps2 times
    # Now the lattice is in thermal equilibrium we can calculate properties of the system
    for i in range(Sweeps2):
        # So it's doing the metropolis algorithm another Sweeps2 times on the lattice
        Wolff(lattice, Beta)         
        # It then calculates the energy and magnetisation after every loop
        Ene = EnergyCalculation(lattice)
        # Mag must be the absolute value of the magnetisation due to the Wolff algorithm
        Mag = abs(MagnetisationCalculation(lattice))
        # Sum the relevent quantities
        E1 += Ene
        # E1 is the sum of the energies
        M1 += Mag
        # M1 is the sum of the magnetisations
        M2 += Mag*Mag 
        # M2 is the sum of the magnetisations squared
        E2 += Ene*Ene
        # E2 is the sum of the energies squared
        
    # Create lists of various quantities
    Energy[m]         = n1*E1
    Magnetisation[m]  = n1*M1
    SpecificHeat[m]   = (n1*E2 - n2*E1*E1)*Beta2
    Susceptibility[m] = (n1*M2 - n2*M1*M1)*Beta
    
# This fits the energy data to a polynomial of 9th order with respect to T
EnergyPolyfit = np.poly1d(np.polyfit(T, Energy, 9))
# This calculates the second derivative of the polynomial
yprimeprimeEnergy = EnergyPolyfit.deriv().deriv()
# This solves the equation, secondDerivative=0, using the Newton-Raphson method
EnergyTc = scipy.optimize.newton(yprimeprimeEnergy, kbTc)
# This is text to add onto the graph with the result for Tc
EnergyTcText = AnchoredText(r'$T_C = %4.3f$' %EnergyTc, loc=2, frameon=False)

# Do the same for magnetisation
MagnetisationPolyfit = np.poly1d(np.polyfit(T, Magnetisation, 9))
yprimeprimeMagnetisation = MagnetisationPolyfit.deriv().deriv()
MagnetisationTc = scipy.optimize.newton(yprimeprimeMagnetisation, kbTc)
MagnetisationTcText = AnchoredText(r'$T_C = %4.3f$' %MagnetisationTc, loc=1, frameon=False)

# For specific heat and susceptibility we want to calculate the first derivative instead,
# because the quantities diverge at Tc
SpecificHeatPolyfit = np.poly1d(np.polyfit(T, SpecificHeat, 9))
yprimeSpecificHeat = SpecificHeatPolyfit.deriv()
SpecificHeatTc = scipy.optimize.newton(yprimeSpecificHeat, kbTc)
SpecificHeatTcText = AnchoredText(r'$T_C = %4.3f$' %SpecificHeatTc, loc=2, frameon=False)

SusceptibilityPolyfit = np.poly1d(np.polyfit(T, Susceptibility, 11))
yprimeSusceptibility = SusceptibilityPolyfit.deriv()
SusceptibilityTc = scipy.optimize.newton(yprimeSusceptibility, kbTc)
SusceptibilityTcText = AnchoredText(r'$T_C = %4.3f$' %SusceptibilityTc, loc=1, frameon=False)

# Plot everything

f = plt.figure(figsize=(18, 10))    

sp = f.add_subplot(2, 2, 1 )
plt.plot(T, Energy, 'o', color="black");
plt.plot(np.unique(T), EnergyPolyfit(np.unique(T)), color="r");
plt.axvline(EnergyTc, linewidth=1, color='r');
sp.add_artist(EnergyTcText);
plt.xlabel("Temperature", fontsize=20);
plt.ylabel("Energy ", fontsize=20);

sp =  f.add_subplot(2, 2, 2 );
plt.plot(T, Magnetisation, 'o', color="black");
plt.plot(np.unique(T), MagnetisationPolyfit(np.unique(T)), color="r");
plt.axvline(MagnetisationTc, linewidth=1, color='r');
sp.add_artist(MagnetisationTcText);
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