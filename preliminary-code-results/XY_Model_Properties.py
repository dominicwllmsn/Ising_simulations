import numpy as np
from numpy import pi
from numpy.random import rand
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import scipy

# This function will generate the initial random state
# N is the number of sites in the lattice
def InitialState(N):
    # This picks a random number from 0 to 2*pi,
    # which represents a spin with magnitude 1,
    # with an angle, theta, between 0 and 2*pi,
    # such that each spin can be given by the equation e^{i*theta}
    state = np.random.random((1,N))*2*pi
    # This converts the 'state' into an indexable list
    state = state[0][:]
    return state

# Let's do the good old metropolis algorithm
# T is the temperature
def Metropolis(T, matrix):
    # N is the number of sites in the matrix
    # What we're about to do is very subtle,
    # we will collapse the 2D array into a 1D list,
    # such that every site can be described by a single index,
    # where the index ranges from 0 to N-1
    N = int(np.size(matrix))
    # L is the length and width of the matrix,
    # it is a square matrix and hence L is the square root of N
    L = int(np.sqrt(N))
    # For each site in the matrix this will return the index of the nearest neighbour sites,
    # the reader should confirm this by analytical determining them for a simple, say 2x2, matrix
    NeighbourIndex = {i: ((i//L)*L+(i+1)%L,(i+L)%N,(i//L)*L+(i-1)%L,(i-L)%N) for i in list(range(N))}
    # beta is of course just the inverse temperature
    beta = float(1/T)
    # We will now perform one sweep
    # i iterates from 0 to N-1
    for i in range(N):
        # Randomly choose a spin
        spin = np.random.randint(0, N-1)
        # We're going to do this bit 3 times,
        # because it will optimise the code,
        # look back on this once you understand what's happening in the for loop,
        # and it should beceom very clear
        for m in range(3):
            # Calculate the current energy of the random spin chosen,
            # this is dependent on the cosine of the difference in the angle of the spin,
            # and the angle of the nearest neighbour spins
            InitialEnergy = -sum(np.cos(matrix[spin]-matrix[n]) for n in NeighbourIndex[spin]) 
            # Now change the angle of the spin by a random amount dTheta,
            # where dTheta can range from -pi to pi
            dTheta = pi * (np.random.random()*2-1)
            # Now we need a temporary spin value to account for the change in theta we just made
            TempSpin = matrix[spin] + dTheta
            # Calculate the energy accounting for this change
            FinalEnergy = -sum(np.cos(TempSpin-matrix[n]) for n in NeighbourIndex[spin]) 
            # deltaE is the change in energy
            deltaE = FinalEnergy - InitialEnergy
            # If the change in energy is negative,
            # then make the change with probability 1
            if deltaE < 0:
                matrix[spin] += dTheta 
            # Else if the the change in energy is greater than or equal to 0,
            # make the change with a probability dependent on the Boltzmann factor
            elif rand() < np.exp(-beta*deltaE):
                matrix[spin] += dTheta
    return matrix

# This calculates the magnetisation of a given configuration
def MagnetisationCalculation(matrix):
    magCos = np.sum(np.cos(matrix))**2
    magSin = np.sum(np.sin(matrix))**2
    magnetisation = np.sqrt(magCos + magSin)
    return magnetisation

# This function will calculate the energy in the configuration
def EnergyCalculation(matrix):
    # As before N is just the number of sites in the matrix
    N = int(np.size(matrix))
    # And L is the length and width of the matrix
    L = int(np.sqrt(N))
    # Set the energy to 0
    energy = 0
    # We have our nearest neighbour index again
    NeighbourIndex = {i: ((i//L)*L+(i+1)%L,(i+L)%N,(i//L)*L+(i-1)%L,(i-L)%N) for i in list(range(N))}
    # For every spin calculate the energy associated with each nearest neighbour spin,
    # and then do that for every spin summing them all
    for i in range(N):
        energy += -sum(np.cos(matrix[i]-matrix[n]) for n in NeighbourIndex[i])
    # The energy is divided by 4 to avoid quadruple counting
    return energy/4

# Main Code

TempPoints = 2**6        # This is the number of temperature points
N          = 2**4        # This is the number of total sites in the lattice
Sweeps1    = 2**9        # This is the number of sweeps to reach thermal equilibrium
Sweeps2    = 2**9        # This is the number of sweeps to do our calculations using the Monte Carlo method

n1 = 1/(Sweeps2*N)
n2 = 1/(Sweeps2*Sweeps2*N)

# Create the temperatures we want to test over
T = list(np.linspace(2.5, 0.01, TempPoints))

# Create lists of zeros of length TempPoints
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
        Metropolis(T[m], lattice)

    # Do this loop Sweeps2 times
    # Now the lattice is in thermal equilibrium we can calculate properties of the system
    for i in range(Sweeps2):
        # So it's doing the metropolis algorithm another Sweeps2 times on the lattice
        Metropolis(T[m], lattice)         
        # It then calculates the energy and magnetisation after every loop
        Ene = EnergyCalculation(lattice)
        Mag = MagnetisationCalculation(lattice)
        
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
    
    # We'll give ourselves an output everytime it completes the loop,
    # just so we know how quickly we're making progress
    print('Progress: ' + str(m+1) + '/' + str(len(T)))

# This fits the energy data to a polynomial of 11th order with respect to T    
EnergyPolyfit = np.poly1d(np.polyfit(T, Energy, 11))
# This calculates the second derivative of the polynomial
yprimeprimeEnergy = EnergyPolyfit.deriv().deriv()
# This solves the equation, secondDerivative=0, using the Newton-Raphson method,
# where x0 is the starting value for the first iteration
x0 = 1.3
EnergyTc = scipy.optimize.newton(yprimeprimeEnergy, x0)
# This is text to add onto the graph with the result for Tc
EnergyTcText = AnchoredText(r'$T_C = %4.3f$' %EnergyTc, loc=2, frameon=False)

# Do the same for magnetisation
MagnetisationPolyfit = np.poly1d(np.polyfit(T, Magnetisation, 13))
yprimeprimeMagnetisation = MagnetisationPolyfit.deriv().deriv()
MagnetisationTc = scipy.optimize.newton(yprimeprimeMagnetisation, x0)
MagnetisationTcText = AnchoredText(r'$T_C = %4.3f$' %MagnetisationTc, loc=1, frameon=False)

# For specific heat and susceptibility we want to calculate the first derivative instead
SpecificHeatPolyfit = np.poly1d(np.polyfit(T, SpecificHeat, 13))
yprimeSpecificHeat = SpecificHeatPolyfit.deriv()
SpecificHeatTc = scipy.optimize.newton(yprimeSpecificHeat, x0)
#SpecificHeatTc = T[np.argmax(SpecificHeat)]
SpecificHeatTcText = AnchoredText(r'$T_C = %4.3f$' %SpecificHeatTc, loc=2, frameon=False)

SusceptibilityPolyfit = np.poly1d(np.polyfit(T, Susceptibility, 13))
yprimeSusceptibility = SusceptibilityPolyfit.deriv()
SusceptibilityTc = scipy.optimize.newton(yprimeSusceptibility, x0)
#SusceptibilityTc = T[np.argmax(Susceptibility)]
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