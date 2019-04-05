import numpy as np
import math
from numpy.random import rand
import matplotlib.pyplot as plt

def InitialState(N):   
    # This generates a random spin NxN configuration for the initial condition
    state = 2*np.random.randint(2, size=(N,N))-1
    return state

def Metropolis(matrix, beta):
    # This is the Metropolis algorithm
    # Do this NxN times
    for i in range(N):
        for j in range(N):
                a = np.random.randint(0, N)
                b = np.random.randint(0, N)
                # Choose two random numbers between 0 and N-1
                spin =  matrix[a, b]
                # This is the spin of the ath by bth element
                # It's either going to be 1 (up) or -1 (down)
                neighbours = matrix[(a+1)%N,b] + matrix[a,(b+1)%N] + matrix[(a-1)%N,b] + matrix[a,(b-1)%N]
                # From the sum of the spins of the nearest neighbours calculate the cost,
                # where periodic boundary conditions have been applied
                dE = 2*J*spin*neighbours
                # It's multiplied by 2 because it's the change in energy,
                # so if the spin were to flip then dE = J*spin*n - J*-spin*n = 2*J*spin*n
                if dE < 0:
                    spin *= -1
                # If the change in energy is less than zero then flip the spin
                # i.e. up goes to down or down goes to up
                elif rand() < np.exp(-dE*beta):
                    spin *= -1
                # If not then only flip the spin if exp(-dE*beta) is greater than a random number between 0 and 1,
                # this is more likely the higher the temperature
                matrix[a, b] = spin
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
N          = 2**4         # This is the size of the lattice
Sweeps1    = 2**8         # This is the number of sweeps to reach thermal equilibrium
Sweeps2    = 2**8         # This is the number of sweeps to do our calculations using the Monte Carlo method

J = 1
# J is the strength of the exchange interaction

n1 = 1/(Sweeps2*N*N)
n2 = 1/(Sweeps2*Sweeps2*N*N)
kbTc = 2*J/math.log(1+math.sqrt(2))
# This is the theoretical value of kbTc for the 2D-Ising model,
# so it seems sensible to centre our distribution around this number
T = np.random.normal(kbTc, 0.5, TempPoints)
# T is a normal (Gaussian) distribution centered on kbTc with a standard deviation of 0.64,
# and TempPoints numbers are produced
# The second argument of T is the standard deviation
T  = T[(T>1.2) & (T<3.8)]
# This gets rid of numbers in the array T thare aren't within 1.2 and 3.8
# We're doing this because we're only interested in the distribution around kbTc
TempPoints = np.size(T)
# This is now the new number of temperature points
T = np.sort(T)[::-1]
# We've sorted T in order of descending numbers

Energy = np.zeros(TempPoints)
Magnetization = np.zeros(TempPoints)
SpecificHeat = np.zeros(TempPoints)
Susceptibility = np.zeros(TempPoints)
# Creates a list of zeros of length TempPoints

lattice = InitialState(N)
# Create the intial lattice

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
        Metropolis(lattice, Beta)

    # Do this loop Sweeps2 times
    # Now the lattice is in thermal equilibrium we can calculate properties of the system
    for i in range(Sweeps2):
        # So it's doing the metropolis algorithm another Sweeps2 times on the lattice
        Metropolis(lattice, Beta)         
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
    Magnetization[m]  = n1*M1
    SpecificHeat[m]   = (n1*E2 - n2*E1*E1)*Beta2
    Susceptibility[m] = (n1*M2 - n2*M1*M1)*Beta

# Plot everything

f = plt.figure(figsize=(18, 10))    

sp = f.add_subplot(2, 2, 1 )
plt.plot(T/kbTc, Energy, 'o', color="black");
plt.xlabel(r"$T/T_C$", fontsize=20);
plt.ylabel("Energy ", fontsize=20);

sp =  f.add_subplot(2, 2, 2 );
plt.plot(T/kbTc, abs(Magnetization), 'o', color="black");
plt.xlabel(r"$T/T_C$", fontsize=20);
plt.ylabel("Magnetisation ", fontsize=20);

sp =  f.add_subplot(2, 2, 3 );
plt.plot(T/kbTc, SpecificHeat, 'o', color="black");
plt.xlabel(r"$T/T_C$", fontsize=20);
plt.ylabel("Specific Heat ", fontsize=20);

sp =  f.add_subplot(2, 2, 4 );
plt.plot(T/kbTc, Susceptibility, 'o', color="black");
plt.xlabel(r"$T/T_C$", fontsize=20);
plt.ylabel("Susceptibility", fontsize=20);

plt.savefig('Test.png')