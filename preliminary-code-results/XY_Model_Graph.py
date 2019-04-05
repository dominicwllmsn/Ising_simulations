import numpy as np
from numpy import pi
from numpy.random import rand
import matplotlib.pyplot as plt

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

# This function converts a list into a matrix to help us plot the spins
def ListToMatrix(matrix, L):
    # Reshape the list into a matrix of dimensions L by L
    matrix = np.reshape(matrix,(L,L))
    return matrix

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
    # Our output is a new matrix that has completed one sweep of the Metropolis algorithm
    return matrix

# This is going to be our function for plotting and visualising the function
def Plot(matrix, T, i = None):
    # First convert the list to a matrix
    matrix = ListToMatrix(matrix, L)
    # X is the horizontal position of each spin,
    # and Y is the vertical position of each spin
    X, Y = np.meshgrid(np.arange(0, L), np.arange(0, L))
    # U is the horizontal component of the vector
    U = np.cos(matrix)
    # V is the vertical component of the vector
    V = np.sin(matrix)
    # Now plot the quiver graph
    plt.quiver(X, Y, U, V)
    # This removes the axis numbers
    plt.xticks([])
    plt.yticks([])
    # Give each graph a relevant name
    plt.title('T = %.2f' %T + ', Size = ' + str(L) + 'x' + str(L) + ', Sweeps = %i' %i)
    # Save the graph
    plt.savefig('Sweep_'+str(i)+'.png', dpi=200) 
    # Show me the graph,
    # although this can be commented out to reduce computation time if you use a lot of sweeps
    plt.show()
    return

# These are our parameters
# L is the width and length of the lattice
L = 10
# N is the number of sites in the lattice
N = L*L
# T is the temperature
T = 0.01
# Sweeps is the number of sweeps we are going to perform
Sweeps = 11

# Create the intial lattice
lattice = InitialState(N)    

# We'll implement a 'for' loop to plot the graph for each sweep
for i in range(Sweeps):
    Plot(lattice, T, i)
    # After each plot it does one iteration of the Metropolis algorithm
    lattice = Metropolis(T, lattice) 




