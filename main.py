import model
import numpy as np
import os
import time
from API import *


def explore():
    '''
    Function will explore and learn but also utilize what's learned
    '''
    world = int(input("Enter the world to be trained: "))
    traverses = int(input("How many traverses? "))

    # verbosity
    verbose = str(input("Verbose? 'Y' or 'N' "))
    v = True if verbose.capitalize() == "Y" else False

    # Get saved data
    qTable, goodStates, badStates, parameters = getData(world)
    gamma, epsilon, alpha, epoch = parameters

    # Traversal
    for traverse in range(1, traverses + 1):
        start = time.time()

        qTable, goodStates, badStates = model.learn(
            qTable, world=world, alpha=alpha,
            gamma=gamma, epsilon=epsilon, goodStates=goodStates,
            badStates=badStates, traverse=epoch,
            verbose=v)

        end = time.time()

        epoch += 1
        epsilon = model.epsilonDecay(epsilon, epoch)
        alpha = model.alphaDecay(alpha, epoch)

        print(f'Time taken for world{world}, epoch{epoch}: {end-start}s')

        # Save every run in case we have to end the runs midway through
        np.save(f"./runs/world{world}/qTable{world}.npy", qTable)

        np.save(
            f"./runs/world{world}/goodStatesWorld{world}.npy", goodStates)
        np.save(f"./runs/world{world}/badStatesWorld{world}.npy", badStates)

        np.save(f"./runs/world{world}/parameters{world}.npy",
                np.array([gamma, epsilon, alpha, epoch]))


def getData(world):
    filepath = f"./runs/world{world}/"
    filename = f"{filepath}qTable{world}.npy"

    if not (os.path.exists(filepath)):
        os.makedirs(filepath)

    if not (os.path.isfile(filename)):
        np.save(filename, np.zeros((40, 40, 4)))
    qTable = np.load(filename)

    if not os.path.isfile(f"{filepath}goodStatesWorld{world}.npy"):
        goodStates = []
    else:
        good = np.load(f"{filepath}goodStatesWorld{world}.npy")
        goodStates = good.tolist()

    if not os.path.isfile(f"{filepath}badStatesWorld{world}.npy"):
        badStates = []
    else:
        bad = np.load(f"{filepath}badStatesWorld{world}.npy")
        badStates = bad.tolist()

    # Loading parameters
    if not os.path.isfile(f'{filepath}parameters{world}.npy'):

        # Beginnning parameters for gamma, epsilon, alpha, and epochs
        # gamma = 0.95 - no changes
        # epsilon = 0.9 starting out - then decaying
        # alpha = 0.5 starting out - then decaying
        # epochs = the number of runs so far for that world
        parameters = np.array([0.95, 0.9, 0.4, 0])
    else:
        parameters = np.load(f'{filepath}parameters{world}.npy')

    return qTable, goodStates, badStates, parameters


def printQTable(world=0, param=True):
    filename = f"./runs/world{world}/qTable{world}.npy"
    qTable = np.load(filename)

    for x in range(40):
        for y in range(40):
            #print(f'x={x}, y={y}')
            print(qTable[x][y], end="  ")

        print()

    if param:
        print(np.load(f'./runs/world{world}/parameters{world}.npy'))


def main():
    # resetTeam()
    explore()


if __name__ == "__main__":
    main()
