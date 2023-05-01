import model
import numpy as np
import os
import time


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
            qTable, world=world, mode='explore', alpha=alpha,
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


'''
NOTE: NOT FIXED YET, DON'T USE EXPLOIT()

def exploit():
    #Function will only exploit the current policy for maximizing score

    epsilon = 0.9
    world = int(input("Enter the world"))
    traverses = int(input("How many traverses?"))
    verbose = str(input("Verbose? 'Y' or 'N'"))
    filePath = f"./runs/qTableWorld{world}"

    qTable = np.load(filePath + ".npy")

    if verbose == "Y":
        v = True
    else:
        v = False

    obstacles = np.load(f"./runs/obstaclesWorld{world}" + ".npy")
    goodStates = np.load(f"./runs/goodStatesWorld{world}" + ".npy")
    badStates = np.load(f"./runs/badStatesWorld{world}" + ".npy")

    obstacles = obstacles.tolist()
    goodStates = goodStates.tolist()
    badStates = badStates.tolist()

    runNum = len([i for i in os.listdir(f"runs/world{world}")])

    for traverse in range(traverses):
        qTable, goodStates, badStates, obstacles = model.learn(
            qTable, worldId=world, mode="Ex", learningRate=0.0001, gamma=0.9,
            epsilon=epsilon, goodStates=goodStates, badStates=badStates,
            traverse=traverse, obstacles=obstacles, runNum=runNum, verbose=v)
            
'''


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
        # epsilon = 0.6 starting out - then decaying
        # alpha = 0.5 starting out - then decaying
        # epochs = the number of runs so far for that world
        parameters = np.array([0.95, 0.5, 0.1, 0])
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
    inp = input("Exploring(1) or exploit(2)? ")

    if inp == "1":
        explore()
    else:
        exploit()

    while True:
        stop = input("Stop?")
        if stop:
            break


if __name__ == "__main__":
    main()
