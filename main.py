import model
import numpy as np
import os


def explore():
    '''
    Function will explore and learn but also utilize what's learned
    '''
    gamma = 0.95
    epsilon = 0.9
    alpha = 0.5  # learning rate
    world = int(input("Enter the world to be trained: "))
    traverses = int(input("How many traverses? "))

    # verbosity
    verbose = str(input("Verbose? 'Y' or 'N' "))
    v = True if verbose.capitalize() == "Y" else False

    # Checking and creating files
    # filepath will contain only .png files of previous runs
    # .npy file will be saved directly to runs
    filepath = f"./runs/world{world}/"
    filename = f"./runs/qTable{world}.npy"

    if not (os.path.exists(filepath)):
        os.makedirs(filepath)

    if not (os.path.isfile(filename)):
        np.save(filename, np.zeros((40, 40, 4)))
    qTable = np.load(filename)

    # Loading obstacles, goodStates, and badStates
    if not os.path.isfile(f"./runs/obstaclesWorld{world}.npy"):
        obstacles = []
    else:
        obstacles = np.load(f"./runs/obstaclesWorld{world}.npy")

    if not os.path.isfile(f"./runs/goodStatesWorld{world}.npy"):
        goodStates = []
    else:
        goodStates = np.load(f"./runs/goodStatesWorld{world}.npy")

    if not os.path.isfile(f"./runs/badStatesWorld{world}.npy"):
        badStates = []
    else:
        badStates = np.load(f"./runs/badStatesWorld{world}.npy")

    runNum = len([i for i in os.listdir(f"runs/world{world}")])

    for traverse in range(traverses):
        qTable, goodStates, badStates, obstacles = model.learn(
            qTable, worldId=world, mode='explore', alpha=alpha,
            gamma=gamma, epsilon=epsilon, goodStates=goodStates,
            badStates=badStates, traverse=traverse, obstacles=obstacles,
            runNum=runNum, verbose=v)
        epsilon = model.epsilonDecay(epsilon, traverse, traverses)

    # Save once at the end
    np.save(filename, qTable)
    np.save(f"./runs/obstaclesWorld{world}", obstacles)
    np.save(f"./runs/goodStatesWorld{world}", goodStates)
    np.save(f"./runs/badStatesWorld{world}", badStates)


'''
NOTE: NOT FIXED YET, DON'T USE EXPLOIT()
'''


def exploit():
    '''
    Function will only exploit the current policy for maximizing score
    '''
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


def main():
    inp = input("Exploring(1) or exploit(2)? ")

    if inp == "1":
        explore()
    else:
        exploit()


if __name__ == "__main__":
    main()
