import model
import numpy as np
import os


def train():
    goodStates = []
    badStates = []
    obstacles = []
    epsilon = 0.9
    world = int(input("Enter the world to be trained"))
    traverses = int(input("How many traverses?"))
    verbose = str(input("Verbose? 'Y' or 'N'"))
    qTable = model.tableInitiate()

    if verbose == "Y":
        v = True
    else:
        v = False

    if not (os.path.exists(f"./runs/world{world}/")):
        os.makedirs(f"./runs/world{world}/")

    runNum = len([i for i in os.listdir(f"runs/world{world}")])
    filePath = f"./runs/qTableWorld{world}"

    for traverse in range(traverses):
        qTable, goodStates, badStates, obstacles = model.learn(
            qTable, worldId=world, mode='train', learningRate=0.0001,
            gamma=0.9, epsilon=epsilon, goodTermStates=goodStates,
            badTermStates=badStates, traverse=traverse, obstacles=obstacles,
            runNum=runNum, verbose=v)
        epsilon = model.epsilon_decay(epsilon, traverse, traverses)

        np.save(filePath, qTable)

    np.save(f"./runs/obstaclesWorld{world}", obstacles)
    np.save(f"./runs/goodStatesWorld{world}", goodStates)
    np.save(f"./runs/badStatesWorld{world}", badStates)


def exploit():
    epsilon = 0.9
    world = int(input("Enter the world to be trained"))
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
    inp = input("Training(1) or exploit(2)?")

    if inp == "1":
        train()
    else:
        exploit()


if __name__ == "__main__":
    main()
