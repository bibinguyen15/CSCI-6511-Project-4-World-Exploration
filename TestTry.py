import model
import numpy as np
import os


def explore():
    goodStates = []
    badStates = []
    obstacles = []
    epsilon = 0.9
    world = int(input("Enter the world to be trained"))
    traverses = int(input("How many traverses?"))
    verbose = str(input("Verbose? 'Y' or 'N'"))
    qTable = model.tableInitiate()

    if not (os.path.exists(f'./runs/world{world}/')):
        os.markdirs(f'./runs/world{world}/')

    return


def exploit():
    return


def main():
    inp = input("Explore(1) or exploit(2)?")

    if inp == "1":
        explore()
    else:
        exploit()


if __name__ == "__main__":
    main()
