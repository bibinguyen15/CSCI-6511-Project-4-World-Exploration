import numpy as np
from API import *
import random
import visualization as v
from matplotlib import pyplot
import os
import time


def learn(qTable, world, alpha, gamma, epsilon, goodStates, badStates, traverse, verbose=True):
    '''
    ACTIVE learning function where we are traversing the world
    '''
    # Spawning into the world
    spawn = enterWorld(world)

    if verbose:
        print("Spawn: ", spawn)

    # If spawning failed, that means we're still in some world
    if spawn['code'] != 'OK':
        message = spawn['message']
        i = -1
        while message[i].isdigit():
            i -= 1
        worldNum = int(message[i + 1:])
        if worldNum != world:
            print(
                f'You are currently in world {worldNum}, would you like to continue exploring that instead?')
            loadPrevious = input(
                f"If yes, enter any key; otherwise, to start exploring world{world} instead, leave blank: ")
            if not loadPrevious:
                resetTeam()
                enterWorld(world)

    # if successfully get into world, or currently already in that world:
    locResponse = getLocation()
    if verbose:
        print("locResponse", locResponse)

    if locResponse['code'] != 'OK':
        print(
            f"getLocation() failed\nResponse: {locResponse}")
        return -1

    location = int(locResponse["state"].split(':')[0]), int(
        locResponse["state"].split(':')[1])  # location is a tuple (x, y)

    # Initialize terminal states and state tracker
    terminalState = False
    good = False

    # accumulate the rewards so far for plotting reward over step
    rewardsAcquired = []

    # create a list of everywhere we've been for the viz
    visited = []

    # SET UP FIGURE FOR VISUALIZATION.
    pyplot.figure(1, figsize=(10, 10))
    currBoard = [[float('-inf')] * 40 for temp in range(40)]

    # keep track of where we've been for the visualization
    visited.append(location)

    while True:
        # ////////////////// CODE FOR VISUALIZATION
        currBoard[location[1]][location[0]] = 1
        for i in range(len(currBoard)):
            for j in range(len(currBoard)):
                if (currBoard[i][j] != 0):
                    currBoard[i][j] -= .1

        v.updateGrid(currBoard, goodStates, badStates,
                     int(traverse), world, location, verbose)
        # //////////////// END CODE FOR VISUALIZATION

        if np.random.uniform() < epsilon:
            unexploited = np.where(
                qTable[location[0]][location[1]].astype(int) == 0)[0]
            exploited = np.where(
                qTable[location[0]][location[1]].astype(int) != 0)[0]

            # Handling avoiding certain moves if at border
            avoid, void = -1, -1
            if location[0] == 0:
                avoid = 3  # avoid going west
            elif location[0] == 39:
                avoid = 2  # avoid going east
            if location[1] == 0:
                void = 1  # avoid going south
            elif location[1] == 39:
                void = 0  # avoid going north

            if unexploited.size != 0:
                choice = [i for i in unexploited if i
                          != avoid and i != void]
                if choice:
                    moveNum = int(np.random.choice(choice))
                else:
                    moveNum = int(np.random.choice(exploited))
            else:
                moveNum = int(np.random.choice(exploited))

        # If exploiting, we're choosing the move with the highest Q value for that position in the world
        else:
            moveNum = np.argmax(qTable[location[0]][location[1]])

        # make the move - transition into a new state
        moveResponse = makeMove(world, numToMove(moveNum))

        if verbose:
            print("moveResponse", moveResponse)

        if moveResponse["code"] != "OK":
            print(
                f"makeMove() failed\nResponse: {moveResponse}")

            moveFailed = True
            while moveFailed:
                moveResponse = makeMove(world, numToMove(moveNum))
                time.sleep(2)
                print("\n\nRetrying move...\n\n")

                if moveResponse["code"] == 'OK':
                    moveFailed = False

        # Not in terminal state
        if moveResponse["newState"] is not None:
            newLoc = int(moveResponse["newState"]["x"]), int(
                moveResponse["newState"]["y"])

            # continue to track where we have been
            visited.append(newLoc)

        else:
            # we hit a terminal state
            terminalState = True
            print(
                "\n\n--------------------------\nTERMINAL STATE\n--------------------------\n\n")

        # Calculate reward
        reward = float(moveResponse["reward"])
        rewardsAcquired.append(reward)

        updateQTable(location, qTable, reward, gamma,
                     newLoc, alpha, moveNum)

        # update our current location variable to our now current location
        location = newLoc

        # if we are in a terminal state then we need to collect the information for our visualization
        # and we need to end our current training traverse
        if terminalState:
            print(f"Terminal State REWARD: {reward}")

            if reward > 0:
                # we hit a positive reward so keep track of it as a good reward terminal-state
                good = True
            if not(location in goodStates) and not(location in badStates):
                # update our accounting of good and bad terminal states for the visualization
                if good:
                    goodStates.append(location)
                else:
                    badStates.append(location)

            # update our visualization a last time before moving onto the next traverse
            v.updateGrid(currBoard, goodStates, badStates,
                         int(traverse), world, location, verbose)
            break

    # possibly not needed but this seperates out the plot
    pyplot.figure(2, figsize=(5, 5))

    # cumulative average for plotting reward by step over time purposes
    cumulativeAverage = np.cumsum(
        rewardsAcquired) / (np.arange(len(rewardsAcquired)) + 1)

    # plot reward over each step of the agent
    v.plotLearning(world, int(traverse), cumulativeAverage)

    return qTable, goodStates, badStates


def numToMove(num):
    '''
    Converting each position in np array into direction
    '''
    if num == 0:
        return 'N'
    elif num == 1:
        return 'S'
    elif num == 2:
        return 'E'
    elif num == 3:
        return 'W'

    # return False
    return 'ERROR!'


def updateQTable(location, qTable, reward, gamma, newLoc, alpha, move):
    '''
    new Q(s,a) = (1-alpha)* Q(s,a) + alpha * [R(s,a,s') + gamma * maxQ(s',a')]
    '''

    sample = reward + gamma * qTable[newLoc[0], newLoc[1], :].max()

    newQ = (1 - alpha) * qTable[location[0], location[1],
                                move] + alpha * sample

    # update q_table with new value
    qTable[location[0], location[1], move] = newQ


def epsilonDecay(epsilon, traverse):
    if traverse < 5 or epsilon > 0.15:
        epsilon = epsilon * np.exp(-.11 * traverse)
    # elif epsilon > 0.1:
    else:
        #print("Small decay")
        epsilon = epsilon * np.exp(-.01 * traverse)

    print(f"\nEPSILON: {epsilon}\n")

    return epsilon


def alphaDecay(alpha, epoch):
    decayRate = 0.1
    alpha *= (1 / (1 + decayRate * epoch))

    print(f"\nALPHA: {alpha}\n")

    return alpha

