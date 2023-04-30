import numpy as np
from API import *
import random
import visualization as v
from matplotlib import pyplot
import os
import time


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


def learn(qTable, world, mode, alpha, gamma, epsilon, goodStates, badStates, traverse, obstacles, verbose=True):
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
        for obstacle in obstacles:
            #print(obstacles, obstacle, visited)
            if tuple(obstacle) in visited:
                obstacles.remove(obstacle)
        v.updateGrid(currBoard, goodStates, badStates,
                     obstacles, int(traverse), world, location, verbose)
        # //////////////// END CODE FOR VISUALIZATION

        if mode == 'explore':
            if np.random.uniform() < epsilon:
                unexploitd = np.where(
                    qTable[location[0]][location[1]].astype(int) == 0)[0]
                exploitd = np.where(
                    qTable[location[0]][location[1]].astype(int) != 0)[0]

                if unexploitd.size != 0:
                    moveNum = int(np.random.choice(unexploitd))
                else:
                    moveNum = int(np.random.choice(exploitd))
            else:
                moveNum = np.argmax(qTable[location[0]][location[1]])

        # If exploiting, we're choosing the move with the highest Q value for that position in the world
        else:
            moveNum = np.argmax(qTable[location[0]][location[1]])

        # make the move - transition into a new state
        moveResponse = makeMove(world, numToMove(moveNum))

        if verbose:
            print("moveResponse", moveResponse)
        # OK response looks like {"code":"OK","world":0,"runId":"931","reward":-0.1000000000,"scoreIncrement":-0.0800000000,"newState":{"x":"0","y":3}}

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

            '''
            NOTE: note sure any of this is needed but will leave it here for now until further information is found
            '''

            # Get expected location
            expectedLoc = list(location)  # since tuple cannot be changed

            # convert the move we tried to make into an expected location where we think we'll end up (expected_loc)
            recentMove = numToMove(moveNum)

            if recentMove == "N":
                expectedLoc[1] += 1
            elif recentMove == "S":
                expectedLoc[1] -= 1
            elif recentMove == "E":
                expectedLoc[0] += 1
            elif recentMove == "W":
                expectedLoc[0] -= 1

            expectedLoc = expectedLoc

            if verbose:
                print(f"New Loc: {newLoc} (where we actually are now)")
                print(
                    f"Expected Loc: {expectedLoc} (where we thought we were going to be)")

            if (mode == "explore") and newLoc != expectedLoc:
                obstacles.append(expectedLoc)

            '''
            NOTE: not sure if this needs to be
            Why would append newLoc if the newLoc has been visited before?
            '''
            # continue to track where we have been
            visited.append(newLoc)

            # if we placed an obstacle there in the vis, remove it
            for obstacle in obstacles:
                if tuple(obstacle) in visited:
                    obstacles.remove(obstacle)

        else:
            # we hit a terminal state
            terminalState = True
            print(
                "\n\n--------------------------\nTERMINAL STATE ENCOUNTERED\n--------------------------\n\n")

        # Calculate reward
        reward = float(moveResponse["reward"])
        rewardsAcquired.append(reward)

        '''
        NOTE: not sure why only update Q values if explore,
        since the point of RL is that we keep learning,
        even if we're traversing a path known
        '''
        if mode == 'explore':
            updateQTable(location, qTable, reward, gamma,
                         newLoc, alpha, moveNum)

        # update our current location variable to our now current location
        location = newLoc

        # if we are in a terminal state then we need to collect the information for our visualization
        # and we need to end our current training traverse
        if terminalState:
            print(f"Terminal State REWARD: {reward}")

            '''
            NOTE: I just realized at this point that aside from this, we're not taking
            into account the good or bad states at all when choosing a move.
            Something we can look into tomorrow to implement for a more successful algo
            '''

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
                         obstacles, int(traverse), world, location, verbose)
            break

    # possibly not needed but this seperates out the plot
    pyplot.figure(2, figsize=(5, 5))

    # cumulative average for plotting reward by step over time purposes
    cumulativeAverage = np.cumsum(
        rewardsAcquired) / (np.arange(len(rewardsAcquired)) + 1)

    # plot reward over each step of the agent
    v.plotLearning(world, int(traverse), cumulativeAverage)

    return qTable, goodStates, badStates, obstacles


def epsilonDecay(epsilon, traverse):
    if traverse < 5:
        epsilon = epsilon * np.exp(-.1 * traverse)
    else:
        epsilon = epsilon * np.exp(-.01 * traverse)

    print(f"\nNEW EPSILON: {epsilon}\n")

    return epsilon


def alphaDecay(alpha, epoch):
    decayRate = 0.1
    alpha *= (1 / (1 + decayRate * epoch))

    print(f"\nNEW ALPHA: {alpha}\n")

    return alpha


'''
MORE NOTES:
-Alpha decay: done
-It seems that they're not taking into account of when we're at the border of the map
This makes it inefficient since it might randomly choose to go left when it's already
at the border of the map and then adding that to "badStates" (which I don't think is
even a necessary record).
'''
