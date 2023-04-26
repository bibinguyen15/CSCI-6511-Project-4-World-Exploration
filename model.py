import numpy as np
from API import *
import random
import movement_viz as v
from matplotlib import pyplot
import os


def tableInitiate():
    return (np.zeros((40, 40, 4)))


def numToMove(num):
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


def updateQTable(location, qTable, reward, gamma, newLoc, learningRate, moveNum):
    '''
    bellman eq: NEW Q(s,a) = Q(s,a) + learning_rate * [R(s,a) + gamma * maxQ'(s',a') - Q(s,a)]
    '''

    # collecting the current understanding of the best q value based upon our new location, weight it by gamma and add reward
    # sample = reward + gamma * \
    # qTable[newLoc[0], newLoc[1], :].max() - qTable[location[0],
    # location[1], moveNum]
    sample = reward + gamma * qTable[newLoc[0], newLoc[1], :].max()

    # use the previous location to
    newQ = (1 - learningRate) * qTable[location[0], location[1],
                                       moveNum] + learningRate * sample

    # update q_table with new value
    qTable[location[0], location[1], moveNum] = newQ


def learn(qTable, worldId=0, mode='explore', learningRate=0.001, gamma=0.9, epsilon=0.9, goodTermStates=[], badTermStates=[], traverse=0, obstacles=[], runNum=0, verbose=True):
    '''
    ~MAIN LEARNING FUNCTION~
    takes in:
    -the Q-table data structure (numpy 3-dimensional array)
    -worldID (for api and plotting)
    -mode (explore or exploit)
    -learning rate (affects q-table calculation)
    -gamma (weighting of the rewards)
    -epsilon (determines the amount of random exploration the agen does)
    -good_term_states
    -bad_term_states
    -eposh
    -run number
    -verbosity

    returns: q_table [NumPy Array], good_term_states [list], bad_term_states [list], obstacles [list]


    '''

    # create the api instance
    #a = api.API(worldId=worldId)
    #wRes = a.enter_world()
    wRes = enterWorld(worldId)

    if verbose:
        print("wRes: ", wRes)

    # init terminal state reached
    terminalState = False

    # create a var to track the type of terminal state
    good = False

    # accumulate the rewards so far for plotting reward over step
    rewardsAcquired = []

    # find out where we are
    locResponse = getLocation()

    # create a list of everywhere we've been for the viz
    visited = []

    if verbose:
        print("locResponse", locResponse)

    # OK response looks like {"code":"OK","world":"0","state":"0:2"}
    if locResponse['code'] != 'OK':
        print(
            f"something broke on getLocation() call \nresponse looks like: {locResponse}")
        return -1

    # convert JSON into a tuple (x,y)
    location = int(locResponse["state"].split(':')[0]), int(
        locResponse["state"].split(':')[1])  # location is a tuple (x, y)

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
            if obstacle in visited:
                obstacles.remove(obstacle)
        v.update_grid(currBoard, goodTermStates, badTermStates,
                      obstacles, runNum, traverse, worldId, location, verbose)
        # //////////////// END CODE FOR VISUALIZATION

        # in q-table, get index of best option for movement based on our current state in the world
        if mode == 'explore':
            # use an epsilon greedy approach to randomly explore or exploit
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

        else:
            # mode is exploit -we'll use what we already have in the q-table to decide on our moves
            moveNum = np.argmax(qTable[location[0]][location[1]])

        # make the move - transition into a new state
        moveResponse = makeMove(world, numToMove(moveNum))

        if verbose:
            print("moveResponse", moveResponse)
        # OK response looks like {"code":"OK","worldId":0,"runId":"931","reward":-0.1000000000,"scoreIncrement":-0.0800000000,"newState":{"x":"0","y":3}}

        if moveResponse["code"] != "OK":
            # handel the unexpected
            print(
                f"something broke on makeMove call \nresponse lookes like: {moveResponse}")

            moveFailed = True
            while moveFailed:
                moveResponse = makeMove(world, numToMove(moveNum))

                print("\n\ntrying move again!!\n\n")

                if moveResponse["code"] == 'OK':
                    move_failed = False

        # check that we're not in a terminal state, and if not convert new location JSON into tuple
        if moveResponse["newState"] is not None:
            # we're now in newLoc, which will be a tuple of where we are according to the API
            # KEEP IN MIND the movment of our agent is apparently STOCHASTIC
            newLoc = int(moveResponse["newState"]["x"]), int(
                moveResponse["newState"]["y"])  # tuple (x,y)

            # keep track of if we hit any obstacles
            expectedLoc = list(location)

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

            expectedLoc = tuple(expectedLoc)

            if verbose:
                print(f"New Loc: {newLoc} (where we actually are now):")
                print(
                    f"Expected Loc: {expectedLoc} (where we thought we were going to be):")

            if (mode == "explore"):
                obstacles.append(expectedLoc)

            # continue to track where we have been
            visited.append(newLoc)

            # if we placed an obstacle there in the vis, remove it
            for obstacle in obstacles:
                if obstacle in visited:
                    obstacles.remove(obstacle)

        else:
            # we hit a terminal state
            terminalState = True
            print(
                "\n\n--------------------------\nTERMINAL STATE ENCOUNTERED\n--------------------------\n\n")

        # get the reward for the most recent move we made
        reward = float(moveResponse["reward"])

        # add reward to plot
        rewardsAcquired.append(reward)

        # if we are exploring the model then update the q-table for the state we were in before
        # using the bellman-human algorithim
        if mode == 'explore':
            updateQTable(location, qTable, reward, gamma,
                         newLoc, learningRate, moveNum)

        # update our current location variable to our now current location
        location = newLoc

        # if we are in a terminal state then we need to collect the information for our visualization
        # and we need to end our current training traverse
        if terminalState:
            print(f"Terminal State REWARD: {reward}")

            if reward > 0:
                # we hit a positive reward so keep track of it as a good reward terminal-state
                good = True
            if not(location in goodTermStates) and not(location in badTermStates):
                # update our accounting of good and bad terminal states for the visualization
                if good:
                    goodTermStates.append(location)
                else:
                    badTermStates.append(location)

            # update our visualization a last time before moving onto the next traverse
            v.update_grid(currBoard, goodTermStates, badTermStates,
                          obstacles, runNum, traverse, worldId, location, verbose)
            break

    # possibly not needed but this seperates out the plot
    pyplot.figure(2, figsize=(5, 5))

    # cumulative average for plotting reward by step over time purposes
    cumulative_average = np.cumsum(
        rewardsAcquired) / (np.arange(len(rewardsAcquired)) + 1)
    # plot reward over each step of the agent
    utils.plot_learning(worldId, traverse, cumulative_average, runNum)

    return qTable, goodTermStates, badTermStates, obstacles


def plot_learning(worldId, traverse, cumulativeAverage, rn):
    pyplot.figure(2)
    pyplot.plot(cumulativeAverage)
    pyplot.xscale('log')
    if not os.path.exists(f'runs/world_{worldId}/attempt_{rn}'):
        os.makedirs(f'runs/world_{worldId}/attempt_{rn}')
    pyplot.savefig(
        f'runs/world_{worldId}/attempt_{rn}/world_{worldId}_traverse{traverse}learning.png')


def epsilonDecay(epsilon, traverse, traverses):
    '''
    function to exponentially decrease the episilon value 
    acroccs the total number of traverses we train on
    this leads us to exploit less as we progress through traverses 
    '''

    epsilon = epsilon * np.exp(-.01 * traverse)

    print(f"\nNEW EPSILON: {epsilon}\n")
    return epsilon
