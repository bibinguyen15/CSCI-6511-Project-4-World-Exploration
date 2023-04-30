from matplotlib import pyplot
import matplotlib.pyplot as plt
import os
import random
import time


def updateGrid(data, goodStates, badStates, obstacles, epoch, world, location, verbose):
    pyplot.figure(1)
    pyplot.clf()
    pyplot.imshow(data)
    pyplot.draw()
    pyplot.title(f'WORLD: {world} EPOCH: {epoch}')
    pyplot.ylim(-1, 41)
    pyplot.xlim(-1, 41)
    for z in obstacles:
        pyplot.plot(z[0], z[1], marker="s", color='k')
    for x in goodStates:
        pyplot.plot(x[0], x[1], marker="P", color='b')
    for y in badStates:
        pyplot.plot(y[0], y[1], marker="X", color='r')
    pyplot.plot(location[0], location[1], marker="*", color='indigo')

    # use verbosity to adjust the visibility of the plot
    if verbose:
        pyplot.show(block=False)
        pyplot.pause(0.0001)

    if not os.path.exists("./runs/world{}/visuals/".format(world)):
        os.makedirs("./runs/world{}/visuals/".format(world))
    pyplot.savefig(
        "./runs/world{}/visuals/epoch{}.png".format(world, epoch))


def plotLearning(world, epoch, cumulativeAverage):
    plt.figure(2)
    plt.plot(cumulativeAverage)
    plt.xscale('log')
    if not os.path.exists(f'runs/world{world}/visuals/'):
        os.makedirs(f'runs/world{world}/visuals/')
    plt.savefig(
        f'runs/world{world}/visuals/epoch{epoch}learning.png')
