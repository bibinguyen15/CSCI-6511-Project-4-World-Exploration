from matplotlib import pyplot
import matplotlib.pyplot as plt
import os
import random
import time


def updatePlot(data, goodStates, badStates, epoch, world, location, verbose):
    pyplot.figure(1)
    pyplot.clf()
    pyplot.imshow(data)
    pyplot.draw()
    pyplot.title(f'WORLD: {world} EPOCH: {epoch}')
    pyplot.ylim(-1, 41)
    pyplot.xlim(-1, 41)

    for x in goodStates:
        pyplot.plot(x[0], x[1], marker="D", color='b')
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

