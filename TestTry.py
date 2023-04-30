from main import printQTable
import numpy as np
import os
from model import alphaDecay, epsilonDecay
import math

'''
inp = input("Explore(1) or exploit(2)?")


filepath = f"./test/input{inp}/"

filename = filepath + f'testfile{inp}.npy'

if not (os.path.exists(filepath)):
    os.makedirs(filepath)
if not (os.path.isfile(filename)):
    np.save(filename, np.zeros((3, 3, 3)))

data = np.load(filename)

print(filename)
print(data)

'''
'''
world = 0

filename = f"./runs/obstaclesWorld{world}.npy"


filename = f"./runs/qTable{world}.npy"
qTable = np.load(filename)
#print(qTable)
for x in range(40):
    for y in range(40):
        #if x == y:
        #qTable[x][y][x % 4 - 1] = x +
        print(f'x={x}, y={y}')
        print(qTable[x][y], end="  ")

    print()
#qTable = np.zeros((40, 40, 4))
#np.save(filename, qTable)
'''
'''
def foo(qTable, pos, value):
    x, y, z = pos
    qTable[x][y][z] = value


inp = 1
filepath = f"./test/input{inp}/"

filename = filepath + f'testfile{inp}.npy'

qTable = np.load(filename)
#qTable = np.zeros((4, 4, 4))
pos = (3, 2, 1)
#print(qTable)

#foo(qTable, pos, 10)

print(qTable)
#np.save(filename, qTable)

'''


# Another learning rate decay function
def decayAlpha(epoch):
    initialRate = 0.9
    drop = 0.5
    epochsDrop = 5
    alpha = initialRate * \
        math.pow(drop, math.floor((1 + epoch) / epochsDrop))
    return alpha

# Another epsilon decay function


def decayEpsilon(epsilon, epoch):
    epsilonEnd = 0.001
    decay = math.pow((epsilonEnd / epsilon), (1 / epoch))
    epsilon *= decay
    return epsilon


#printQTable(0)
world = 2
parameters = np.load(f"./runs/world{world}/parameters{world}.npy")

gamma, epsilon, alpha, epoch = parameters
print(epsilon, alpha, epoch)
epsilon = 0.8
#alpha = 0.1
#epoch = 1

#for i in range(int(epoch), 11):
#epsilon = epsilonDecay(epsilon, i)

#np.save(f"./runs/world{world}/parameters{world}.npy",
#[gamma, epsilon, alpha, epoch])
