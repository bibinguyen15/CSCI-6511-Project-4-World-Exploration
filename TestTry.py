
import numpy as np
import os

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

world = 0
#filename = f"./runs/qTable{world}.npy"
filename = f"./runs/obstaclesWorld{world}.npy"
#qTable = np.load(filename)
obstacles = np.load(filename)
obstacles = list(obstacles)

print(obstacles)
#for x in range(len(obstacles)):
#obstacles[x] = tuple(obstacles)

print(obstacles)

'''
#print(qTable)
for x in range(len(obstacles)):
    for y in range(len(obstacles)):
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

