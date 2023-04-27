
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

if input("Enter blank to not go in if statement"):
    print("Youre in if statement")


