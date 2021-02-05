import numpy as np

#1D vector slicing and indexing
nn = np.random.randint(1,50,20)
arr = np.arange(25)

print(arr)
print(arr[1])
print(nn)
print(nn[1])
print(arr[1:4])
print(arr[:5])
print(arr[5:])

arr[1:4] = 100
sliceRand = nn[:8]
sliceRand [:] = 99

sliceRand = nn[:8].copy()
sliceRand [:] = 99

#2D Matrix slicing and indexing
rd = np.random.rand(5,5)

print(rd[0])
print(rd[0][0])
print(rd[0,1])
print(rd[:2,1:3])

res = rd > 0.2
print(rd > 0.2)

cond = rd[res]
print (rd[res])

# res = rd[rd > 0.2]