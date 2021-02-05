import numpy as np
nn = np.random.randint(1,50,20)
arr = np.arange(25)
ar = np.reshape(arr, (5,5))

arr.min
arr.max
arr.argmax
arr.argmin

maxnum = np.max(nn)
minnum = np.min(nn)

locmax = np.argmax(nn)
locmin = np.argmin(nn)

print(np.shape(arr))
print(np.shape(ar))
print(arr.dtype)