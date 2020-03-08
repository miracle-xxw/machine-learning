from numpy import *

print(random.rand(4,4))

print("--------")
randMat = mat(random.rand(4,4))
invRandMat = randMat.I

myEye = randMat * invRandMat
print(myEye)

print(myEye - eye(4))