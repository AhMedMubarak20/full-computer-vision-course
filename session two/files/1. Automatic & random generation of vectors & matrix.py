
# Vectors and Matrices from List
import numpy
num = [[1,2,3],[2,3,5]]
print(type(num))
nu = numpy.array(num)
print(type(nu))

# # Automatic Creation of Vectors and Matrices
# x = numpy.arange(0,10,1)
# a,b = numpy.mgrid[0:5,0:5]
# n = numpy.zeros(6)
# k = numpy.ones(6)
# zm = numpy.zeros((6,3))
# km = numpy.ones((6,3))

# #Random Generation and Identity Matrix
# y= numpy.linspace(0,10,25)
# r = numpy.random.rand(5,3)
# rr = numpy.random.randn(5,3)
# i = numpy.eye(5)
# rand = numpy.random.randint(1,50,20)