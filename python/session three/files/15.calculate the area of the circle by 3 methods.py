#program to find area of circle in Python using math file
import math
r = float(input("Enter the radius of the circle: "))
area = math.pi* r * r
print("%.2f" %area)

print ("--------------------------------------------------")

#program to find area of circle in Python using π
PI = 3.14
r = float(input("Enter the radius of the circle: "))
area = PI * r * r
print("%.2f" %area)

print("------------------------------------------------------")
#program to find area of circle using functions
import math

#Function that calculates area of circle
def area_of_circle(r):
    a = r**2 * math.pi
    return a

r = float(input("Enter the radius of the circle: "))
print("%.2f" %area_of_circle(r))