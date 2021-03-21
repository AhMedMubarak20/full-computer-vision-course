names=["ahmed","mohamed","ali"]
print(type(names))
print("---------------------------")
names = ("ahmed","mohamed","ali")
print(type(names))
# names[0]="ibrahim" #error because it's not support
num = (1,2,3)
print(names+num)
print(names*2)
print(min(num))
print(max(num))
# del num[0] #error because it's not support
del num
# print(num) #error because we delete it
num = (3,6,7,9,1,3,4)
num = list(num)
print(type(num))