list1 = [1,2,2,3,4,5,6,7,8,9]
print(list1)
print("---------------------------------------")
print(list1[0])
print("---------------------------------------")
print(list1[0:4])
print("---------------------------------------")
print(list1[:5])
print("---------------------------------------")
list1[4]=22
print(list1)
print("---------------------------------------")
del list1[0]
print(list1)
print("---------------------------------------")
del list1
# print(list1) # error because we delete it
print("---------------------------------------")
name = ["ahmed","mohamed","ibrahim","ali"]
age = [18,25,33,66]
print(name+age)
print(name*3)
print(len(name))
print(3 in name)
print("ahmed" in name)
print("---------------------------------------")
num = [11,12,13,14,15,16,17,18,19]
print(min(num))
print((max(num)))
num.append(20)
print(num)
num.append(11)
print(num.count(11))
x = [5,7,8,9]
num.extend(x)
print(num)
print(num.index(5))
num.insert(2,"ahmed")
print(num)
num.remove("ahmed")
print(num)
num.reverse()
print(num)
num.sort()
print(num)