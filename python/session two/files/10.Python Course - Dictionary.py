search = {"name":"ahmed","age":23}
print(search)
print(search["name"])
search["name"]="mohamed"
print(search)
del search["age"]
print(search)
search.clear()
print(search)
del search
# print (search) #error because we deleted it
search = {"name":"ahmed","age":23}
print(len(search))
search = str(search)
print(type(search))
print("----------------------------------------------------------")
dict1={1,2,3,4,5,6,7,8,9}
dict2 = dict1.copy()
print(dict2)


