age = int(input("ENTER YOUR AGE : "))
if age > 18 :
    print("good")
elif age == 18 :
    print ("acceptable")
else :
    print("bad")
print ("-----------------------------------------")
age = int(input("ENTER YOUR AGE : "))
gender = str(input("ENTER YOUR GENDER : "))
if age > 18 and gender == "m" :
    print("go")
else :
    print("back")