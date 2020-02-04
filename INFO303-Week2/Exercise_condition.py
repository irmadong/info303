#problem 1
try:
   hours = float(input('Enter Hours Worked:'))
   rate = float(input('Rate Per Hour:'))
   pay = hours*rate
   #several different options to print the string
   print('Pay:',pay)

except:
   print('Error, please enter numeric input.')

#problem2
num1 = int(input('Enter an integer number:'))
num2 = int(input('Enter another integer number:'))
if num1 > num2:
   print(str(num2)+","+ str(num1))
else:
   print(str(num1)+","+str(num2))

#problem3
num = int(input('Enter an integer value less than 20: '))
if num >= 20:
   print('Too High')
else:
   print('Thank you')


#problem4
num = int(input('Enter an integer value between 10 and 20 (inclusive): '))
if num >= 10 and num <= 20:
   print('Thank you')
else:
   print('Incorrect answer')

#problem5
color=input("Enter the color:").casefold()
if color == "red":
    print("I like red too")
else:
    print("I don't like", color,"I prefer Red")

#problem6
rain = input("Is it raining?").lower()
if rain == "yes":
    wind = input("Is it windy?").lower()
    if wind == "yes":
        print("Its too windy for an umbrella")
    else:
        print("Take an umbrella")
else:
    print("Enjoy your day")

#problem 7
try:
   age = float(input('Enter your age:'))
   if age >= 18:
      print('You can vote!')
   elif age == 17:
      print('You can learn to drive.')
   elif age == 16:
      print('You can buy a lottery ticket')
   else:
      print('You can go Trick-or-Treating')
except:
   print('Please enter a numeric value!')

#problem 8
num =int(input('Enter an integer number between 10 and 20:'))
if num < 10:
   print('Too low')
elif num <=20:
   print('Correct')
else:
   print('Too High')

#problem 9
num =int(input('Enter a 1, 2, or 3:'))
if num == 1:
   print('Thank you')
elif num == 2:
   print('Well done')
elif num == 3:
   print('Correct')
else:
   print('Please enter a 1,2,or 3:')

#problem 10
firstname = input('Enter your first name: ')
if len(firstname) < 5:
   surname = input('Enter your surname: ')
   fullname = firstname + surname
   print(fullname.upper())
else:
   print(firstname.lower())

#problem 11
word = input('Please enter a word:')
length = len(word)
firstletter = word[0]
rest = word[1:length]
if firstletter != 'a' and firstletter != 'e' and firstletter != 'i' and firstletter != 'o' and firstletter != 'u':
   new = rest + firstletter+ 'ay'
else:
   new = word + 'way'
print(new.lower())

#problem 12
print("1) Square")
print("2) Triangle")
print()
print()
num=input("Enter a number:")
if num == "1":
    length = input("The length of the side is:")
    print('The area is ',length**2)
elif num =="2":
    base = input("Enter the base of the triangle (in integer):")
    height = input("Enter the height of the triangle (in inteher):")
    area = (base * height)/2
    print ("The base is", base)
    print("The height is ", height)
    print("The area of the triange is", area)

else:
    print("Please print 1 or 2")


