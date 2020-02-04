#problem1
hour = int(input("Enter the hour: " ))
rate = int(input("Enter the rate: "))
pay = hour*rate
print("Pay:{}".format(pay))
#problem2
number1 = int(input("Enter the first numnber: " ))
number2 = int(input("Enter the second number: " ))
number3 = number1+number2
print("The total is {}".format(number3))
#problem3
number1 = int(input("Enter the first numnber: " ))
number2 = int(input("Enter the second number: " ))
number3 = int(input("Enter the third numnber: " ))
total = number1+number2+number3
print("The answer is {}".format(total))
#problem4
start = int(input("How many slices of pizze did you start with? " ))
eaten = int(input("How many slices of pizza have you had?"))
left = start-eaten
print("The number of slices of the pizza left {}".format(left))

#problem 5
name = input('Enter your name:')
age = int(input('Enter your age (integer):'))
next_age = age+1
print("{name} will be {age}".format(name = name, age = next_age))

#problem 6
bill =float(input("Enter the total price:"))
num_ppl = int(input("Enter the number of diners"))
per_person = bill/num_ppl
print("Each person has to pay :${}".format(per_person))

#Problem 7
days = float(input('Enter the total number of days:'))
hours = days * 24
minutes = hours * 60
seconds = minutes * 60
print('There are {a} hours, {b} minutes, and {c} seconds in {d} days!'.format(a=hours,b=minutes,c=seconds,d=days))

#Problem 8
kg = float(input("Please enter a weight in kilogram:"))
pound = kg*2.204
print("{kg} KG is {pound} Pound".format(kg = kg,pound=pound))
#Problem9
larger=int(input('Enter a number over 100: '))
smaller =int(input('Enter a number under 10: '))
answer = larger//smaller
print(smaller,'goes into', larger,answer,'times')

#problem 10
full = input("What's your full name?")
length = len(full)
print("The length of the full name is{}".format(length))

#problem 11
first = input('Enter your first name:')
last= input('Enter your last name:')
fullname = first + ' ' + last
length = len(fullname)
print('Your full name is ' + fullname)
print("The length is " + str(length))

#problem12
first = input('Enter your first name in lower case letters: ')
last = input('Enter your last name in lower case letters: ')
name = first.title() + ' ' + last.title()
print('Your full name is ' + name +".")

#problem13
song = input('What is the first line of the song? ')
length = len(song)
print("The length of the first line of the song is "+ str(length))
start = int(input('Enter a starting number: '))
end = int(input('Enter an ending number: '))
part = song[start:end]
print("The line between {start} and {end} is {part}".format(start=start, end=end,part=part))

#problem14
word = input('Enter an english word: ')
word = word.upper()
print(word)

#problem 15
num = float(input('Enter a number with lots of decimal places: '))
answer = num*2
print( 'two times of {num}  is {answer} '.format(num =num,answer=answer))
num=round(num,2)
answer=round(answer,2)
print( 'two times of {num} is {answer} (in two decimal)'.format(num =num,answer=answer))
print(str(round(num,2)) + ' multiplied by 2 is: ' + str(round(answer,2)))

#problem 16
import math
num = int(input('Enter an integer number over 500: '))
answer = math.sqrt(num)
print('The square root of', round(num,2), 'is', round(answer,2))

#problem 17
import math
print(round(math.pi,5))

#problem 18
import math
radius = int(input('Enter the radius of a circle:'))
area = math.pi*(radius**2)
print('The radius is :'+ str(radius))
print("The area of that circle is "+ str(area))

#problem 19
import math
radius = int(input('Enter the radius of a cylinder:'))
depth = int(input('Enter the depth of the cylinder:'))
area = math.pi*(pow(radius,2))
volume = round(depth*area,3)
print( 'The volume of a cylinder with a radius of {radius} and a depth of {depth} is{volume}'.format(radius = radius, depth = depth, volume = volume))

#problem20
num1 = int(input('Enter a number:'))
num2 = int(input('Enter another number:'))
ans1 = num1//num2
ans2=num1%num2
print(num1, 'divided', num2, 'is', ans1)
print('The remaining is', ans2)


