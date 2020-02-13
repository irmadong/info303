#These tasks represents tasks that you should probably be able to do without looking
#up how to do them on the Internet.

#1. Ask the user to enter their name and print a message that says 'Hello, {name}!
print ('Hello ' + input('Please enter your name') + '!')

#2. Modify the previous program/statement to convert the users name to title case.
print ('Hello ' + input('Please enter your name').title() + '!')

#3 Create a variable and assign the value of 6 to it.
#  Multiply that value by 22 and print out the result with a user-friendly message such as
#  The value of 6 multiplied by 22 is 132.
x = 6
print(f'The value of {x} multiplied by 22 is {x*22}.')

#4. Ask the user to enter two integer values.
#   Perform a floor division of the first number (numerator) and the second number (denominator)
#   Then, print out the result
x = int(input('Please enter an integer:'))
y = int(input('Please enter another integer:'))
z = x // y
print('The floor division of {a} and {b} is {c}'.format(a=x, b=y, c=z))

#5. Ask the user for two integer values.
#   Divide the first number by the second number.
#   Then, print off a message that reads something like the following:
#   19 divided by 10 equals 1 with a remainder of 9.
a = int(input('Please enter an integer:'))
b = int(input('Please enter another integer:'))
c = a%b
d = a//b
print(F'{a} divided by {b} equals {d} with a remainder of {c}.')

#6. In the previous program, if the user enters a string (text) instead of integers,
#   print a message (instead of the traceback message) such as the following:
#   Please enter integers instead of text.
try:
    a = int(input('Please enter an integer:'))
    b = int(input('Please enter another integer:'))
    c = a % b
    d = a // b
    print(F'{a} divided by {b} equals {d} with a remainder of {c}.')
    #alternative
    print(str(a) + ' divided by ' + str(b) + ' equals ' + str(d) + ' with a remainder of ' + str(c))
except:
    print('Please enter integers instead of text.')

#7. Ask the user to enter a sentence.
#   Print the sentence with just the first word in the sentence capitalized.
#   'Hello my name is tom.'
#   Then, print the sentence with each word capitalized.
#   'Hello My Name Is Tom.'
#   Then, print the sentence with each word in lower case and again with each word in all capital letters.
sentence = input('Please enter a sentence.')
print(sentence.capitalize() + '.')
print(sentence.title() + '.')
print(sentence.lower() + '.')
print(sentence.upper() + '.')

#8.  Ask the user to enter a word.
#   Print a message indicating the length of the word.  For instance, enter the following message:
#   You entered 'Hello', which is 5 characters long.
word = input('Please enter a word.')
print(f'You entered {word!r}, which is {len(word)} characters long.')
print(f'You entered \'{word}\', which is {len(word)} characters long.')

#9.  Set a variable equal to the following string: 'We are learning python to eventually learn machine learning.'
#    Then, print off just the word Python with a capital P.
#    Then, print off the words machine learning with a capital M and capital L.
sentence = 'We are learning python to eventually learn machine learning.'
print(sentence[16:22].title())
print(sentence[43:].title())
#Without the period
print(sentence[43:59].title())

#10. Create a variable with the following string:
#    I told my friend, "Python is my favorite language!" with the quotation marks
#    Then, print out that variable.
#    Next, reassign the value of your variable to:
#    The language of 'Python' is named after Monty Python, not the snake with Python in single quotations.
#    Then, print out that variable.
#    Next, reassign the value of your variable to:
#    One of Python's strengths is its diverse and supportive community with the apostrophe.
#    Then, print out that variable.
phrase = 'I told my friend, "Python is my favorite language!"'
print(phrase)
#Option 2 with escape characters
phrase = "I told my friend, \"Python is my favorite language!\""
print(phrase)
phrase = 'The language of \'Python\' is named after Monty Python, not the snake with Python.'
print(phrase)
phrase = "The language of 'Python' is named after Monty Python, not the snake with Python."
print(phrase)
phrase = "One of Python's strengths is its diverse and supportive community."
print(phrase)
phrase = 'One of Python\'s strengths is its diverse and supportive community.'
print(phrase)

#11. Create a variable to store my first name ('Tom') and a second variable to store my last name ('Mattson')
#    Then, create a third variable that concatenates the two names together with a space between the names.
#    Then, print the third variable
first_name = 'Tom'
last_name = 'Mattson'
full_name = first_name + ' ' + last_name
print(full_name)

#12. In the previous program, instead of creating a third variable, combine the first two variables in a print
#    statement.  Try to do this multiple ways!
first_name = 'Tom'
last_name = 'Mattson'
print(first_name + ' ' + last_name)
print(f'{first_name} {last_name}')
print('%s %s' %(first_name, last_name))
print('{} {}'.format(first_name, last_name))

#13. Print the following with the tabs and the returns:
#Courses:
#   MGMT325
#   INFO201
#   INFO303
print('Courses:\n\tMGMT325\n\tINFO201\n\tINFO303')

#14.  Using the following variable:
fav_course = '          INFO 303           '
#   First, print the variable with the leading spaces stripped
#   Second, print the variable with the trailing spaces stripped
#   Thirs, print the variable with both the leading and trailing spaces stripped
print(fav_course.lstrip())
print(fav_course.rstrip())
print(fav_course.strip())

#15. Ask the user to enter two numbers.
#    Use a try/except block to handle the possibility that the user enters text instead of numbers in their response.
#    Then, add the two numbers together and print the result.
#    Then, multiply the two numbers together and print the result.
#    Then, subtract the two numbers (first - second) and multiply that difference by 7 and print out the result.
#    Then, divide the two numbers (first number is the numerator), raise that result to the fourth power
#    and print out the result.
#    For the last calculation, print out the data type.
#    Finally, add a docstring comment at the start of your answer
"""This is a multi-line
   docstring comment discussing the answer to this question"""
try:
    a = float(input('Please enter a number:'))
    b = float(input('Please enter a number:'))
    c = a + b
    print(c)
    c = a * b
    print(c)
    c = (a-b) * 7
    print(c)
    c = (a / b) ** 4
    print(c)
    print(type(c))
except:
    print('Non-numeric data entered.')

#16. Ask the user to enter a number.
#    If the number is greater than or equal to 25, print an appropriate message.
num = float(input('Please enter a number:'))
if num >= 25:
    print('The number that you entered is greater than or equal 25')

#17. Ask the user to enter a number.
#    If the number is between 15 (inclusive) and 40 (inclusive), print an appropriate message.
num = float(input('Please enter a number:'))
if num >= 15 and num <=40:
    print('The number that you entered is between 15 and 40')

#18. Ask the user to enter either tom, bill, joe, or sally.
#    If they enter, bill (in any case), then print an appropriate message.
name = input('Please enter either tom, bill, joe or sally:')
if name.lower() == 'bill':
    print (f'You entered {name}, which was the response I was looking for!!!!')

#19. Ask the user to enter either tom, bill, joe, or sally.
#    If they enter, bill (in any case) or tom (in any case), then print an appropriate message.
name = input('Please enter either tom, bill, joe or sally:')
if name.lower() == 'bill' or 'tom':
    print (f'You entered {name}, which was one of the responses I was looking for!!!!')
###or
if name.lower() == 'bill' or name.lower() == 'tom':
    print (f'You entered {name}, which was one of the responses I was looking for!!!!')

#20. Ask the user to enter two numbers.  Divide the first number by the second number.
#   If the result is greater than 4, print an appropriate message.  For instance:
#   16.0 divided by 3.0 is 5.33, which is greater than 4
#   Otherwise, print an appropriate message.  For instance,
#   16.0 divided by 4.0 is 4.0, which is not greater than 4
#   Make sure the result of the division in your message is formatted with a reasonable number of decimal places.
a = float(input('Please enter a number:'))
b = float(input('Please enter a number:'))
c = a / b
if c > 4:
    #Here are a few different ways to print out the result
    print(F'{a} divided by {b} is {c}, which is greater than 4')
    print(F'{a} divided by {b} is {round(c,2)}, which is greater than 4')
    print(F'{a} divided by {b} is {"{0:.2f}".format(c)}, which is greater than 4')
else:
    #Here are a few different ways to print out the result
    print(F'{a} divided by {b} is {c}, which is not greater than 4')
    print(F'{a} divided by {b} is {round(c,2)}, which is not greater than 4')
    print(F'{a} divided by {b} is {"{0:.2f}".format(c)}, which is not greater than 4')

#21. Ask the user to enter two numbers.  Raise the first number to the second number.
#    Using a single if conditional, check for the following:
#    If the result is between 0 and 500 (inclusive of both numbers), print an appropriate message.
#    If the result is greater than 500, print an appropriate message.
#    For all other values (i.e., negative numbers), print an appropriate message.
#A sample message for one of the conditions would be the following:
#-7.0 raised to the power of 3.0 is -343.0, which is negative (less than zero)

a = float(input('Please enter a number:'))
b = float(input('Please enter a number:'))
c = a ** b
if c >= 0 and c <= 500:
    print(F'{a} raised to the power of {b} is {round(c,2)}, which is between 0 and 500')
elif c > 500:
    print(F'{a} raised to the power of {b} is {round(c, 2)}, which is greater than 500')
else:
    print(F'{a} raised to the power of {b} is {round(c,2)}, which is negative (less than zero)')

#22. Ask the user to enter two words.
#    If the length of the first word is longer than the second word, print an appropriate message.
#    Otherwise, print an appropriate message.
# A sample message would be the following:
#The first word of 'doggy' is 5 characters long, which is longer than the second word of 'cat' that is 3 characters long.
word1 = input('Please enter a word:')
word2 = input('Please enter a second word:')
if len(word1) > len(word2):
    print(f'The first word of {word1!r} is {len(word1)} characters long, '
          f'which is longer than the second word of {word2!r} that is {len(word2)} characters long.')
else:
    print(f'The first word of {word1!r} is {len(word1)} characters long, '
          f'which is NOT greater than the second word of {word2!r} that is {len(word2)} characters long.')

#23. Ask the user to enter two words.
#    If the two words are equal (do not change the case that the user enters), enter an appropriate message.
#    Otherwise, print an appropriate message.
# A sample message would be the following:
#The first word of 'doggy' is equal to the second word of 'doggy'.
word1 = input('Please enter a word:')
word2 = input('Please enter a second word:')
if word1 == word2:
    print(f'The first word of {word1!r} is equal to the second word of {word2!r}.')
else:
    print(f'The first word of {word1!r} is not equal to the second word of {word2!r}.')

#Or
if word1 != word2:
    print(f'The first word of {word1!r} is not equal to the second word of {word2!r}.')
else:
    print(f'The first word of {word1!r} is equal to the second word of {word2!r}.')

#24. Modify the previous question to compare the two words, regardless of case.  Therefore,
#    dog should equal 'Dog' for this question.
word1 = input('Please enter a word:')
word2 = input('Please enter a second word:')
if word1.lower() == word2.lower():
    print(f'The first word of {word1!r} is equal to the second word of {word2!r}.')
else:
    print(f'The first word of {word1!r} is not equal to the second word of {word2!r}.')

#25. Ask the user to enter a word.
#    If the word contains the letter r and the letter a, then print an appropriate message.
#    Otherwise, print an appropriate message.
word1 = input('Please enter a word:')
if ('r' in word1) and ('a' in word1):
    print(F'The word {word1} contains at least one "r" and one "a".')
else:
    print(F'The word {word1} does not contain at least one "r" and one "a".')

#26. Check whether two strings are equal even if the order of words or
#    the characters are different.  To do this comparison, ignore the case of the words.
#    For instance, consider this string:
#    Str1 = “Hello and Welcome”
#    Str2 = “welcome and Hello”
#    These two should be the same!
#    If the two strings are the same, then print an appropriate message.
#    Otherwise, print an appropriate message.
#    For this question, use the following two variables!
#    After testing your code with these two strings, modify them so they are not equal.
str_a = 'Hello and welcome'
str_b = 'Welcome and Hello'
if sorted(str_a.lower()) == sorted(str_b.lower()):
    print('Both strings (after being sorted and converted to lower case) are the same.')
else:
    print('Both strings (after being sorted and converted to lower case) are NOT the same.')

#For the next series of problems, use the random module.
#27  Print the help for random.random.
#    Generate a random real number number between 0 (inclusive) and 1 (not inclusive)
#    If the number is less than 0.5, print an appropriate message.
#    Otherwise, print a different message.
import random
print(help(random.random))
x = random.random()
if x < 0.5:
    print('Computer generated pseudo random number is ' + str(round(x,5)) + ', which is less than 0.5.')
else:
    print('Computer generated pseudo random number is ' + str(round(x, 5)) + ', which is greater than or equal to 0.5.')

#28   Print the help for random.randint.
#     Generate a random integer between -15 and 15.
#     If the number is zero, print an appropriate and user-friendly message with the percent chance
#     that the number would be exactly equal to zero.
#     If the number is negative, print an appropriate and user-friendly message with the percent chance
#     that the number would be negative.
#     If the number is positive, print and appropriate and user-friendly message with the percent chance
#     that the number would be positive.  Within this positive condition, do the following:
#           Pick another random integer between 5 and 10.
#           Then, raise the first random number to the power of the second random integer.
#           Then, print the result with an appropriate and user-friendly message.

import random
print(help(random.randint))
x = random.randint(-15,15)
if x == 0:
    print(f'The computer generated pseudo random number was {x}. '
          f'There was a {round((1/31)*100,5)} percent chance of this happening!')
elif x < 0:
    print(f'The computer generated pseudo random number was {x}, which was less than zero. '
          f'There was a {round((15 / 31) * 100, 5)} percent chance of this happening!')
elif x > 0:
    print(f'The computer generated pseudo random number was {x}, which was greater than zero. '
          f'There was a {round((15 / 31) * 100, 5)} percent chance of this happening!')
    y = random.randint(5,10)
    z = x**y
    print(f'A second random integer between 5 and 10 was selected, which was {y}.')
    print(f'{x} raised to the {y} power was {"{:,.0f}".format(z)}.')

#29  Create a list containing four names.
#    Have the computer randomly choose one of those names.
#    Then, print an appropriate and user-friendly message.
my_list = ['bill','joe','sally', 'sue']
str_a = random.choice(my_list)
print('The computer randomly selected: ', str_a)

#30. Modify the previous example.  If the computer automatically selected the first or third item
#    in the list, print an appropriate and user-friendly message such as 'I was hoping for one of
#    these values was'.  Otherwise, print a different
#    user-appropriate message such as 'I was hoping for one of the other options.'
import random
my_list = ['bill','joe','sally', 'sue']
str_a = random.choice(my_list)
print('The computer randomly selected: ', str_a)
if str_a == my_list[0] or str_a == my_list[2]:
    print('I was hoping for one of these values.')
else:
    print('I was hoping for one of the other options.')

#31.  Generate a random sample (n=5) from the following list.
#    Then, print out the random sample with an appropriate and user-friendly message.!
my_list = [20,40,80,100,120,33,45,65,85, 97,109,123]
print('Choosing 5 random items from a list using sample function:', random.sample(my_list,k=5))

#32.  Modify the previous example, to ask the user to enter the sample size.
#    Check to make sure the sample size is less than the size of the list!
#    If less than the size of the list, generate the random sample.
#    Otherwise, print a message that indicates that your sample size is too big.
my_list = [20,40,80,100,120,33,45,65,85, 97,109,123]
sample_size = int(input('Please enter a sample size:'))
if sample_size < len(my_list):
    print('Choosing ' + str(sample_size) + ' random items from a list using sample function:',
          random.sample(my_list, k=sample_size))
else:
    print('You can\'t generate a random sample of ' + str(sample_size) + ', because the list only '
                                                                         'contains ' + str(len(my_list)) + ' items.')

#33.  Have the user enter three numbers.
#     Check whether the first number is greater than the second number
#     and whether the second number is greater than the number.
#     If so, print an appropriate user-friendly message.
#     If not, print an appropriate user-friendly message.
a = int(input('Enter a number: '))
b = int(input('Enter another number: '))
c = int(input('Enter a third number: '))
if a > b > c:
    print('Success. num1 > num2 > num3')
else:
    print('Failure. num1 > num2 > num3 is not True')

#34.  Ask the user to enter a number.
#     Check whether that number is not equal to zero.
#     If so, check if the number is positive and print an appropriate user-friendly message.
#     If not, print an appropriate user-friendly message.
#     If user enters a zero, then print an appropriate user-friendly message.
num = int(input('Please enter an integer value:'))
if (num != 0):
    if num > 0:
        print('Number (' + str(num) + ') is greater than zero')
    else:
        print('Number (' + str(num) + ') is less than zero')
else:
    print('User entered a zero')

#35.  Ask the user to enter a year and a month.
#     Enter the year: 2020
#     Enter the month: 4
#     Then, print a message determining how many days in the month.  For instance,
#     There are 30 days in this month
#     First, check whether we are in a leap year by checking the following:
#     Year MOD 4 equals zero and year MOD 100 not equal to zero and year MOD 400 equals zero.
#           Then, within the leap year, perform a conditional test on the month that the user entered
#           to determine the number of days in the month.
#           Also account for a user entering an invalid month (i.e., a 13 or 14).
#     If we are not in a leap year, perform a conditional test on the month that the user entered
#           to determine the number of days in the month.
#           Also account for a user entering an invalid month (i.e., a 13 or 14).
#     If the user enters an invalid year, print a message indicating that the user entered an invalid year.

currentYear = int(input('Enter the year: '))
currentMonth = int(input('Enter the month: '))

if ((currentYear % 4) == 0 and (currentYear % 100) != 0 or (currentYear % 400) == 0):
    #print ('Leap Year')
    if (currentMonth == 1 or currentMonth == 3 or currentMonth == 5
            or currentMonth == 7 or currentMonth == 8 or currentMonth == 10 or currentMonth == 12):
        print('There are 31 days in this month')
    elif (currentMonth == 4 or currentMonth == 6 or currentMonth == 9 or currentMonth == 11):
        print('There are 30 days in this month')
    elif (currentMonth == 2):
        print('There are 29 days in this month')
    else:
        print('Invalid month')
elif ((currentYear % 4) != 0 or (currentYear % 100) == 0 or (currentYear % 400) != 0):
    #print('Non Leap Year')
    if (currentMonth == 1 or currentMonth == 3 or currentMonth == 5 or currentMonth == 7
            or currentMonth == 8 or currentMonth == 10 or currentMonth == 12):
        print ('There are 31 days in this month')
    elif (currentMonth == 4 or currentMonth == 6 or currentMonth == 9 or currentMonth == 11):
        print('There are 30 days in this month')
    elif (currentMonth == 2):
        print('There are 28 days in this month')
    else:
        print('Invalid month')
else:
    print('Invalid Year')

#36. Build a list of at least four car manufacturers.
#    Print the third element in the list.
#    Print the last item in the list.  For this task, do this in multiple ways.
cars = ['ford', 'gm', 'nissan', 'toyota', 'bmw']
print(cars[2])
print(cars[4])
print(cars[-1])

#37. Using the previous list, print out the second element in the list in all capital letters.
cars = ['ford', 'gm', 'nissan', 'toyota', 'bmw']
print(cars[1].upper())

#38. Using the previous list, modify the fourth item to a different manufacturer.
#    Then, print off the new fourth item in title case.
#    Then, print off the entire list.
cars = ['ford', 'gm', 'nissan', 'toyota', 'bmw']
cars[3] = 'subaru'
print(cars[3].title())
print(cars)

#39. Using the previous list, delete the fourth item.
del cars[3]
cars.pop(3)
print(cars)

#40. Using the previous list, add a manufacturer to the last position and print the list after you add the item.
cars = ['ford', 'gm', 'nissan', 'toyota', 'bmw']
cars.append('tesla')
print(cars)

#41. Using the previous list, add a manufacturer to the second position and print the list after you add the item.
cars = ['ford', 'gm', 'nissan', 'toyota', 'bmw', 'tesla']
cars.insert(1, 'honda')
print(cars)

#42. Using the previous list, delete the last item in your list and print the list after you remove that item.
cars = ['ford', 'honda', 'gm', 'nissan', 'toyota', 'bmw', 'tesla']
cars.pop()
print(cars)

#43. Using the below list, delete the item with the value of 'tom' without referencing the indexed position.
#    When finished, print the list
names = ['bill','joe','sally','sue','tom', 'tim', 'violet']
names.remove('tom')
print(names)

#44. Using the below list, reverse the order of the list and then print the list.
#    Next, permanently sort the items in ascending order and then print the list.
#    Next, permanently sort the items in descending order and then print the list.
cars = ['ford', 'honda', 'gm', 'nissan', 'toyota', 'bmw', 'tesla']
cars.reverse()
print(cars)
cars.sort()
print(cars)
cars.sort(reverse=True)
print(cars)

#45.  Using the below list, create a variable to store the length of the list.
#     Then, try to use that variable to reference the last item in the list.  Notice how you get an error!
#     Handle that error by printing a user-friendly and appropriate message.  Then, print the last
#     element/item in the list using two different methods.
cars = ['ford', 'honda', 'gm', 'nissan', 'toyota', 'bmw', 'tesla']
length = len(cars)
try:
    print(cars[length])
except:
    print('This list is zero based so the last element cannot be referenced using the length of the list')
    print('The following two statements will print out the last item in the list')
    print(cars[length-1])
    print(cars[-1])

#46. Using the below list and variable, check if the value stored in the variable is
#    contained in the list.  If so, print an appropriate message.  If not, print an appropriate message.
#    After you find that the value is not contained in the list, build another if statement to handle the
#    difference in case (i.e., capital F instead of lower case f)
cars = ['ford', 'honda', 'gm', 'nissan', 'toyota', 'bmw', 'tesla']
car = 'Ford'
if car in cars:
    print('This car is in the list!')
else:
    print('This car is not in the list!')

if (car.lower() in cars) or (car.upper() in cars) or (car.title() in cars):
    print('This car is in the list!')
else:
    print('This car is not in the list!')

#47. Check if the following list is empty!  If so, print an appropriate and user-friendly message.
#    If not, print an appropriate and user-friendly message.
cars = ['ford', 'honda', 'gm', 'nissan', 'toyota', 'bmw', 'tesla']
if cars:
    print(f'Your list contains the following cars: {cars}')
else:
    print('Your cars list is empty')

#48.  Clear the list from the previous question and perform the is empty check again.
cars.clear()
if cars:
    print(f'Your list contains the following cars: {cars}')
else:
    print('Your cars list is empty')

#49.  Create a list of numbers from 1 to 1000 and print the values
nums = list(range(1,1001))
print(nums)

#50.  Create a list of odd numbers from 500 to 600.  Then, print the values.
#     Create a list of even numbers from 500 to 600.  Then, print the values.
nums = list(range(501,600,2))
print(nums)
nums = list(range(500,601,2))
print(nums)