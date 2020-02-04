#new comments
#making a change locally to this file
print("This class is wonderful")
#Without using any variables, ask the user for their
#occupation and their full name.
#Then, print out a user-friendly message
#containing their name and occupation.
#Print out the user-friendly message using several
#different types of print statements (i.e., use multiple different
#methods to generate your answer)
import math
import calendar
import time

print(input("Enter the occupation") + " " + input("Enter the full name"))
print ("Enter the occupation:")
occu = input()
print("Enter the full name: ")
fn = input()
print("The occupation is " + occu + "The full name is " + fn)

#Print out the remainder of 25 divided by the product of 3 multiplied by 2
#Again, do this without using a variable

print(25%(3*2))

#Modify the previous statement to replace the 2 in the previous equation
# with a number provided by a user.  In the input, ask the user to enter an integer
# less than or equal to 5.  Again, do this without defining any variables.

print(25%(3*int(input("Enter a integer less or equal to 5")))) # run the highlighted line shift alt E

#Print out the log of 50 while asking the user to enter a base that is less than or equal to 10
#For this question, create a variable named x to hold the input from the user.
#Reminder:
# a logarithm answers the question: How many of one number do we multiply to get another number?
#For instance, log base 2 of 8 is the number of 2s we need to multiple to get 8, which is 3!
print("Enter a base that is lee than or equal to 10: ")
x = int(input())
print(math.log(50,x))




#Print out a mathematical expression that shows that the previous statement is working correctly

print(int(math.pow(x,math.log(50,x))))

#print off a statement that displays pi and e both rounded to five decimal places.
#Please do this without defining and using any variables.

print(round(math.pi,5))
print(round(math.e,5))

#print off a user-friendly statement that displays the data type (class type) of natural log of 77
#as well as the actual natural log of 77 (rounded off to a reasonable number of decimal places).
#Please do this without using any variables (i.e., just a print statement).
print("The the data type (class type) of natural log of 77 is " + type(math.log(77,math.e)))
print("The result of natural log of 77 is " + math.log(77,math.e))

#Print off the current calendar for January 2020 without using any variables.
#Hint: you the calendar module with the month function might be helpful to use here!

print(calendar.month(2020,1))


#Print off the current local time.  You printout should look similar to the following:
#The current local time is 'Mon Jan 20 14:02:26 2020'
print("The current local time is " + time.asctime(time.localtime(time.time())))
