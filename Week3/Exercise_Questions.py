#Create a tuple containing the names of five countries.
#Then, display the entrie tuple
#Then, ask the user to enter one of those five countries.
#Based on the users selection, display the indxe number (i.e., position in the tuple)
#of that item in the tuple
import random

countries = ("China", "Japan", "England", "Germany", "France")
print(countries)
countryname = input("type the country name:")
print(countries.index(countryname))






#modify the previous program to ask the user to enter an integer value between 0 and 4 after
#you have printed the first message.  Then, display the country associated with that index number!

indexnum = int(input("Type the numnber:"))
print("The index is "+str(indexnum))
print(countries[indexnum])



#Create a list of sports.  Ask the user what their favorite sport is
# and append this sport to the end of the list.
#Sort the list and display it.
sports = ["Swimming" , "Running", "Diving"]
fav = input("What is your favorite sport?")
sports.append(fav)
sports.sort()
print(sports)




#Create a list of subjects that you take in school.  Display those subjects to the user.
#Ask the user which of those subjects that they don't like.
#Then, delete that subject and display the list again.
subject =["Data Structure", "Discrete Math", "AI", "Computer Organization", "OB"]
hate = input("Which of the subject do you dislike most?")
subject.pop(hate)
print(subject)




#create a dictionary of foods. Have the keys be integers and the values be the name of the food.
#Then, print out the key & value pairs in the dictionary.
#Then, ask the user which item they want to delete.
#Then, delete it and display the dictionary again (but just the values/not the keys).
# This time sort the items in the dictionary.
foods={1:"Burger", 2:"Sandwich", 3:"Bbq"}
print(foods.items())
dislike=int(input("Which item do you wanna delete?"))
foods.pop(dislike)
print(foods.values())
sorted(foods.values())





#Create an array which will store a list of integers.
# Generate five random integers and store those random integers in the array
#Display the array, showing each item on a separate line.
arr = []
for i in range(5):
    x = random.randint(0,10)
    arr.append(x)
    print(x)




#Set a total variable to 0 to start with
#while the total is 50 or less, ask the user to input a number.
#Add that number to the total variable and print a message such as
#'The total is ...{total}. Exit the loop when the total is greater
#than 50

total = 0
while total <= 50:
    num = float(input("Input a number: "))
    total = total + num
    print("The total is " + str(total))


#Ask the user to enter a number.
#Then, ask the user to enter another number.
#Add those two numbers together and then ask if they want to add another number.
#if they enter 'y', ask them to enter another number.
#Keep adding these numbers until they do not answer y.
#Once the loop is stopped, display the total

num1 = float(input("Enter number 1:"))
num2 = float(input("Enter number 2:"))
total2 = num1+num2
ans = input("Do you wanna enter another number? ")
while ans is "y" :
    num3 = float(input("Enter another num"))
    total2 += num3
    ans = input("Do you wanna enter another number? ")
print("The total is " + str(total2))


#Create a variable compnum and set the value to 50.  Ask the user to enter a number.
#While the user's guess is not correct, tell them whether their guess is too high or too low.
#Then, ask them to guess again.  Once they enter the same number as the value stored in the compnum variable,
#display the message, 'Well done, you took {count} attempts to guess the correct number.'

compnum = 50
num1 =float(input("Enter a number:"))
count = 0
while num1 != compnum:
    if num1 > compnum :
        print('Too high')
    else:
        print("Too low")
    count+=1
    num1 = float(input("Enter a number:"))

print("Well done, you took {} attempts to guess the correct number".format(count))




#loop over numbers 7 to 19 and print out that number
for i in range(7,20):
    print(i)





#Loop over the odd numbers between 1 and 10 and print out that number
for i in range (1,11,2):
    print(i)


#Starting with number 10, print out 10, 7 and 4 (negative step using the range function)
for i in range (10,3,-3):
    print (i)



#Ask the user to enter their name and then display their name three times

name = input("Enter your name: ")
for i in range (0,3):
    print(name)

#Modify the previous program to ask the user how many times they want you to
#display their name. Please account for the possibility that the user will enter
#something other than an integer in the 'how many times' input box using a try/except block

name = input("Enter your name: ")
try:
    times = int(input("How many times do you want to display their name?"))
    for i in range (0,times):
        print(name)
except:
    print("Print the integer plz")

#Ask the user to enter their name. Then, ask the user to enter an integer number
#Display their name (one letter at a time on each line) and repeat this for the number of
#times they entered in the second user input.
# Also, use a try/except block to handle bad/invalid user inputs
times = 0

try:
    times = int(input("How many times do you want to display their name?"))
    name = input("Enter your name: ")
    for i in range (0,times):
        for x in range (0,name.len()):
            print(name.index(x))

except:
    print('Enter the integer plz') #bug???


#Modify the previous program.  If the number is less than 10, then display their name that number of times.
#Otherwise, display the message, 'Too high of a number!'



#Ask the user to enter a number between 1 and 15 and then display the multiplication table for that number
#Nest your proram in a try/except block
try:
    num = int(input("Enter a number between 1 and 15: "))
    for i in range (1,11):
        print(num,'x',i,'=',num*i)
except:
    print("Plz enter the integer between 1 and 15")



#Ask for a number below 50 and then count down from 50 to that number.
#Display the number that the user entered in the last iteration???

num = int(input('Enter the number below 50:'))
while i != num:
    num = num-1


#Set a variable called total to 0.  Then, ask the user to enter five numbers.
# After each user input, ask them if they want that number included in the total.
# If they do, then add that number to the total.  If they do not want it included,
#then don't add it to the total.  After they have entered all five numbers, display the total



#Ask the user if they want to count up or down?
#If they select up, then ask them for the top number and count from 1 to that number
#if they select down, then ask them to enter a number below 20 and then count down from 20
#to that number.  If they entered something other than up or down, display a message
#'I don't understand!'



#In the following list, print out the second indexed list, then print out the first element in that list
#The first print statement should display [3,8,5] and the next statement should display the 3.
my_list = [[2,9,7],[4,7,9],[3,8,5],[7,97,65]]




#Using the following dictionary, which contains multiple dictionaries,
#have the user select a sales person and then loop over the regions for that selected sales person
#and display the region and the associated sales volume (only for the selected sales person).
#The message should read something like 'Tom sold 345 units in the S region.'  The message
#should be customized for the sales representative and each region!
sales = {'Jon': {'N':434, 'S':365, 'E':467, 'W':987},
'Tom': {'N':765, 'S':345, 'E':967, 'W':417},
'Sally': {'N':800, 'S':405, 'E':707, 'W':712},
'Sue': {'N':555, 'S':333, 'E':963, 'W':129},
         }

