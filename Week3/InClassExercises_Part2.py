###Dictionaries
##A dictionary stores items in key: value pairs...similar to a physical dictionary
#Python provides a data type dictionary (or Python dict),
# which is just like the associative arrays in some other languages.
# The Dictionary data types work with key – value pairs.

#A key can be of any type of objects like a number, string or a list in Python dictionary.
#The values can be accessed by using key rather than the index like in simple arrays.

sample_dictionary = {'Mike': 8049875431,'John': 8087632514, 'Bill': 5166534213} #key:element

#As you can see in the above example:
#Each key is separated from its value by using a ‘:’.
#Each dictionary entry is separated by a comma.
#Python dictionaries are enclosed in the curly brackets, i.e. {}.

print(sample_dictionary['Mike'])

#We can loop through the dict just like a list. See example below of
# using a for loop to display employee dictionary. Note that, the
# Python dictionaries do not keep the order of values as created.

for key, value in sample_dictionary.items():
    print('Phone number for %r is %d.' % (key, value))
#NOTE: The items() function returns a tuple containing the key and the value,
#which we unpack into a kay and a value variable in each iteration of the loop.

#We can also loop over just the keys in a dictionary object
for key in sample_dictionary.keys():
    print('Name is %s (but this is kind of useless '
          'without the associated value).' % (key))

#Or, we loop over just the values in a dictionary object
for value in sample_dictionary.values():
    print('Phone number is %d (but this is kind of useless '
          'without the associated key).' % (value))

#Books dictionary made by a list of tuples, which we cast to a dictionary using the dict function
book_dict = dict([('genre', 'mystery'), ('pages', 250)])
print(book_dict['genre'])
print(book_dict['pages'])
print(book_dict.get('pages'))
#Notice the two different methods for accessing the value of a key in a dictionary

print('My book is in the %r genre and is %s pages long.' % (book_dict['genre'],
                                                                book_dict.get('pages')))
#The benefit to the get method is it won't return an error if the key does not exist in the dictionary
rating_value = book_dict['rating']
rating_value = book_dict.get('rating') #no error message
#If the key is not found in the dictionary, return Five Stars
rating_value = book_dict.get('rating','Five Stars')
print(rating_value)
print(book_dict)

#Add a key: value pair in this dictionary for rating with a value of Five Stars
book_dict['rating'] = 'Five Stars'
print('My book is in the %r genre, is %s pages long, and is rated %s' % (book_dict['genre'],
                                                                        book_dict.get('pages'),
                                                                        book_dict.get('rating')))

#We can delete a key from the dictionary
book_dict.pop('condition')
#but this will throw an error if the key does not exist in the dictionary.
#Therefore, nest this pop in a try....except block.
try:
    book_dict.pop('condition')
except:
    print('That item is not in the dictionary so we cannot pop that item!')

#We can also delete the last item in the dictionary using the popitem function
book_dict.popitem()
print(book_dict)

#We can clear any remaining items from the dictionary using the clear function
#NOTE: This function keeps the dictionary object in memory. It simply removes the key: value pairs.
book_dict.clear()
print(book_dict)

#Finally, we can delete the book_dict from memory entirely.
del book_dict
#After we delete it from memory, we can no longer reference it.  The following line of code,
#will throw an error.
print(book_dict)
try:
    print(book_dict)
except:
    print('The book_dict dictionary object has been deleted from memory')


books_dict1 = dict([('genre', 'mystery'), ('pages', 250)])
books_dict2 = dict([('genre', 'self-help'), ('pages', 75)])
books_dict3 = dict([('genre', 'technology'), ('pages', 500)])

#Let's now create a mutable list that contains these three dictionary objects
#Remember: A list may contain all different types of objects, including dictionaries!
books = [books_dict1, books_dict2, books_dict3]
print(books)





#Now, to reference the dictionaries within our list object, we use the [] to naviagte down
#to and into the respective objects.
print(books[1]['pages'])
print(books[2]['genre'])

#NOTE in this example how the authors key has a list value
books_dict1 = dict([('genre', 'mystery'), ('pages', 250), ('authors', ['joe', 'sally'])])
books_dict2 = dict([('genre', 'self-help'), ('pages', 75), ('authors', ['jim', 'tim'])])
books_dict3 = dict([('genre', 'technology'), ('pages', 500), ('authors', ['jane', 'sue'])])

#Create a list containing the three dictionary objects - Same as above!
books = [books_dict1, books_dict2, books_dict3]
print(books)

#Change the second author of the third book from sue to tom.
#Notice how we reference the first indexed item in the authors key.
#There is no real limit to how many levels we can nest our collections!
books[2]['authors'][1]='tom'
print(books)
print(books[2]['authors'])

fav_foods = {
    'sue': ['pizza', 'pasta', 'salad'],
    'jane': ['turkey', 'apples'],
    'tim': ['oranges', 'apples', 'salad', 'kiwi'],
    'tony': ['bacon', 'spam']
}

#Question!! what's the difference between this and the tuple of dic

#note how we have a list nested in a dictionary
#Let's loop over the dictionary in sorted order
for name, foods in sorted(fav_foods.items()):
    print('\n' + name.title() + '\'s favorite foods are:')
    #now loop over the foods...values in the dictionary object
    for food in foods:
        print('\t' + food.title())

#While Loops allow us to iterate until some condition is no longer True
#In other words, we loop until some expression evaluates to False
val = 1
while val <= 10:
    print('The current loop is: ' + str(val))
    val = val+1
    #or
    #val += 1

#With this structure, we have to be careful to not enter into an infinite loop
#where we will never exit the loop!
val = 1
while val <= 10:
    print('The current loop is: ' + str(val))

#Conversely, we have to be careful that we actually enter the loop
val = 11
while val <= 10:
    print('The current loop is: ' + str(val))

#In this example, let's loop until the user enters the word quit!
prompt = 'Enter a name to print out on the screen.'
prompt += '\nEnter the word quit to end the program.'
name = ''
while name.lower() != 'quit':
    name = input(prompt)
    print(name)

#We can use a boolean True or False flag to enter or exit the while loop
prompt = 'Enter a name to print out on the screen.'
prompt += '\nEnter the word quit to end the program.'
name = ''
flag = True
#while flag == True:
while flag:
    name = input(prompt)
    if name.lower() == 'quit':
        flag = False
        print(name)
    else:
        print(name)

#We can use the break keyword to exit the while loop also
while True:
    name = input(prompt)
    if name.lower() == 'done':
        break
    else:
        print(name)

#If we want to control the flow of execution in our while loop,
# we can use the continue keyword to go to the next iteration (or to the top of the loop)
curval = 0
while curval <= 4:
    curval += 1
    if curval % 2 == 0:
        print('In the continue section with the value of ' + str(curval))
        continue
    print('Outside of the continue if block with a value of ' + str(curval))

#Activities - Time for you to practice on your own!
#Build a dictionary of odd cubes between 1 and 25 (inclusive)
#Print the result and the length of the dictionary object once finished.
#The output should look as follows:
#{1: 1, 3: 27, 5: 125, 7: 343, 9: 729, 11: 1331, 13: 2197, 15: 3375, 17: 4913, 19: 6859, 21: 9261, 23: 12167, 25: 15625}
odd = {1:1}
for i in range(3,26,2):
 odd[i] = pow(i,3)

print(odd)





#Create a dictionary object of five employees with the key being their name and the value as 0 for each employee
#Then, print all of the values in the dictionary.
#Then, loop over all five employees and change the value to a random integer between 100 and 500.
#Next, print out the dictionary to verify that your updates worked.
#Next, Create a list containing two of the employees in the dictionary.
#Then, Loop over the items in the dictionary.  If the item in the dictionary is also in the list of favorites,
#print an appropriate message.  Otherwise, print an appropriate message indicating that the employee is not
#on the favorites list.
#Lastly, check if Tom is in your dictionary of employees. If not, print an appropriate message.
import random
employ= {'a':0,'b':0,'c':0,'d':0,'e':0}
print(employ.values())

for e in employ.keys():
    employ[e] = random.randint(100,500)
print(employ)
fav = ['a','g']
for name in employ.keys():
    if name in fav:
        print("cong")
    else:
        print("sad")


#Calculate the mean, median, mode, and standard deviation of the values in this list

import statistics as s
my_list = [101,101,101,115,112,113,119,222,222,127,128,128,182,230]
print("The mean is " + str(s.mean(my_list)))
print("The median is " + str(s.median(my_list)))




#For this question, use the following tuple:
tm_tuple = ('tom', 'tim', 0.9, 1, ['joe', 14])
#First, append 'sally' and the number 99 to the list object in this tuple.
#Second, delete the 14 from the list object.
#Third, print the tuple.
#Fourth, unpack each element in the tuple into separate variables in a single line.


#Using the below object, check if it is a list or a tuple.
cols = (99, 107, 'sally', 207.43, 100)
#cols = [99, 107, 'sally', 207.43, 100]
#If it is a tuple, print an appropriate message.
#If it is a list, print an appropriate message.
#If it is anything else, print an appropriate message.
#NOTE: Your code should get a different result depending on which
#cols = line of code is commented above.
#Next, loop over all of the elements in the cols object.
#Determine if the object type is a float or an int.
#If so, print an appropriate message.
#If not, don't do anything



#For this question, let's use the following dictionary object
staff = {
    'srist': {'first_name': 'tony', 'last_name': 'matheson', 'office': 'manhattan'},
    'hrsnt': {'first_name': 'Janet', 'last_name': 'matlin', 'office': 'boston'},
    'sresh': {'first_name': 'Leroy', 'last_name': 'hwang', 'office': 'dc'},
    'khads': {'first_name': 'kadina', 'last_name': 'xie', 'office': 'boston'},
}
#Loop over the items in this list and print a message similar to the following:
#Employee ID: sresh
#	Full name: Leroy Hwang
#	Home office: DC

#Employee ID: khads
#	Full name: Kadina Xie
#	Home office: BOSTON
#	Warning this office might close!

#If the employee is assigned to the Boston office, print a warning message that this office might close.




#Create a tuple containing two lists.
#Each list should contain three random integers between 1 and 1000.
#Print the tuple and each number in the tuple separately (i.e., on seperate lines).


#Create a tuple with even numbers from 100 to 1000 (inclusive)
#Print the tuple and its length!



#Create a tuple with every 5's from 100 to 1000 (inclusive)
#100, 105, 110, and so on.
#Print the tuple and its length!



#Create a list of odd numbers starting with 999
# and counting down to 101.
#Print the list.
#Then, create a new list (from the previous list)
# with the numbers that are either between 100 and 199 or between 800 and 899.
#Then, print the new list of numbers.
#Finally, delete the original list from memory.



# Ask the user to enter a word.
#Then loop over each letter in the word.
#For each letter, print the index and the actual letter
#For this question, please use a while loop.




#How many times does the letter 'a' appear in a word that the user enters?
#Then, print a user friendly message indicating how many times that the letter 'a' appears
#in that word.



#Modify the previous example, to now count the number of times that any vowel appears
#Modify the print statement to print a meaningful message



#Does the word that the user entered come before the word tom?
#Based on the condition, print an appropriate message.



#Ask the user to enter a word.
#Then, convert that word to upper case letters and center the word around * with 25 places.
#If the user, enters the word bill, then print:
#***********BILL**********



#Modify the previous program.  If the word that the user enters ends in a vowel,
#change the fill from an * to a -.  Otherwise, keep the *.



#Ask the user to enter a word.
#Then ask the user to enter a search string.
#If the search string is contained in the word, print a message containing the original word
# and the search string.  Then, replace the search string with an *.  For instance,
#if the user enters the word apple and a search string of p, print out (on two lines):
#Your original word was 'apple', which contained the string 'p'
# so we changed the word to 'a**le'
#if the search string is not found, print out a message indicating that the search
# string was not found in the word



#Extract the host from this email header.
#Your code should work regardless of what email address is entered
raw_data = 'From tmattson8@protonmail.com Sun Jan 14 09:22:18 2020'


#Given the following list, iterate it, count the occurrence of each element
# and create a dictionary to show the count of each element.  Finally,
#print the items in the dictionary.
sampleList = [11, 45, 8, 11, 23, 45, 23, 45, 89]
print('Original list ', sampleList)


## Use the following dictionary.
dnsservers = {'us': 'ns1.cyberciti.com', 'uk': 'ns2.cyberciti.biz', 'asia': 'ns3.cyberciti.org'}
#Ask the user to input a geo-location.
#If the location is found, print the name of the server.
#If the location is not found, print a message indicating that the geo-location is not found.




#Modify the previous program to loop until the user enters a valid geo-location.




#Use the time module and a while looping structure to loop an infinite number of times.
#During each iteration through the loop, increment a counter variable by one,
# print a message that informs the user what loop number they are on and
# that they have to click the stop button to exit the loop.
#Then, sleep by two seconds before the next loop starts.



#Modify the previous program to exit after the fifth loop (before the sixth loop has started)
#Please do this with a while loop.
