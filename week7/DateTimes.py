#Dealing with dates can be frustrating because
#dates come in all different formats, which can make it difficult to add
# or subtract dates or to compare values against specific dates.

#In a banking context,
#selecting all the accounts that have past due loan payments
#
#In a HR context, find all employees hired before or after a specific date
#to help calculate benefit eligibility.
#
#In a marketing context, find all customers who made multiple purchases in a given month
#
#In an accounting context, identify all journal entries that were entered in the last week of a quarter
#and the first week of a quarter.


#Once we have a date in a datetime object, we can manipulate the dates quite easily but
#converting them to a datetime object can be challenging.

#Let's first work with the datetime module/library
import datetime

#today fetches the current date
print(datetime.date.today())

#Create a date object with a specific year, month, day
dt = datetime.date(2020, 10, 20)
print(type(dt))
print(dt)
#now we can extract the components of the date object
print(dt.day)
print(dt.month)
print(dt.year)

#We can also convert dates to strings using the strftime function
#This function takes the format as an argument.
dt = datetime.date.today()
sdt = dt.strftime('%d-%m-%Y')
print(type(sdt))
print('The current date is ' + sdt)

sdt = dt.strftime('%a, %d-%m-%y')
print(sdt)
sdt = dt.strftime('%A, %B %d, %Y')
print(sdt)

#Common datetime formatting options are as follows:
#%d refers to day of the month. In 20-10-2019, %d returns 20.
#%m refers to month of the year. In 20-10-2019, %m returns 10.
#%Y refers to year. The letter 'Y' is in upper case. In 20-10-2019, %Y returns 2019.
#%y refers to year in two-digit format. In 20-10-2019, %y returns 19.
#%a returns the first three letter of the weekday Sun
#%A returns the complete name of the weekday Sunday
#%b returns the first three letters of the month Oct
#%B returns the complete name of the month October
#Some time flags are as follows:
#%I converts 24 hour time format to 12 hour format.
#%p returns AM PM based on time values.
#%H returns hours of the time value.
#%M returns minutes of the time value.
#%S returns seconds of the time value.

#We can also work with time objects using the datetime class
t = datetime.time(21, 2, 3)
print(t)
print(type(t))
print(t.hour)
print(t.minute)
print(t.second)
#Add in microseconds to our datetime.time object
t = datetime.time(21, 2, 3, 4352)
print(t.microsecond)

#Now that we have a time object, we can format our time however we want/need
#using the strftime function
print(t.strftime('%I:%M %p'))
print(t.strftime('%I:%M:%S %p'))

#Now let's work with both dates and times together by extracting the current date & time using the
#now() function and just a datetime function
dt = datetime.datetime(2020, 3, 20, 11, 33, 14)
print(type(dt))
print(dt)
dt = datetime.datetime.now()
print(type(dt))
print(dt)
#Depending on where you live in the world, the format of the above datetime might be in a different
#format.

#The International Organization for Standardization (ISO) date
# format is a standard way to express a numeric calendar date that eliminates ambiguity.
#Note that the isoformat() function results in a string and not a datetime object.
my_date = dt.isoformat()
print(type(my_date))
print(my_date)

#Let's say we have a string that we want to convert into a datetime object
#This may happen when we read in data from a text file or extract data containing dates from a database
#To simulate this, let's first grab the current date and time and insert it into a string object
#Something like this might be read in from a text file
dt = datetime.datetime.now().isoformat()
print(type(dt))
#Now, let's convert this string object to a datetime object so we can eventually do some
#date manipulations with the date object, which we cannot do with a string (or text) object.

#The first useful function here is the strptime function
parsed_date = datetime.datetime.strptime(my_date, '%Y-%m-%dT%H:%M:%S.%f')
print(type(parsed_date))
print(parsed_date)
#Now we can print out the date in whatever format we want using the strftime function
print(parsed_date.strftime('%A %B %d %X'))
print(parsed_date.strftime('%A %B %d %H:%M'))
print(parsed_date.strftime('%A %B %d %H:%M:%S'))

#Here are a few other examples with different strings containing dates that need to be converted
#to datetime objects.
date_time_str1 = 'Jun 28 2018  7:40AM'
date_time_str2 = 'September 18, 2017, 22:19:55'
date_time_str3 = 'Sun,05/12/99,12:30PM'
date_time_str4 = '2018-03-12T10:12:45Z'
#The strptime function requires the second argument to match the format of the string
#Often this involves simply looking at the raw data file to determine what format
#our dates were given to us.
date_time_obj1 = datetime.strptime(date_time_str1, '%b %d %Y %I:%M%p')
date_time_obj2 = datetime.strptime(date_time_str2, '%B %d, %Y, %H:%M:%S')
date_time_obj3 = datetime.strptime(date_time_str3, '%a,%d/%m/%y,%I:%M%p')
date_time_obj4 = datetime.strptime(date_time_str4, '%Y-%m-%dT%H:%M:%SZ')

#Now we can extract the dates and times from our datetime objects
print('For Time1  Date:', date_time_obj1.date(), 'Time:', date_time_obj1.time())
print('For Time2  Date:', date_time_obj2.date(), 'Time:', date_time_obj2.time())
print('For Time3  Date:', date_time_obj3.date(), 'Time:', date_time_obj3.time())
print('For Time4  Date:', date_time_obj4.date(), 'Time:', date_time_obj4.time())

#The datetime library gives us a basic way of adding and subtracting dates
#30 days ahead
dt = datetime.datetime.now()
delta = datetime.timedelta(days=30)
print(dt + delta)

#30 days back
print(dt - delta)

#Ahead by 10 days and fraction thereof
delta = datetime.timedelta(days=10, hours=3, minutes=30, seconds=30)
print(dt + delta)
#Advance by weeks and fractions thereof
delta = datetime.timedelta(weeks= 4, hours=3, minutes=30, seconds=30)
print(dt + delta)

#However, a more efficient way to do these types of date operations is by using the
#python-dateutil module.  We will have to install this on your machine if it is not already installed
#python -m pip install python-dateutil

#I will use a few functions from these two built-in modules in the remainder of the demonstration.
import datetime
import calendar

#importing the dateutil module in this manner will not work based on how
#this module is coded..
import dateutil
now = datetime.datetime.now()
print(now)
today = datetime.date.today()
print(today)
#What is now plus one month?
ndt = now+dateutil.relativedelta(months=+1)
print(ndt)

from dateutil.relativedelta import *
#The relativedelta library provides us with an easy way to add different date units
# (days, weeks, months, years & so on)
#grab the date and time and place them in two variables
now = datetime.datetime.now()
print(now)
from dateutil.tz import *
now = datetime.datetime.now(dateutil.tz.gettz('America/Los_Angeles'))
print(now)
sysdtname = datetime.datetime.now(dateutil.tz.tzlocal()).tzname()
print(sysdtname)

#Let's revert to my current system date/time
now = datetime.datetime.now()
print(now)

#What is now plus one month
ndt = now+relativedelta(months=+1)
print(ndt)

#What is now minus one month
ndt = now+relativedelta(months=-1)
print(ndt)

#What is now plus one month and one week?
ndt = now+relativedelta(months=+1, weeks=+1)
print(ndt)

#What is next month plus one week at 10:00am
#Note the singular hour argument...replaces the current hour...does not perform any addition or subtraction
ndt = now+relativedelta(months=+1, weeks=+1, hour=10)
print(ndt)

#In the next example, we are adding 10 hours
ndt = now+relativedelta(months=+1, weeks=+1, hours=+10)
print(ndt)

#Here is another example using an absolute relativedelta.
# Notice the use of year and month (both singular),
# which causes the values to be replaced in the original
# datetime rather than performing any arithmetic operation on them.

#Plural adds or subtracts while singular replaces
print(now)
ndt = now+relativedelta(year=1, month=4)
print(ndt)
#notice how the year changes from 2020 to 0001 because of the replacing operation

ndt = now+relativedelta(year=2018, month=4, day=14)
print(ndt)

#One month before one year....This example should add 11 months to the now datetime object
ndt = now + relativedelta(years=+1, months=-1)
print(ndt)

#How does relativedelta handle months with different number of days?
# Notice that adding one month will never cross the month boundary.
#February is a great example because it has a different number of days!
dt1 = datetime.date(2020,2,27)+relativedelta(months=+1)
print(dt1)
dt1 = datetime.date(2020,1,31)+relativedelta(months=+1)
print(dt1)
dt1 = datetime.date(2020,3,31)+relativedelta(months=+1)
print(dt1)

#The logic for adding years is the same, even on leap years.
dt1 = datetime.date(2020,2,28)+relativedelta(years=+1)
print(dt1)
dt1 = datetime.date(2020,2,29)+relativedelta(years=+1)
print(dt1)
#Subtract two years from 2/29/2020
#It will be smart enough to recognize that 2/29/2018 is not actual date
dt1 = datetime.date(2020,2,29)+relativedelta(years=-2)
print(dt1)
#Subtract one year from 3/1/2021 and from 2/28/2021
dt1 = datetime.date(2021,3,1)+relativedelta(years=-1)
print(dt1)
dt1 = datetime.date(2021,2,28)+relativedelta(years=-1)
print(dt1)

#Lets use the date only without the time for the next few examples
today = datetime.date.today()
print(today)
#What is the date for next friday?
ndt = today+relativedelta(weekday=calendar.FRIDAY)
print(ndt)

#What is the date for next Tuesday
ndt = today+relativedelta(weekday=calendar.TUESDAY)
print(ndt)
#If you don't want to use the calendar object, you can use the abbreviations
#What is the date for next Thursday
ndt = today+relativedelta(weekday=TH)
print(ndt)

#What is the last Thursday in this month?
#substitute the current day with 31 and find the previous thursday
ndt = today+relativedelta(day=31, weekday=TH(-1))
print(ndt)
#What is the second to last Friday in the month
ndt = today+relativedelta(day=31, weekday=FR(-2))
print(ndt)

#What is date for next wednesday
ndt = today+relativedelta(weekday=WE(+1))
print(ndt)

#What is the date for not this coming Wednesday but the Wednesday after that?
ndt = today+relativedelta(weekday=WE(+2))
print(ndt)

#Find me next Sunday?
ndt = today+relativedelta(weekday=SU(+1))
print(ndt)
#so long as it is not today!!!
ndt = today+relativedelta(days=+1, weekday=SU(+1))
print(ndt)

#Find the first day of the 5th week of 2020
#This will give me the sunday of the sixth week
ndt =  datetime.datetime(2020,1,1)+relativedelta(weekday=SU, weeks=+5)
print(ndt)
#If we want the start of the 5th week, subtract one from weekday
ndt =  datetime.datetime(2020,1,1)+relativedelta(weekday=SU(-1), weeks=+5)
print(ndt)

#Now let's work with rrule, which are recurrence rules
#I will also work with the parser to parse portions of dates in the following examples
from dateutil.rrule import *
from dateutil.parser import *

#Recall from above, how we converted strings to dates....we had to specify the format for it to work properly
date_time_str1 = 'Jun 28 2018  7:40AM'
date_time_str2 = 'September 23, 2019, 22:19:55'
date_time_str3 = 'Sun,06/14/19 12:30PM'
date_time_str4 = '2020-03-12T10:12:45Z'
#The strptime function requires the second argument to match the format of the string
date_time_obj1 = datetime.strptime(date_time_str1, '%b %d %Y %I:%M%p')
date_time_obj2 = datetime.strptime(date_time_str2, '%B %d, %Y, %H:%M:%S')
date_time_obj3 = datetime.strptime(date_time_str3, '%a,%d/%m/%y,%I:%M%p')
date_time_obj4 = datetime.strptime(date_time_str4, '%Y-%m-%dT%H:%M:%SZ')
#Now lets work with dateutil parser, which should be easier!
#NOTE how the parse function takes a string and converts it to a date object
tmstr = parse(date_time_str1)
print(tmstr)
tmstr = parse(date_time_str2)
print(tmstr)
tmstr = parse(date_time_str3)
print(tmstr)
#It is not perfect...some strings are not recognized
#Notice the error here!
date_time_str3 = 'Sun,05/12/99,12:30PM'
tmstr = parse(date_time_str3)
#If we have a non-standard format that the dateutil.parser can't handle, we will probably just revert to
#using the strptime function where we feed in the specific format or possibly a find and replace...
tmstr = parse(date_time_str3.replace(',', ' '))
print(tmstr)

tmstr = parse(date_time_str4)
print(tmstr)

sdt = 'Today is 29 of October of 2020, exactly ' \
      'at 11:20:11 with timezone -04:00.'
tmstr = parse(sdt, fuzzy=True)
print(tmstr)

#We have to be careful with ambiguous dates
#Where is the year in this string
tmstr = parse('12-08-11', yearfirst=True)
print(tmstr)

tmstr = parse('2020.08.11 AD at 15:08:56 PDT', ignoretz=True)
print(tmstr)

#We can use rrule to create lists with different date units
#Create a list object of datetime objects starting with 1/1/2020 until 5/1/2020
tm_list = list(rrule(DAILY, dtstart = parse('20200101T090000'), until = parse('20200501T000000')))
print(tm_list)
print(len(tm_list))

#Now, create a list object of weeks between 1/1/2020 until 5/1/2020
tm_list = list(rrule(WEEKLY, dtstart = parse('20200101T090000'),until = parse('20200501T000000')))
print(tm_list)
print(len(tm_list))

#Change the unit from weekly to monthly!
tm_list = list(rrule(MONTHLY, dtstart = parse('20200101T090000'),until = parse('20200501T000000')))
print(tm_list)
print(len(tm_list))

#Create a list object containing the next 10 days from 2/12/2020
tm_list = list(rrule(DAILY, count=10, dtstart=parse('20200212T090000')))
print(tm_list)
print(len(tm_list))

#Create a list object containing the next 3 months from 2/12/2020
tm_list = list(rrule(MONTHLY, count=3, dtstart=parse('20200212T090000')))
print(tm_list)
print(len(tm_list))

#Create a list object containing every other day starting on 2/12/2020
# but only create five instances
tm_list = list(rrule(DAILY, interval=2, count=5, dtstart=parse('20200212T090000')))
print(tm_list)
print(len(tm_list))

#Create a list object containing every third month starting on 1/1/2020 (with four instances)
tm_list = list(rrule(MONTHLY, interval=3, count=4, dtstart=parse('20200101T090000')))
print(tm_list)
print(len(tm_list))

#Create a list containing the days in months 1,3,5,and 7 in years 2020-2022
tm_list = list(rrule(DAILY, bymonth=(1,3,5,7), dtstart=parse('20200101T090000'),
                     until=parse('20221231T090000')))
print(tm_list)
print(len(tm_list))

#Create a list of every other week starting on 1/1/2020 (first ten such instances)
tm_list = list(rrule(WEEKLY, count=10, dtstart=parse('20200101T090000')))
print(tm_list)
print(len(tm_list))

#Create a list of every other week, 6 occurrences, starting on 1/1/2020.
tm_list = list(rrule(WEEKLY, interval=2, count=6, dtstart=parse('20200101T090000')))
print(tm_list)
print(len(tm_list))

#Create a list of dates on Tuesday and Thursday for the first 5 weeks of 2020.
tm_list = list(rrule(WEEKLY, count=10, wkst=SU, byweekday=(TU,TH), dtstart=parse('20200101T090000')))
print(tm_list)
print(len(tm_list))

#Create a list of dates on Wednesday and Thursday for every other week for the first 8 weeks in 2020.
tm_list = list(rrule(WEEKLY, interval=2, count=8, wkst=SU, byweekday=(WE,FR), dtstart=parse('20200101T090000')))
print(tm_list)
print(len(tm_list))

#Create a list of dates for the 1st Friday of each month in 2020 for 3 occurrences.
tm_list = list(rrule(MONTHLY, count=3, byweekday=FR(1), dtstart=parse('20200101T090000')))
print(tm_list)
print(len(tm_list))

#Every other month on the 1st and last Sunday of the month for 10 occurrences.
tm_list = list(rrule(MONTHLY, interval=2, count=10,
             byweekday=(SU(1), SU(-1)),
             dtstart=parse('20200101T090000')))
print(tm_list)
print(len(tm_list))

#rruleset is beneath rrule so no new import is needed.
#These allow us to specify specific dates to exclude from our set.

#Create a list using a rruleset
#for 14 days (each day) starting on 1/1/2020,
# but skip Saturday and Sunday occurrences.
set = rruleset()
set.rrule(rrule(DAILY, count=14, dtstart=parse('20200101T090000')))
set.exrule(rrule(YEARLY, byweekday=(SA,SU), dtstart=parse('20200101T090000')))
tm_list = list(set)
print(tm_list)

#Create a list using a rruleset for 4 weeks starting on 1/1/2020.
# However, make sure 1/13/2020 is included and exclude 1/15/2020.
set = rruleset()
set.rrule(rrule(WEEKLY, count=4, dtstart=parse('20200101T090000')))
set.rdate(datetime.datetime(2020, 1, 13, 9, 0))
set.exdate(datetime.datetime(2020, 1, 15, 9, 0))
tm_list = list(set)
print(tm_list)


#Another way to create a list using recurrence rules is the rrulestr() function
#We will pass the rrule and the start date, which will return an rrule object.
#We can then cast this object as a list!

#Create a list of every 10 days (5 instances only) starting on 1/1/2020 using a rrulestr.
tm_rstr = rrulestr('FREQ=DAILY;INTERVAL=10;COUNT=5',
               dtstart=parse('20200101T090000'))
print(type(tm_rstr))
tm_list = list(tm_rstr)
print(tm_list)