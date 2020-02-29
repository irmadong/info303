#Open the employees.json file and print the following message for each employee in the json file:
#Tim Jones lives in Detroit, MI
#Sally Jonhson lives in Richmond, VA
#Tyrone Conners lives in Virginia Beach, VA
#The above message samples are not a complete list of all of the employees in the file



#Open the employees.json file and calculate the average and standard deviation of the age for all of the employees.
#When finished, print a user-friendly message such as the following:
#The average age of the employees is 40.00.
#The standard deviation of the age of the employees is 15.47.




#Using the data contained in the sportsfeed.xml file, create a list containing dictionary objects for each item
# with the title, description and link data elements.  The tag should be the key and the text should be the value.
#Build a function to create this list of dictionary objects.
#Then, build a function that will print out the specific key for each item. In this function, the key should
#be passed as an argument and so should the list containing the dictionary objects.
# If the user passes description as an argument, the following should be printed:

#Description: Andy Reid outcoached Kyle Shanahan, and Patrick Mahomes wasnt the best player on the field.
#Description: The QB saved the day for the Chiefs much like he had all postseason long.
#...
#Description: The second week of the LCS and LEC is about to conclude, and the top teams in each region brings some surprising additions.

#NOTE: Do not print off any extra spaces between the items



#Open a delimited text file (Employees.txt) and convert it to a JSON file.  The structure
#of your JSON file should be similar to the following:
#{
#    "employees": [
#        {
#            "Employee_ID": 10,
#            "First_Name": "Laverne",
#            "Last_Name": "Cotton",
#            "Email_Address": "lcotton@gmail.com",
#            "Country_Id": 101
#        },
#        {
#            "Employee_ID": 11,
#            "First_Name": "Colette",
#            "Last_Name": "Logsdon",
#            "Email_Address": "clogsdon@gmail.com",
#            "Country_Id": 101
#        },
#Name your json file testing.json.





#Create a well formed xml file using the following structures:
#Do not get frustrated with this example problem.  This is a very challenging problem.
#We have multiple nested data structures that we have to map to our XML structure.
#My suggestion is to do this in stages.  Start with one node.  Don't try to build the
#entire XML file in a single step.
tm_pos = [{'PONumber':'99503', 'OrderDate':'2020-10-22','Address_Shipping':{'Name':'Violet Adams',
                                                                    'Street':'123 Maple Street',
                                                                    'City':'Mill River', 'State':'CA',
                                                                    'Zip':'10999', 'Country':'USA'},
            'Address_Billing':{'Name':'Jane Yee', 'Street':'24 High Street',
                                                                    'City':'Winsted', 'State':'CT',
                                                                    'Zip':'06098', 'Country':'USA'},
            'DeliveryNotes': 'None'},
            {'PONumber':'99504', 'OrderDate':'2020-10-27','Address_Shipping':{'Name':'Raymond James Jr',
                                                                    'Street':'974 Main Street',
                                                                    'City':'Richmond', 'State':'VA',
                                                                    'Zip':'23173', 'Country':'USA'},
            'Address_Billing':{'Name':'Raymond James Sr', 'Street':'324 North Street',
                                                                    'City':'Honolulu', 'State':'HI',
                                                                    'Zip':'88808', 'Country':'USA'},
            'DeliveryNotes': 'Be careful when leaving packages'}]
tm_items = [{'99503': {'Items': [{'PartNumber': '872-AA', 'ProductName':'LawnMower', 'Quantity':1,
                             'USPrice': 150, 'ShipDate': '2020-10-24', 'Comment':'Electric'},
                               {'PartNumber': '926-AA', 'ProductName': 'Saw', 'Quantity': 1,
                                'USPrice': 225, 'ShipDate': '2020-10-23', 'Comment': 'Needs tuning'}
                               ]}},
            {'99504': {'Items': [{'PartNumber': '879-AA', 'ProductName':'Snow Shovel', 'Quantity':1,
                             'USPrice': 150, 'ShipDate': '2020-10-28', 'Comment':'Rush Delivery'},
                               {'PartNumber': '926-AA', 'ProductName': 'Saw', 'Quantity': 1,
                                'USPrice': 225, 'ShipDate': '2020-10-28', 'Comment': 'Store indoors for the winter'},
                              {'PartNumber': '872-AA', 'ProductName':'LawnMower', 'Quantity':1,
                                'USPrice': 150, 'ShipDate': '2020-10-29', 'Comment':'Electric'}
                               ]}}
            ]
#Your XML file should look something like the following:
#NOTE: The data contained in this sample file may not match the above structure!
#<PurchaseOrders>
#<PurchaseOrder PONumber=99503 OrderDate="2020-10-20">
#  <Address Type="Shipping">
#    <Name>Sally Jane Smith</Name>
#    <Street>123 Maple Street</Street>
#    <City>Mill Valley</City>
#    <State>CA</State>
#    <Zip>10999</Zip>
#    <Country>USA</Country>
#  </Address>
#  <Address Type="Billing">
#    <Name>Tai Yee</Name>
#    <Street>8 Oak Avenue</Street>
#    <City>Old Town</City>
#    <State>PA</State>
#    <Zip>95819</Zip>
#    <Country>USA</Country>
#  </Address>
#  <DeliveryNotes>Please leave packages in shed by driveway.</DeliveryNotes>
#  <Items>
#    <Item PartNumber="872-AA">
#      <ProductName>Lawnmower</ProductName>
#      <Quantity>1</Quantity>
#      <USPrice>148.95</USPrice>
#      <ShipDate>2020-10-29</ShipDate>
#      <Comment>Confirm this is electric</Comment>
#    </Item>
#    <Item PartNumber="926-AC">
#      <ProductName>Lawnmower</ProductName>
#      <Quantity>1</Quantity>
#      <USPrice>198.95</USPrice>
#      <ShipDate>2020-10-29</ShipDate>
#      <Comment>Gas powered lawn mower</Comment>
#    </Item>
#  </Items>
#</PurchaseOrder>
#<PurchaseOrder>
#Repeat the same structure for the next purchaseorder
#</PurchaseOrder>
#</PurchaseOrders>

