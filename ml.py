try:
    import matlab
except Exception as ex:
    print("itom is possibly compiled without Matlab support. This demo is not working")
    raise ex

session = matlab.MatlabSession() #a Matlab console is opened

session.setString("myString", "test") #creates the string variable 'myString' in the Matlab workspace with the value 'test'
print("myString:", session.getString("myString")) #returns 'test' as answer in itom
session.setValue("myArray", dataObject.randN([2,3],'int16')) #creates a 2x3 random matrix in Matlab (name: myArray)
arr = session.getValue("myArray") #returns the 2x3 array 'myArray' from Matlab as Numpy array
print(arr)

#read the current working directory of matlab
session.run("curDir = cd")
print(session.getString("curDir"))

#run directly executes the command (as string). This is the same than typing this command into the command line of Matlab.
#use this to also execute functions in Matlab. At first, send all required variables to the Matlab workspace, then execute a function
#that uses these variables.

del session #closes the session and deletes the instance

#session.close() only closes the session
