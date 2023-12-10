#from routeProvider import findGreedyHead
import pickle
from routeProvider import getBest

with open("pythonData/modelCentroid{}".format(50), 'rb') as file: 
        Q = pickle.load(file)

#print(Q[:5, :5, 60 , :])


userData = [1,1,1,1,1,1,1,1,1,1] # How to get the form data into this encoding we will do later.
print(getBest(None, 10000, 9600, userData, "St. Peter’s Church"))
print(getBest(None, 10000, 9600, userData, None))
print(getBest(None, 10000, 9600, userData, None))
print(getBest("St. Michaelis Church", 10000, 9600, userData, None))
#print(modelGetSteps("St. Michaelis Church", 10000, 9600, userData, None))
# None as starting can lead to "empty" routes as place can be randomly selected that isn't part of the routematrix yet