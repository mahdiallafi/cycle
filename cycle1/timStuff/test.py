#from routeProvider import findGreedyHead
import pickle
from routeProvider import getBestGreedy
"""
with open("pythonData/modelCentroid{}".format(50), 'rb') as file: 
        Q = pickle.load(file)

print(Q[10, 11, 99 , 79])
"""

userData = [1,1,1,1,1,1,1,1,1,1] # How to get the form data into this encoding we will do later.
print(getBestGreedy(None, 15000, 90000, userData, "St. Peter’s Church"))
print(getBestGreedy(None, 15000, 90000, userData, None))
print(getBestGreedy(None, 15000, 90000, userData, None))
print(getBestGreedy("Volksparkstadion", 15000, 90000, userData, "St. Peter’s Church"))
# None as starting can lead to "empty" routes as place can be randomly selected that isn't part of the routematrix yet