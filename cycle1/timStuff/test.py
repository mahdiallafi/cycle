#from routeProvider import findGreedyHead
import pickle
from routeProvider import getBest


userData = [1,1,1,1,1,1,1,1,1,1] # How to get the form data into this encoding we will do later.
print(getBest(None, 10000, userData, "St. Peter’s Church"))
print(getBest(None, 10000, userData, None))
print(getBest(None, 10000, userData, None))
print(getBest("St. Michaelis Church", 10000, userData, None))