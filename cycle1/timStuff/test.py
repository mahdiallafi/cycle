#from routeProvider import findGreedyHead
import pickle
"""
with open("pythonData/modelCentroid{}".format(50), 'rb') as file: 
        Q = pickle.load(file)

print(Q[10, 11, 99 , 79])
"""

with open("pythonData/users", 'rb') as file: 
    users = pickle.load(file) 

print(users.iloc[0])