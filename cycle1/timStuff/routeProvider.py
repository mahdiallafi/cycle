"""
This script reads:  ratingDict created by k_means.py (ratings for each centroid)
                    routes.csv created by distances.py
                    places.csv

It can train the Q-Matrices and saves them under pythonData/modelCentroid{centroid}
It provides the methods:    getBestGreedy(place, distance, time, userData) which provides the best greedy result for the given user
                            getQResult(TODO)
                            place = in Name not index
                            distance and time in meters and seconds
                            userData as a list
"""

import pandas as pd
import numpy as np
import csv
import pickle
import requests
from placeRater import ratePlace
from k_means import getCentroidOrder


### Basic setup of global data for functions to use
### Loading in routes and places and preparing them for use
with open("pythonData/ratingDict", 'rb') as file: 
    ratingDict = pickle.load(file)
ratingColumns = ["age1","age2","age3","age4","age5","age6","age7","age8","male","non-binary","female","history","art","nature","sports","sciences","sights","fun_activities"]
places=pd.read_csv('places.csv', sep = ';', names=["google_id", "name", "description", "googleMapsURL", "address"] + ratingColumns)
placeToIdx = {}
idxToPlace = {}
with open("places.csv", "r", newline="") as file:
    reader = csv.reader(file, delimiter = ';')
    c = 0
    for row in reader:
        key = row[1]
        placeToIdx[key] = c
        idxToPlace[c] = key
        c += 1
routes=pd.read_csv('routes.csv', names=["origin", "dest", "dist", "time"])
routes = routes[1:] # top row of routes is rownames
routes[["dist", "time"]] = routes[["dist", "time"]].astype(int)
routesIndexed = routes.copy()
routesIndexed['origin'] = routes['origin'].map(placeToIdx)
routesIndexed['dest'] = routes['dest'].map(placeToIdx)


# Greedy Implementation
def maxScore(option, ratingVector):
    return ratingVector[option.dest]

def maxScoreTime(option, ratingVector):
    return ratingVector[option.dest] / (option.time / 1000)

def maxScoreDistance(option, ratingVector):
    return ratingVector[option.dest] / (option.dist / 1000)
    
def getDecision(options, valueFunction, ratingVector):
    maxValue = -1
    for opt in options.itertuples():
        if valueFunction(opt, ratingVector) > maxValue:
            maxValue = valueFunction(opt, ratingVector)
            decisionIdx = opt.Index
    return decisionIdx

def findGreedy(curPlace, distanceLeft, timeLeft, goal, visited, valueFunction, ratingVector):
    if curPlace == goal:
        return [curPlace]

    options = routes[routes["origin"] == curPlace]
    options = options[~options["dest"].isin(visited)]
    options = options.reset_index(drop=True)
    
    while len(options) > 0:
        decisionIdx = getDecision(options, valueFunction, ratingVector)
        decision = options.iloc[decisionIdx]
        if decision["dist"] < distanceLeft and decision["time"] < timeLeft:
            newRoute = visited.copy()
            newRoute.append(decision["dest"])
            res = findGreedy(decision["dest"], distanceLeft - decision["dist"], timeLeft - decision["time"], goal, newRoute, valueFunction, ratingVector)
            if res != None:
                return [curPlace] + res
        options = options.drop(decisionIdx)
        options = options.reset_index(drop=True)
    
    if goal is None:
        return [curPlace]
    return None

def findGreedyHead(curPlace, distanceLeft, timeLeft, valueFunction, ratingVector, destination):
    options = routes[routes["origin"] == curPlace]
    options = options.reset_index(drop=True)

    while len(options) > 0:
        decisionIdx = getDecision(options, valueFunction, ratingVector)
        decision = options.iloc[decisionIdx]
        if decision["dist"] < distanceLeft and decision["time"] < timeLeft:
            res = findGreedy(decision["dest"], distanceLeft - decision["dist"], timeLeft - decision["time"], destination, [decision["dest"]], valueFunction, ratingVector)
            if res != None:
                return [curPlace] + res
        options = options.drop(decisionIdx)
        options = options.reset_index(drop=True)
    return [curPlace]

# This is the function which will be called from outside this script
def getBest(place, distance, time, userData, destination):
    # Build ratingVector for user (it is acc a dict name of place -> rating)
    ratingVector = {}
    for sight in places.iterrows():
        ratingVector[sight[1]["name"]] = ratePlace(sight[1], userData)
    
    if place is None:
        rows = places.sample(n=3)
        placeList = rows.values[:, 1]
    else:
        placeList = [place]

    for place in placeList:
        # Greedy results
        results = [[None, findGreedyHead(place, distance, time, maxScore, ratingVector, destination)], [None, findGreedyHead(place, distance, time, maxScoreDistance, ratingVector, destination)]]
        
        # Now we get the Q-learning results
        results.append([None, modelGetSteps(place, distance, time, userData, destination)])
    
    for idx, result in enumerate(results):
        sum = 0
        for visitedPlace in result[1]:
            sum += ratingVector[visitedPlace]
        results[idx][0] = sum

    bestRes = sorted(results, reverse=True)[0][1]
    returnList = []
    for element in bestRes:
        returnList.append(places.iloc[placeToIdx[element]])
    return pd.DataFrame(returnList)
        
### GREEDY IMPLEMENTATION OVER

### Q-learning
from collections import namedtuple
State = namedtuple("State", ["place", "distance", "time"])
gamma = 1
alpha = 0.8
distanceStep = 100 # in m
timeStep = 120 # in seconds
distanceIntervalls = 100
timeIntervalls = 80

MAXDISTANCE = distanceStep * distanceIntervalls
MAXTIME = timeStep * timeIntervalls

def available_actions(state):
    realisticOptions = []
    for opt in routes.loc[(routesIndexed['origin'] == state.place)].iterrows():
        newOpt = State(placeToIdx[opt[1]["dest"]], (int) (state.distance * distanceStep - opt[1]["dist"]) // distanceStep, (int) (state.time * timeStep - opt[1]["time"]) // timeStep)
        if newOpt.distance >= 0 and newOpt.time >= 0:
            realisticOptions.append(newOpt)
    return realisticOptions

def update(current_state, action, gamma, alpha, placeRatingsIndexed):
    max_value = np.max(Q[action.place, :, action.distance , action.time])
    Q[current_state.place, action.place, current_state.distance, current_state.time] = (1-alpha)*Q[current_state.place, action.place, current_state.distance, current_state.time] + alpha * (placeRatingsIndexed[action.place] + gamma * max_value)
    return 

def scoreQ():
    return (np.sum(Q / np.max(Q) * 100))

# Function that will be called from outside
def modelGetSteps(place, distance, time, userData, destination):
    if place is None:
        row = places.sample(n=1)
        place = row.values[0][1]

    closestCentroid = getCentroidOrder(userData)[0][1]

    with open("pythonData/modelCentroid{}".format(closestCentroid), 'rb') as file: 
        Q = pickle.load(file)

    current_state = State(placeToIdx[place], distance // distanceStep - 1, time // timeStep - 1)
    steps = [current_state]
    visited = []
    
    ### TODO ACTUALLY WE DON'T HAVE A FIXED DESTINATION SO TIME TO FIX
    while True:
        next_step_list = np.where(Q[current_state.place, :, current_state.distance, current_state.time] == np.max(Q[current_state.place, :, current_state.distance, current_state.time]))[0]
        if len(next_step_list) == Q.shape[0]:
            # RAN OUT OF RESSOURCES
            if (destination is None or steps[-1].place == placeToIdx[destination]):
                break
            elif len(steps) == 1:
                break
            Q[steps[-2].place, steps[-1].place, steps[-2].distance, steps[-2].time] = 0
            current_state = steps[-2]
            steps = steps[:-1]
            visited = visited[:-1]
            continue
        elif next_step_list[0] in visited or current_state.distance * distanceStep < routes.loc[(routesIndexed['origin'] == current_state.place) & (routesIndexed['dest'] == next_step_list[0]), "dist"].values[0]:
            # Top pick was already visited or is unaffordable
            # We will set score to zero and redo
            Q[current_state.place, next_step_list[0], current_state.distance, current_state.time] = 0
            continue
     
        next_step = State(next_step_list[0], (int) (current_state.distance * distanceStep - routes.loc[(routesIndexed['origin'] == current_state.place) & (routesIndexed['dest'] == next_step_list[0]), "dist"].values[0]) // distanceStep, (int) (current_state.time * timeStep - routes.loc[(routesIndexed['origin'] == current_state.place) & (routesIndexed['dest'] == next_step_list[0]), 'time'].values[0]) // timeStep)
        steps.append(next_step)
        visited.append(next_step.place)
        
        current_state = next_step


    returnList = []
    for step in steps:
        returnList.append(places.iloc[step.place])
    return pd.DataFrame(returnList)



# Training
"""
print("Will be trained to deal with maximum start distance of:", MAXDISTANCE, "m")
print("Will be trained to deal with maximum start time of:", MAXTIME, "s")

Q = np.zeros([len(places), len(places), distanceIntervalls, timeIntervalls])
print("Shape of our matrix:", Q.shape)

for idx, ratings in ratingDict.items():
    placeRatingsIndexed = {}

    #Q = np.zeros([len(places), len(places), distanceIntervalls, timeIntervalls])
    with open("pythonData/modelCentroid{}".format(idx), 'rb') as file: 
        Q = pickle.load(file)

    for i, rating in enumerate(ratings):
        placeRatingsIndexed[i] = rating

    for i in range (Q.shape[0]):
        for j in range (Q.shape[2]):            
            for k in range (Q.shape[3]):
                current_state = State(i, j, k)
                available_act = available_actions(current_state)
                for action in available_act:
                    update(current_state, action, gamma, alpha, placeRatingsIndexed)

    with open("pythonData/modelCentroid{}".format(idx), 'wb') as file: 
        pickle.dump(Q, file)

    print("Trained and saved: pythonData/QmodelCentroid{}".format(idx))
"""