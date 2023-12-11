"""
This script reads:  ratingDict created by k_means.py (ratings for each centroid)
                    routes.csv created by distances.py
                    places.csv

It can train the Q-Matrices and saves them under cycle1/timStuff/pythonData/modelCentroid{centroid}
It provides the methods:    getBest(place, distance, userData, destination) which provides the best result for the given user (selected from greedy and Q-learning)
                            place = in Name not index
                            distance in meters
                            userData as a list
"""

import pandas as pd
import numpy as np
import csv
import pickle
from .placeRater import ratePlace
from .k_means import getCentroidOrder


### Basic setup of global data for functions to use
### Loading in routes and places and preparing them for use
with open("cycle1/timStuff/pythonData/ratingDict", 'rb') as file: 
    ratingDict = pickle.load(file)
ratingColumns = ["age1","age2","age3","age4","age5","age6","age7","age8","male","non-binary","female","history","art","nature","museums","churches","sights","fun_activities"]
places=pd.read_csv('cycle1/timStuff/places.csv', sep = ';', names=["google_id", "name", "description", "googleMapsURL", "address"] + ratingColumns)
placeToIdx = {}
idxToPlace = {}
with open("cycle1/timStuff/places.csv", "r", newline="") as file:
    reader = csv.reader(file, delimiter = ';')
    c = 0
    for row in reader:
        key = row[1]
        placeToIdx[key] = c
        idxToPlace[c] = key
        c += 1
routes=pd.read_csv('cycle1/timStuff/routes.csv', names=["origin", "dest", "dist", "time"])
routes = routes[1:] # top row of routes is rownames
routes[["dist", "time"]] = routes[["dist", "time"]].astype(int)
routesIndexed = routes.copy()
routesIndexed['origin'] = routes['origin'].map(placeToIdx)
routesIndexed['dest'] = routes['dest'].map(placeToIdx)


# Greedy Implementation
def maxScore(option, ratingVector):
    return ratingVector[option.dest]

def maxScoreDistance(option, ratingVector):
    return ratingVector[option.dest] / (option.dist / 1000)
    
def getDecision(options, valueFunction, ratingVector):
    maxValue = -1
    for opt in options.itertuples():
        if valueFunction(opt, ratingVector) > maxValue:
            maxValue = valueFunction(opt, ratingVector)
            decisionIdx = opt.Index
    return decisionIdx

def findGreedy(curPlace, distanceLeft, goal, visited, valueFunction, ratingVector):
    if curPlace == goal:
        return [curPlace]

    options = routes[routes["origin"] == curPlace]
    options = options[~options["dest"].isin(visited)]
    options = options.reset_index(drop=True)
    
    while len(options) > 0:
        decisionIdx = getDecision(options, valueFunction, ratingVector)
        decision = options.iloc[decisionIdx]
        if decision["dist"] < distanceLeft:
            newRoute = visited.copy()
            newRoute.append(decision["dest"])
            res = findGreedy(decision["dest"], distanceLeft - decision["dist"], goal, newRoute, valueFunction, ratingVector)
            if res != None:
                return [curPlace] + res
        options = options.drop(decisionIdx)
        options = options.reset_index(drop=True)
    
    if goal is None:
        return [curPlace]
    return None

def findGreedyHead(curPlace, distanceLeft, valueFunction, ratingVector, destination):
    options = routes[routes["origin"] == curPlace]
    options = options.reset_index(drop=True)

    while len(options) > 0:
        decisionIdx = getDecision(options, valueFunction, ratingVector)
        decision = options.iloc[decisionIdx]
        if decision["dist"] < distanceLeft:
            res = findGreedy(decision["dest"], distanceLeft - decision["dist"], destination, [decision["dest"]], valueFunction, ratingVector)
            if res != None:
                return [curPlace] + res
        options = options.drop(decisionIdx)
        options = options.reset_index(drop=True)
    return [curPlace]

# This is the function which will be called from outside this script
def getBest(place, distance, userData, destination):
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
        results = [[None, findGreedyHead(place, distance, maxScore, ratingVector, destination)], [None, findGreedyHead(place, distance, maxScoreDistance, ratingVector, destination)]]
        
        # Now we get the Q-learning results
        if distance < MAXDISTANCE:
            results.append([None, modelGetSteps(place, distance, userData, destination)])
    
    for idx, result in enumerate(results):
        sum = 0
        for visitedPlace in result[1]:
            sum += ratingVector[visitedPlace]
        results[idx][0] = sum

    bestRes = sorted(results, reverse=True)[0][1]
    returnList = []
    for element in bestRes:
        returnList.append(places.iloc[placeToIdx[element]])  
        pd.set_option('display.max_columns', None)   
    return pd.DataFrame(returnList)
        
### GREEDY IMPLEMENTATION OVER

### Q-learning
from collections import namedtuple
State = namedtuple("State", ["place", "distance", "time"])
gamma = 1
alpha = 0.8
distanceStep = 100 # in m
distanceIntervalls = 150
timeIntervalls = 1

MAXDISTANCE = distanceStep * distanceIntervalls

def available_actions(state):
    realisticOptions = []
    for opt in routes.loc[(routesIndexed['origin'] == state.place)].iterrows():
        newOpt = State(placeToIdx[opt[1]["dest"]], (int) (state.distance * distanceStep - opt[1]["dist"]) // distanceStep, 0)
        if newOpt.distance >= 0:
            realisticOptions.append(newOpt)
    return realisticOptions

def update(current_state, action, gamma, alpha, placeRatingsIndexed):
    max_value = np.max(Q[action.place, :, action.distance , action.time])
    Q[current_state.place, action.place, current_state.distance, current_state.time] = (1-alpha)*Q[current_state.place, action.place, current_state.distance, current_state.time] + alpha * (placeRatingsIndexed[action.place] + gamma * max_value)
    return 

def scoreQ():
    return (np.sum(Q / np.max(Q) * 100))

# Function that will be called from outside
def modelGetSteps(place, distance, userData, destination):
    if place is None:
        row = places.sample(n=1)
        place = row.values[0][1]

    closestCentroid = getCentroidOrder(userData)[0][1]

    with open("cycle1/timStuff/pythonData/modelCentroid{}".format(closestCentroid), 'rb') as file: 
        Q = pickle.load(file)

    current_state = State(placeToIdx[place], distance // distanceStep - 1, 0)
    steps = [current_state]
    visited = []

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
     
        next_step = State(next_step_list[0], (int) (current_state.distance * distanceStep - routes.loc[(routesIndexed['origin'] == current_state.place) & (routesIndexed['dest'] == next_step_list[0]), "dist"].values[0]) // distanceStep, 0)
        steps.append(next_step)
        visited.append(next_step.place)
        
        current_state = next_step

    return [idxToPlace[step.place] for step in steps]



# Training
"""
print("Will be trained to deal with maximum start distance of:", MAXDISTANCE, "m")

Q = np.zeros([len(places), len(places), distanceIntervalls, timeIntervalls])
print("Shape of our matrix:", Q.shape)

for idx, ratings in ratingDict.items():
    placeRatingsIndexed = {}

    #Q = np.zeros([len(places), len(places), distanceIntervalls, timeIntervalls])
    with open("cycle1/timStuff/pythonData/modelCentroid{}".format(idx), 'rb') as file: 
        Q = pickle.load(file)

    for i, rating in enumerate(ratings):
        placeRatingsIndexed[i] = rating

    for i in range (Q.shape[0]):
        for j in range (Q.shape[2]):            
            current_state = State(i, j, 0)
            available_act = available_actions(current_state)
            for action in available_act:
                update(current_state, action, gamma, alpha, placeRatingsIndexed)

    with open("cycle1/timStuff/pythonData/modelCentroid{}".format(idx), 'wb') as file: 
        pickle.dump(Q, file)

    print("Trained and saved: cycle1/timStuff/pythonData/QmodelCentroid{}".format(idx))
"""