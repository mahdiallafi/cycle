import pandas as pd
import csv
import pickle
import requests
from userGenerationAndEval import getRatings


with open("pythonData/ratingDict", 'rb') as file: 
    ratingDict = pickle.load(file)

with open("pythonData/bestCentroidSet-size:50", 'rb') as file: 
    centroids = pickle.load(file)

ratingColumns = ["age1","age2","age3","age4","age5","age6","age7","age8","male","non-binary","female","history","art","nature","sports","sciences","sights","fun_activities"]
places=pd.read_csv('places.csv', sep = ';', names=["google_id", "name", "description", "googleMapsURL", "address"] + ratingColumns)
routes=pd.read_csv('routes.csv', names=["origin", "dest", "dist", "time"])
# top row of routes is rownames
routes = routes[1:]
routes[["dist", "time"]] = routes[["dist", "time"]].astype(int)

placeToIdx = {}
idxToplace = {}
with open("places.csv", "r", newline="") as file:
    reader = csv.reader(file, delimiter = ';')
    c = 0
    for row in reader:
        key = row[0]
        placeToIdx[key] = c
        idxToplace[c] = key
        c += 1

personStats =  [0.25,0,0.7,0.5,0.3,0.3,0.5,0.8,0.5]

placeRatings = {}
for i, rating in enumerate(getRatings(0, personStats, centroids, ratingDict)):
    placeRatings[places.iloc[i]["name"]] = rating

# Greedy Implementation
def maxScore(option):
    return placeRatings[option.dest]

def maxScoreTime(option):
    return placeRatings[option.dest] / (option.dist / 10000)

def maxScoreMoney(option):
    return placeRatings[option.dest] / (option.time / 100)
    
def getDecision(options, valueFunction):
    maxValue = 0
    for opt in options.itertuples():
        if valueFunction(opt) > maxValue:
            maxValue = valueFunction(opt)
            decisionIdx = opt.Index
    return decisionIdx

def findGreedy(curPlace, distanceLeft, timeLeft, goal, visited, valueFunction):
    if curPlace == goal:
        return placeRatings[curPlace]
    options = routes[routes["origin"] == curPlace]
    options = options[~options["dest"].isin(visited)]
    options = options.reset_index(drop=True)
    
    while len(options) > 0:
        decisionIdx = getDecision(options, valueFunction)
        decision = options.iloc[decisionIdx]
        if decision["dist"] < distanceLeft and decision["time"] < timeLeft:
            newRoute = visited.copy()
            newRoute.append(decision["dest"])
            res = findGreedy(decision["dest"], distanceLeft - decision["dist"], timeLeft - decision["time"], goal, newRoute, valueFunction)
            if res != None:
                return res + placeRatings[curPlace]
        options = options.drop(decisionIdx)
        options = options.reset_index(drop=True)
    return None

def findGreedyHead(curPlace, distanceLeft, timeLeft, valueFunction):
    options = routes[routes["origin"] == curPlace]
    options = options.reset_index(drop=True)
    
    while len(options) > 0:
        decisionIdx = getDecision(options, valueFunction)
        decision = options.iloc[decisionIdx]
        if decision["dist"] < distanceLeft and decision["time"] < timeLeft:
            res = findGreedy(decision["dest"], distanceLeft - decision["dist"], timeLeft - decision["time"], curPlace, [decision["dest"]], valueFunction)
            if res != None:
                return res
        options = options.drop(decisionIdx)
        options = options.reset_index(drop=True)
    return None

print(findGreedyHead("Volksparkstadion", 90000, 5000, maxScore))