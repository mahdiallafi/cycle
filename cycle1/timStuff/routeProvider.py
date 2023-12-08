import pandas as pd
import csv
import pickle
import requests
from userGenerationAndEval import getRatings
from userGenerationAndEval import getCentroidOrder

with open("pythonData/ratingDict", 'rb') as file: 
    ratingDict = pickle.load(file)

with open("pythonData/bestCentroidSet-size:100", 'rb') as file: 
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

# Greedy Implementation
def maxScore(option):
    return placeRatings[option.dest]

def maxScoreTime(option):
    return placeRatings[option.dest] / (option.time / 1000)

def maxScoreDistance(option):
    return placeRatings[option.dest] / (option.dist / 1000)
    
def getDecision(options, valueFunction):
    maxValue = 0
    for opt in options.itertuples():
        if valueFunction(opt) > maxValue:
            maxValue = valueFunction(opt)
            decisionIdx = opt.Index
    return decisionIdx

def findGreedy(curPlace, distanceLeft, timeLeft, goal, visited, valueFunction):
    if curPlace == goal:
        return [curPlace]
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
                return res + [curPlace]
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

from collections import namedtuple
State = namedtuple("State", ["place", "distance", "time"])
gamma = 1
alpha = 0.8

def available_actions(state):
    current_state_row = placeRatingsIndexed[state.place]
    existingOptions = np.where(current_state_row >= 0)[1]
    realisticOptions = []
    for opt in existingOptions:
        newOpt = State(opt, (int) (state.distance * distanceStep - routes.loc[(routesIndexed['origin'] == state.place) & (routesIndexed['dest'] == opt), "dist"].values[0]) // distanceStep, (int) (state.time * timeStep - routes.loc[(routesIndexed['origin'] == state.place) & (routesIndexed['dest'] == opt), 'time'].values[0]) // timeStep)
        if newOpt.distance >= 0 and newOpt.time >= 0:
            realisticOptions.append(newOpt)
    return realisticOptions

def update(current_state, action, gamma, alpha):
    max_value = np.max(Q[action.place, :, action.distance , action.time])
    Q[current_state.place, action.place, current_state.distance, current_state.time] = (1-alpha)*Q[current_state.place, action.place, current_state.distance, current_state.time] + alpha * (placeRatingsIndexed[action.place] + gamma * max_value)
    return 

def scoreQ():
    return (np.sum(Q / np.max(Q) * 100))

def modelGetSteps(Q, startState):
    destination = startState.place # .place is Index not in "name"
    steps = []
    visited = []
    current_state = startState
    
    while True:
        next_step_list = np.where(Q[current_state.place, :, current_state.distance, current_state.time] == np.max(Q[current_state.place, :, current_state.distance, current_state.time]))[0]
        if len(next_step_list) == len(Q.shape[0]):
            # RAN OUT OF RESSOURCES
            if (steps[-1].place == destination):
                break
            elif len(steps) == 1:
                return []
            Q[steps[-2].place, steps[-1].place, steps[-2].distance, steps[-2].time] = 0
            current_state = steps[-2]
            steps = steps[:-1]
            visited = visited[:-1]
            continue
        elif next_sftep_list[0] in visited or current_state.distance * distanceStep < routes.loc[(routesIndexed['origin'] == current_state.place) & (routesIndexed['dest'] == next_step_list[0]), "dist"].values[0]:
            # Top pick was already visited or is unaffordable
            # We will set score to zero and redo
            Q[current_state.place, next_step_list[0], current_state.distance, current_state.time] = 0
            continue
     
        next_step = State(next_step_list[0], (int) (current_state.distance * distanceStep - routes.loc[(routesIndexed['origin'] == current_state.place) & (routesIndexed['dest'] == next_step_list[0]), "dist"].values[0]) // distanceStep, (int) (current_state.time * timeStep - routes.loc[(routesIndexed['origin'] == current_state.place) & (routesIndexed['dest'] == next_step_list[0]), 'timeCost'].values[0]) // timeStep)
        steps.append(next_step)
        visited.append(next_step.place)
        
        current_state = next_step

    return steps

# Training

distanceStep = 100 # in m
timeStep = 180 # in seconds
distanceIntervalls = 100
timeIntervalls = 80

MAXDISTANCE = distanceStep * distanceIntervalls
MAXTIME = timeStep * timeIntervalls

print("Will be trained to deal with maximum start distance of:", MAXDISTACNE, "m")
print("Will be trained to deal with maximum start time of:", MAXTIME, "s")

Q = np.zeros([len(places), len(places), distanceIntervalls, timeIntervalls])
print("Shape of our matrix:", Q.shape)

# maybe iterative instead of random training?
scores = []
trainingTime = 10000

for i in range(trainingTime):
    if (i % 20000 == 0) and i != 0:
        score = scoreQ()
        scores.append(score)
        print("Score for i =", i, " ", str(score))
        
    current_state = State(np.random.randint(0, int(Q.shape[0])), np.random.randint(1, Q.shape[2]), np.random.randint(1, Q.shape[3]))
    available_act = available_actions(current_state)
    if len(available_act) == 0:
        continue
    action = available_act[np.random.randint(0, len(available_act))]
    update(current_state, action, gamma, alpha)

plt.plot(scores)
plt.show()



with open("shape(20, 20, 55, 60)ms6ts700g0.8a0.8times7060000+", 'wb') as file: 
    pickle.dump(Q, file) 


personStats =  [0,0,0,0,0,0,0,0,1]
placeRatings = {}
placeRatingsIndexed = {}
for i, rating in enumerate(getRatings(0, personStats, centroids, ratingDict)):
    placeRatings[places.iloc[i]["name"]] = rating
    placeRatingsIndexed[i] = rating

print(placeRatings)

print(findGreedyHead("Volksparkstadion", 90000, 5000, maxScore))