""" 
This script reads:  users
                    places.csv

It finds the best k (100) centroids and stores them and the ratingtables for all centroids
"""

import pandas as pd
import numpy as np
import csv
import pickle
import matplotlib.pyplot as plt
from .placeRater import ratePlace

# Load in already created randomized users
with open("cycle1/timStuff/pythonData/users", 'rb') as file: 
    users = pickle.load(file) 

# THIS FOLLOWING SECTION IS THE K-MEANS:

# Determines which is the closest centroid for each row of data
def getLabels(centroids, df):
    distances = []
    numberOfClusters = len(centroids)
    for k in range(0, numberOfClusters):
        centerX = centroids[k]
        # Hardcoded number of features here
        distances.append(np.sqrt((df.values[:, 0] - centerX[0])**2 + (df.values[:, 1] - centerX[1])**2 + (df.values[:, 2] - centerX[2])**2 + (df.values[:, 3] - centerX[3])**2 + (df.values[:, 4] - centerX[4])**2 + (df.values[:, 5] - centerX[5])**2 + (df.values[:, 6] - centerX[6])**2 + (df.values[:, 7] - centerX[7])**2 + (df.values[:, 8] - centerX[8])**2))
        # Somehow not the same ? :( distances.append(np.sqrt([(df.values[:, i] - centerX[i])**2 for i in range(len(df.columns))]))
    return np.argmin(distances, axis=0)

# Returns the average distance of centroids to their labeled values
def estimateError(centroids, df):
    values = df.values
    totalError = 0    
    for label in df["label"].unique():
        totalError += np.sum(np.linalg.norm(df[df["label"] == label].values[:, :-1] - centroids[label], axis=1))
        
    return totalError / len(df.values)

# Returns the k centroids and the labeling for a given dataset
def getKCentroids(k, df):
    centroids = dict()

    for i in range(0,k):
        centroids[i]= [np.random.uniform(0,1) for i in range(len(df.columns))]
    # how many times I will adjust the centroids
    # Source: Provided Example of Lecture
    for i in range(15):
        # Hardcoded features here:
        df = pd.DataFrame(dict(age=df.values[:, 0], gender=df.values[:, 1], history=df.values[:, 2], art=df.values[:, 3], nature=df.values[:, 4], sports=df.values[:, 5], sciences=df.values[:, 6], sights=df.values[:, 7], fun_activities=df.values[:, 8], label=getLabels(centroids,df)))
        grouped = df.groupby('label')

        # Hardcoded features here:
        for key, group in grouped:
            ac = group['age'].mean()
            bc = group['gender'].mean()
            cc = group['history'].mean()
            dc = group['art'].mean()
            ec = group['nature'].mean()
            fc = group['sports'].mean()
            gc = group['sciences'].mean()
            hc = group['sights'].mean()
            ic = group['fun_activities'].mean()
            centroids[key] = [ac, bc, cc, dc, ec, fc, gc, hc, ic]
    return (centroids, df)

# Returns the feature vector sorted by most to least significant feature
def getKFeatures(k, df):
    bestCentroids, bestLabeling = getKCentroids(k, df)
    centroidOrder = bestLabeling["label"].value_counts().sort_values(ascending=False).index.tolist()
    returnList = []
    for x in centroidOrder:
        returnList.append(bestCentroids[x])

    # On very rare occasion it happens a centroid is not used as a label at all. In that case he will still be part of the feature Vector
    if len(returnList) < k:
        for centroid in bestCentroids.values():
            if centroid not in returnList:
                returnList.append(centroid)
        while len(returnList) < k:
            returnList.append(list(bestCentroids.values())[0])
            # if a picture is only black we still need k centroids
    return returnList

# For visualising how the error behaves for changing amount of centroids
def getElbow(df, lowerbound, upperbound):
    errors = []
    # how man centroids i will try out
    for i in range(lowerbound,upperbound):
        errors.append((i, estimateError(*getKCentroids(i, df))))
        
    x = [item[0] for item in errors]
    y = [item[1] for item in errors]
    plt.plot(x, y)
    plt.xlabel("Number of Centroids")
    plt.ylabel("Error")
    return plt
# K-MEANS IMPLEMENTATION OVER

# Look for a good set of centroids
# We will choose 100 for our amount of centroids
centroidAmount = 100

# Loads in prev. determined best centroids
fileName = "bestCentroidSet-size:{}".format(centroidAmount)

with open("cycle1/timStuff/pythonData/" + fileName, 'rb') as file: 
    bestCentroids = pickle.load(file) 
with open("cycle1/timStuff/pythonData/" + fileName + "ERROR", 'rb') as file:     
    lowestError = pickle.load(file)

"""
# This piece of code can be used to retrain or try to find better ones
#lowestError = 999
#bestCentroids = []

for i in range(1):
    centroids, df = getKCentroids(centroidAmount, users)
    error = estimateError(centroids, df)
    if error < lowestError:
        lowestError = error
        bestCentroids = centroids

with open("cycle1/timStuff/pythonData/" + fileName, 'wb') as file: 
    pickle.dump(bestCentroids, file)
with open("cycle1/timStuff/pythonData/" + fileName + "ERROR", 'wb') as file:     
    pickle.dump(lowestError, file)
print(lowestError)
"""

# Returns our centroids sorted by how close they are to given data
def getCentroidOrder(data):
    list = []
    for number, centroid in bestCentroids.items():
        # hardcoded number of features here
        dist = np.sqrt((data[0] - centroid[0])**2 + (data[1] - centroid[1])**2 + (data[2] - centroid[2])**2 + (data[3] - centroid[3])**2 + (data[4] - centroid[4])**2 + (data[5] - centroid[5])**2 + (data[6] - centroid[6])**2 + (data[7] - centroid[7])**2 + (data[8] - centroid[8])**2)
        list.append((dist, number))
    return sorted(list)

# Time to create the ratinglist for each centroid

# Load in places
"""
ratingColumns = ["age1","age2","age3","age4","age5","age6","age7","age8","male","non-binary","female","history","art","nature","museums","churches","sights","fun_activities"]
places=pd.read_csv('places.csv', sep = ';', names=["google_id", "name", "description", "googleMapsURL", "address"] + ratingColumns)
places[ratingColumns] = places[ratingColumns].astype(float)

# Creates lookup table for [centroid (by index), ratingsList] pairs
ratingDict = {}
for id, values in bestCentroids.items():
    R = []
    for sight in places.iterrows():
        R.append(ratePlace(sight[1], values))
    ratingDict[id] = R
with open("cycle1/timStuff/pythonData/ratingDict", 'wb') as file: 
    pickle.dump(ratingDict, file)

"""