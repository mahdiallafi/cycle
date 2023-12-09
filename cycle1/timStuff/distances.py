""" This script reads:  places.csv
                        key (API Key)

    It takes the places and request the distance matrix from google maps API and saves it under routes.csv
"""
import pandas as pd
import csv
import pickle
import requests
import os


# Load the places dataframe
ratingColumns = ["age1","age2","age3","age4","age5","age6","age7","age8","male","non-binary","female","history","art","nature","sports","sciences","sights","fun_activities"]
places=pd.read_csv('places.csv', sep = ';', names=["google_id", "name", "description", "googleMapsURL", "address"] + ratingColumns)
addressToName = {}
for place in places.iterrows():
    addressToName[place[1]["address"]] = place[1]["name"]

places = places[["google_id", "name"]]

# For testing purposes we will only work with 5 places
places = places[:5]

# User APIKEY
with open("key", 'rb') as file: 
    apiKey = pickle.load(file) 

# Build request
placesPart = ""
for place in places.iterrows():
    placesPart += "place_id:{}|".format(place[1]["google_id"])
placesPart = placesPart[:len(placesPart) -1]

requestURL = "https://maps.googleapis.com/maps/api/distancematrix/json?mode=bicycling&destinations=" + placesPart + "&origins=" + placesPart + "&key=" + apiKey

# Make request but check before it we have already done it before
if not os.path.exists("pythonData/" + placesPart):
    requestReturn = requests.get(requestURL)
    requestReturn = requestReturn.json()
    print("Made request")
    with open("pythonData/" + placesPart, 'wb') as file: 
        pickle.dump(requestReturn, file) 
else:
    print("Loaded already existing request")
    with open("pythonData/" + placesPart, 'rb') as file: 
        requestReturn = pickle.load(file)

# Create routes.csv from response
with open('routes.csv', 'w', newline='') as csvfile:
    csvWriter = csv.writer(csvfile)
    csvWriter.writerow(["Origin", "Dest", "Distance in seconds", "Time in seconds"])    
    for i, origin in enumerate(requestReturn["origin_addresses"]):
        for j, dest in enumerate(requestReturn['destination_addresses']):
            if origin == dest:
                continue
            csvWriter.writerow([addressToName[origin], addressToName[dest], requestReturn["rows"][i]["elements"][j]["distance"]["value"], requestReturn["rows"][i]["elements"][j]["duration"]["value"]])