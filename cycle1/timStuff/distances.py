""" 
This script reads:  places.csv
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

addressToName["Stadtpark, Hamburg, Germany"] = "Citypark"
addressToName["Elbphilharmonie Hamburg, 20457 Hamburg, Germany"] = "Elbphilharmonie"
addressToName["Planten un Blomen, Marseiller Promenade, 20355 Hamburg, Germany"] = "Planten un Blomen"
addressToName["Volks Park, 22525 Hamburg, Germany"] = "Altona Volkspark"
addressToName["Alsterpark, 20148 Hamburg, Germany"] = "Alsterpark"
addressToName["Schanzenpark, Schröderstiftstraße, 20357 Hamburg, Germany"] = "Schanzenpark"
addressToName["St Pauli, 20355 Hamburg, Germany"] = "Minigolf course Planten un Blomen"
addressToName["Alter Wandrahm 19, 20457 Hamburg, Germany"] = "Speicherstadt"

places = places[["google_id", "name"]]

"""
# User APIKEY
with open("key", 'rb') as file: 
    apiKey = pickle.load(file) 


# Build request
# Maximum matrix size is 100 elements so we have to split our data up into smaller chunks
placesPartFirstHalf = ""
placesPartSecondHalf = ""
for place in places.iterrows():
    if (place[0] < 20):
        placesPartFirstHalf += "place_id:{}|".format(place[1]["google_id"])
    else:
        placesPartSecondHalf += "place_id:{}|".format(place[1]["google_id"])

placesPartFirstHalf = placesPartFirstHalf[:len(placesPartFirstHalf) -1]
placesPartSecondHalf = placesPartSecondHalf[:len(placesPartSecondHalf) -1]

pairs = [("place_id:{}|place_id:{}".format(places.iloc[i]["google_id"], places.iloc[i + 1]["google_id"]) if i + 1 < len(places) else None) for i in range(0, len(places), 2)]
pairs = pairs[:-1]

responses = []
for pair in pairs:
    requestOne = "https://maps.googleapis.com/maps/api/distancematrix/json?mode=bicycling&destinations=" + pair + "&origins=" + placesPartFirstHalf + "&key=" + apiKey
    requestTwo = "https://maps.googleapis.com/maps/api/distancematrix/json?mode=bicycling&destinations=" + pair + "&origins=" + placesPartSecondHalf + "&key=" + apiKey
    if not os.path.exists("pythonData/" + "requestOne" + pair):
        requestOneReturn = requests.get(requestOne)
        requestOneReturn = requestOneReturn.json()
        with open("pythonData/" + "requestOne" + pair, 'wb') as file: 
            pickle.dump(requestOneReturn, file)
    else:
        with open("pythonData/" + "requestOne" + pair, 'rb') as file: 
            requestOneReturn = pickle.load(file)

    if not os.path.exists("pythonData/" + "requestTwo" + pair):
        requestTwoReturn = requests.get(requestTwo)
        requestTwoReturn = requestTwoReturn.json()
        with open("pythonData/" + "requestTwo" + pair, 'wb') as file: 
            pickle.dump(requestTwoReturn, file)
    else:
        with open("pythonData/" + "requestTwo" + pair, 'rb') as file: 
            requestTwoReturn = pickle.load(file)
    responses.append(requestOneReturn)
    responses.append(requestTwoReturn)

with open("pythonData/" + "allResponses", 'wb') as file: 
    pickle.dump(responses, file)
"""

with open("pythonData/" + "allResponses", 'rb') as file: 
    responses = pickle.load(file)

# Create routes.csv from response
with open('routes.csv', 'w', newline='') as csvfile:
    csvWriter = csv.writer(csvfile)
    csvWriter.writerow(["Origin", "Dest", "Distance in seconds", "Time in seconds"])
    for requestReturn in responses:
        print("-------")
        print(requestReturn)   
        for i, origin in enumerate(requestReturn["origin_addresses"]):
            for j, dest in enumerate(requestReturn['destination_addresses']):
                if origin == dest:
                    continue
                csvWriter.writerow([addressToName[origin], addressToName[dest], requestReturn["rows"][i]["elements"][j]["distance"]["value"], requestReturn["rows"][i]["elements"][j]["duration"]["value"]])

