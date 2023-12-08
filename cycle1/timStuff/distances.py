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


with open("key", 'rb') as file: 
    apiKey = pickle.load(file) 


placesPart = ""
for place in places.iterrows():
    placesPart += "place_id:{}|".format(place[1]["google_id"])
placesPart = placesPart[:len(placesPart) -1]

requestURL = "https://maps.googleapis.com/maps/api/distancematrix/json?mode=bicycling&destinations=" + placesPart + "&origins=" + placesPart + "&key=" + apiKey

if not os.path.exists("pythonData/" + placesPart):
    #requestReturn = requests.get(requestURL)
    print("Made request")
    with open("pythonData/" + placesPart, 'wb') as file: 
        pickle.dump(requestReturn, file) 
else:
    print("Loaded already existing request")
    with open("pythonData/" + placesPart, 'rb') as file: 
        requestReturn = pickle.load(file)


jsonData = requestReturn.json()

with open('routes.csv', 'w', newline='') as csvfile:
    csvWriter = csv.writer(csvfile)
    csvWriter.writerow(["Origin", "Dest", "Distance in seconds", "Time in seconds"])    
    for i, origin in enumerate(jsonData["origin_addresses"]):
        for j, dest in enumerate(jsonData['destination_addresses']):
            if origin == dest:
                continue
            csvWriter.writerow([addressToName[origin], addressToName[dest], jsonData["rows"][i]["elements"][j]["distance"]["value"], jsonData["rows"][i]["elements"][j]["duration"]["value"]])