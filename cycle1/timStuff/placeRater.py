# Method creates place rating from place and user stats
def ratePlace(placeStats, personStats):
    sum = 0
    if (personStats[0] == 0):
        sum += placeStats["age1"]
    elif (personStats[0] == 0.25):
        sum += placeStats["age2"]
    elif (personStats[0] == 0.5):
        sum += placeStats["age3"]
    elif (personStats[0] == 0.75):
        sum += placeStats["age4"]
    elif (personStats[0] == 1):
        sum += placeStats["age5"]
    elif (personStats[0] == 1.25):
        sum += placeStats["age6"]
    elif (personStats[0] == 1.5):
        sum += placeStats["age7"]
    elif (personStats[0] == 1.75):
        sum += placeStats["age8"]

    if (personStats[1] == 0):
        sum += placeStats["male"]
    elif (personStats[1] == 0.5):
        sum += placeStats["non-binary"]
    elif (personStats[1] == 1):
        sum += placeStats["female"]
        
    sum += placeStats["history"] * personStats[2]
    sum += placeStats["art"] * personStats[3]
    sum += placeStats["nature"] * personStats[4]
    sum += placeStats["sports"] * personStats[5]
    sum += placeStats["sciences"] * personStats[6]
    sum += placeStats["sights"] * personStats[7]
    sum += placeStats["fun_activities"] * personStats[8]
    return sum