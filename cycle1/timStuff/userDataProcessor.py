def processUserData(mapped_age, gender, history, art, nature, museums, churches, sights, funActivities):
    ageVal = 0
    if mapped_age == '6-15':
        ageVal = 0
    elif mapped_age == '16-25':
        ageVal = 0.25
    elif mapped_age == '26-35':
        ageVal = 0.5
    elif mapped_age == '36-45':
        ageVal = 0.75
    elif mapped_age == '46-55':
        ageVal = 1
    elif mapped_age == '56-65':
        ageVal = 1.25
    elif mapped_age == '66-75':
        ageVal = 1.5
    elif mapped_age == '75+':
        ageVal = 1.75

    genderVal = 0
    if gender == 'male':
        genderVal = 0
    elif gender == 'others':
        genderVal = 0.5
    elif gender == 'female':
        genderVal = 1


    return [ageVal, genderVal, history / 100, art / 100, nature / 100, museums / 100, churches / 100, sights / 100, funActivities / 100]