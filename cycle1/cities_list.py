import pandas as pd

# Assuming your dataset is a CSV file with tab-separated values
cities = pd.read_csv('/home/mahdi/cycleing/python/cycleapp/cycle1/timStuff/places.csv', delimiter=';', low_memory=False)

# Display the second column
second_column = cities.iloc[:, 1]
return second_column
