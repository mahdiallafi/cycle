import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler



"""
Definition of Dataset features:

Age: `6 - 15`, `16 - 25`, `26 - 35`, `36 - 45`, `46 - 55`, `56 - 65`, `66 - 75`, `75+` (Card. 8)

Gender: Male, Female, Non-Binary (Card. 3)

History, Art, Nature, Sports, Sciences, Sights, Fun Activities: Continuous between 0% - 100% interest (sliders)
"""

columns = ["age", "gender", "history", "art", "nature", "museums", "churches", "sights", "fun_activities"]

# Create randomized users

# Create an empty DataFrame with specified columns
users = pd.DataFrame(columns=columns)
# Generate blobs
blobsCluster, labels = make_blobs(n_samples=10000, n_features=len(users.columns), centers=30)
scaler = MinMaxScaler()
blobsCluster = scaler.fit_transform(blobsCluster)
blobsRandom = np.random.random_sample((10000, 9))
blobs = np.concatenate((blobsCluster, blobsRandom))

# Create a DataFrame using the generated blobs
users = pd.DataFrame(blobs, columns=users.columns)

male_mask = users['gender'] < 0.45
divers_mask = (users['gender'] >= 0.49) & (users['gender'] <= 0.51)
female_mask = users['gender'] > 0.55
users.loc[male_mask, 'gender'] = 0
users.loc[divers_mask, 'gender'] = 0.5
users.loc[female_mask, 'gender'] = 1

for i in range(8):
    age_mask = (users['age'] > i * (1/8)) & (users['age'] < i * (1/8) + 1/8)
    users.loc[age_mask, 'age'] = i * (0.25)

# Now we could save them with pickle but we don't want to override our existing dataset
"""
with open("pythonData/users", 'wb') as file: 
    pickle.dump(users, file) 
"""