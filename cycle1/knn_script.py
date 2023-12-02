# knn_app/knn_script.py
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

def run_knn(age, gender, interests, poi, maxTime, minTime, num_neighbors=5):
    # Sample data (replace with your own data)

    print(f"Age: {age}")
    print(f"Gender: {gender}")
    print(f"Interests: {interests}")
    print(f"POI: {poi}")
    print(f"Max Time: {maxTime}")
    print(f"Min Time: {minTime}")
    data = {
        'feature1': [18, 13, 14, 18, 19, 10],
        'feature2': [17, 14, 15, 18, 21, 31],
        'feature3': [1, 0, 1, 1, 0, 0],  # 'male', 'female', 'child'
        'feature4': [1, 0, 1, 0, 0, 1],  # 'history', 'art', 'nature'
        'feature5': [1, 1, 0, 0, 1, 0],  # 'res', 'park', 'mall', 'hotel', 'stadium'
    }

    df = pd.DataFrame(data)
    X = df[['feature1', 'feature2', 'feature3', 'feature4', 'feature5']]
    y = np.array([18, 12, 15, 13, 17, 13])

    # Create and fit the KNN model
    knn = NearestNeighbors(n_neighbors=num_neighbors)  # Using NearestNeighbors for finding k nearest neighbors
    knn.fit(X)

    # Make a prediction using form data
    new_data = pd.DataFrame({
        'feature1': [age],
        'feature2': [maxTime],  # Adjust parameter name
        'feature3': [1 if gender == 'male' else 0],
        'feature4': [1 if 'history' in interests else 0],
        'feature5': [1 if 'res' in poi else 0],
    })

    # Finding k nearest neighbors
    _, indices = knn.kneighbors(new_data)

    # Extracting similar items from the original dataset
    similar_items = df.iloc[indices[0]]

    return similar_items
