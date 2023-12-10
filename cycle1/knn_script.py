def run_knn(age, gender, maxDestination, minDestination, history, art, nature, sights, funActivities,origin,destination, num_neighbors=5):
    # Sample data (replace with your own data)

   

    print(f"Age: {age}")
    print(f"Gender: {gender}")
    print(f"history: {history}")
    print(f"art: {art}")
    print(f"nature: {nature}")
    print(f"sights: {sights}")
    print(f"funActivities: {funActivities}")
    print(f"maxDestination: {maxDestination}")
    print(f"minDestination: {minDestination}")
    print(f"orgin: {origin}")
    print(f"destination: {destination}")

    # Make a prediction using form data
    similar_items = [
        {'locations': 'Brandenburg Gate', 'address': 'Sylvesterallee 7, 22525 Hamburg, Germany', 'description': 'test', 'website': 'https://example.com/brandenburg_gate'},
        {'locations': 'Berlin Cathedral', 'address': 'Harald-Stender-Platz 1, 20359 Hamburg, Germany', 'description': 'https://example.com/berlin_cathedral_image.jpg', 'website': 'https://example.com/berlin_cathedral'},
        {'locations': 'Checkpoint Charlie', 'address': 'Jakobikirchhof 22, 20095 Hamburg, Germany', 'description': 'https://example.com/checkpoint_charlie_image.jpg', 'website': 'https://example.com/checkpoint_charlie'},
        {'locations': 'East Side Gallery', 'address': 'Mönckebergstraße, 20095 Hamburg, Germany', 'description': 'https://example.com/east_side_gallery_image.jpg', 'website': 'https://example.com/east_side_gallery'},
    ]

    return similar_items
