from math import sqrt

def search(data):
    current_set_of_features = []

    for i in range(len(data)-1):
        print('On the ', str(i), 'th level of the search tree')
        feature_to_add = []
        best_so_far = 0

        for k in range(len(data)-1):
            if k not in current_set_of_features:
                print('--Considering adding the ', str(k), 'th feature')
                accuracy = leave_one_out_cross_validation(data, current_set_of_features + [k])

                if accuracy > best_so_far:
                    best_so_far = accuracy
                    feature_to_add = k

        current_set_of_features.append(feature_to_add)
        print('On level ', str(feature_to_add), ' I added feature ', str(feature_to_add), ' to the current set')

    return current_set_of_features

def leave_one_out_cross_validation(data, current_set_of_features):
    number_of_correct_predictions = 0

    for i in range(len(data)):
        object_to_classify = data[i]
        label_object_to_classify = data[:i] + data[i+1:]

        nearest_neighbor_distance = float('inf')
        nearest_neighbor_loc = float('inf')

        for j in range(len(data)):
            if j != i:
                distance = sqrt(sum((data[j][i] - object_to_classify[i]) ** 2 for i in current_set_of_features))

                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_loc = j

        if data[nearest_neighbor_loc][-1] == object_to_classify[-1]:
            number_of_correct_predictions += 1

    return number_of_correct_predictions / len(data)
    