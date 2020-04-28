import json
from collections import Counter

# Function that returns an array of groundtruth labels for an image. Used in determining true and false positives
def get_groundtruth_labels(filepath):
    with open(filepath, encoding='utf-8-sig') as json_file:
        data = json_file.read()
        data = json.loads(data)
        annotations = data['shapes']
        groundtruth_labels = []
        for item in annotations:
            object_class = item['label']
            groundtruth_labels.append(object_class)
        return groundtruth_labels


# Function to remove elements that occur less than k times
def removeElements(lst, k):
    counted = Counter(lst)

    temp_lst = []
    for el in counted:
        if counted[el] < k:
            temp_lst.append(el)

    res_lst = []
    for el in lst:
        # Want list with only one occurences of found objects that occur more than once
        if (el not in temp_lst) and (el not in res_lst):
            res_lst.append(el)
    return (res_lst)

def match_found_to_groundtruth(found_lst, groundtruth_lst):
    # Get true positives from & of two sets
    true_positives = set(found_lst) & set(groundtruth_lst)
    n_true_positives = len(true_positives)

    # Get false positives from items in found list but not in groundtruth list
    false_positives = set(found_lst) - set(groundtruth_lst)
    n_false_positives = len(false_positives)

    false_negatives = set(groundtruth_lst) - set(found_lst)
    n_false_negatives = len(false_negatives)

    return n_true_positives, n_false_positives, n_false_negatives, true_positives, false_positives, false_negatives
