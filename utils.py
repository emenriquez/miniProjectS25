def get_num_classes(dataset, emnist_split):
    if dataset == 'mnist':
        return 10
    emnist_splits = {
        'byclass': 62,
        'bymerge': 47,
        'balanced': 47,
        'letters': 26,
        'digits': 10,
        'mnist': 10
    }
    return emnist_splits.get(emnist_split, 47)
