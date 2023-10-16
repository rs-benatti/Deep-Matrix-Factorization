import pandas as pd
from time import time
import numpy as np


def rework_metadata(A):
    df = pd.DataFrame(A, columns=['Movie', 'Genres'])
    genre_dummies = df['Genres'].str.get_dummies('|')
    genre_dummies = genre_dummies.drop('(no genres listed)', axis=1)
    return genre_dummies.transpose()


def get_years(A):
    years = [int(movie.split('(')[-1].split(')')[0]) if '(' in movie else None for movie in A[:, 0]]
    years = [0 if year is None else year for year in years]
    mean = np.mean(years)
    years = [mean if year == 0 else year for year in years]
    years = np.array(years)

    return years


def distance_between_movies(A, years):
    intersection = A.transpose().dot(A)
    # Because it's binarized, the multiplication is the intersections.

    S = np.sum(A, axis=0).values
    # Vector that counts the number of genres for each film

    U = np.ones(len(S))
    union = (S[:, np.newaxis]).dot(U[np.newaxis, :])
    # each line i is 4980 times the cardinal of genres for movies i
    union = union + union.transpose() - intersection

    years = (years[:, np.newaxis]).dot(U[np.newaxis, :])
    years = np.abs(years - years.transpose()) / 300

    # distances_matrix = 1 - intersection/(union + 1e-10) + years

    distances_matrix = np.ones(intersection.shape) * np.inf
    union = union.values
    intersection = intersection.values
    distances_matrix[intersection != 0] = 1 - intersection[intersection != 0] / union[intersection != 0]
    """for i in range(intersection.shape[0]):
        print(i)
        for j in range(intersection.shape[1]):
            if intersection[i][j] == 0:
                continue
            distances_matrix[i][j] = 1 - intersection[i][j] / union[i][j]
    # distances_matrix = 1 - intersection / (union + 1e-10)
    distances_matrix = distances_matrix.values"""
    return distances_matrix


def get_neighbors(distances_matrix, distance_max=0):
    neighbors = []
    for i, row in enumerate(distances_matrix):
        # List to store coordinates and values for the current row
        neighbors_of_row = []
        # Loop over elements in the row
        for j, value in enumerate(row):
            # Check if the value is under the threshold
            if value <= distance_max:
                # Append coordinates and value to the row result list
                neighbors_of_row.append(j)

        # Append the row result list to the main result list
        neighbors.append(neighbors_of_row)

    return neighbors


def add_ratings(R, neighbors):
    print('beginning of added values')

    added_values = np.full(R.shape, False)
    for i in range(R.shape[0]):
        begin = time()
        for j in range(R.shape[1]):
            if not np.isnan(R[i, j]):
                continue
            mean_ratings = 0
            seen_movies_in_neighborhood = 0
            for neighbor in neighbors[j]:
                if not np.isnan(R[i, neighbor]):
                    mean_ratings += R[i, neighbor]
                    seen_movies_in_neighborhood += 1
            if seen_movies_in_neighborhood != 0:
                mean_ratings /= seen_movies_in_neighborhood
                R[i, j] = mean_ratings
                added_values[i, j] = True

        end = time()
        t = end - begin
        print(i, t, np.sum(added_values[i]))
    return R, added_values


def complete_data(train_size=0.8):
    R = np.load("./dataset/ratings_train_" + str(train_size) + ".npy")
    A = np.load("./dataset/namesngenre.npy")
    reworked_A = rework_metadata(A)
    years = get_years(A)
    distances_matrix = distance_between_movies(reworked_A, years)
    neighbors = get_neighbors(distances_matrix)
    max = 0
    for ele in neighbors:
        if len(ele) > max:
            max = len(ele)
    print(max)
    mean = 0
    for ele in neighbors:
        mean += len(ele)
    print(mean / len(neighbors))
    new_R, added_values = add_ratings(R.copy(), neighbors)
    np.save("./dataset/ratings_train_completed_" + str(train_size) + ".npy", new_R)
    np.save("./dataset/completed_mask_" + str(train_size) + ".npy", added_values)


def train_test_split(train_size=0.8):
    # Load the input data from a numpy file
    ratings_train = np.load("./dataset/ratings_train.npy")
    ratings_test = np.load("dataset/ratings_test.npy")

    # Replace NaN values with 0
    ratings_train[np.isnan(ratings_train)] = 0
    ratings_test[np.isnan(ratings_test)] = 0
    total_data = ratings_train + ratings_test

    non_empty_indices = np.where(total_data != 0)
    non_empty_indices = np.array([non_empty_indices[0], non_empty_indices[1]])

    indices = np.random.choice(np.array(range(non_empty_indices.shape[1])), size=non_empty_indices.shape[1],
                               replace=False)

    train_set_indices = indices[0:int(non_empty_indices.shape[1] * train_size)]
    train_set_indices2d = np.array(non_empty_indices[:, train_set_indices])
    train_set = np.zeros(total_data.shape)
    train_set[train_set_indices2d[0], train_set_indices2d[1]] = total_data[
        train_set_indices2d[0], train_set_indices2d[1]]

    test_set_indices = indices[int(non_empty_indices.shape[1] * train_size):]
    test_set_indices2d = np.array(non_empty_indices[:, test_set_indices])
    test_set = np.zeros(total_data.shape)
    test_set[test_set_indices2d[0], test_set_indices2d[1]] = total_data[test_set_indices2d[0], test_set_indices2d[1]]

    train_set[train_set == 0] = np.nan
    test_set[test_set == 0] = np.nan

    np.save("./dataset/ratings_train_" + str(train_size) + ".npy", train_set)
    np.save("./dataset/ratings_test_" + str(train_size) + ".npy", test_set)
