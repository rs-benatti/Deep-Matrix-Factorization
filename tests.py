import numpy as np
import torch
import deepMF as dmf
import data_completer as dc
import matplotlib.pyplot as plt
from data_completer import rework_metadata

dc.train_test_split(train_size=0.8)
# dc.complete_data(train_size=0.8)

train_size = 0.8
# Load the input data from a numpy file
# ratings_train = np.load("./dataset/ratings_train_completed_" + str(train_size) + ".npy")
ratings_train2 = np.load("./dataset/ratings_train_" + str(train_size) + ".npy")
ratings_test = np.load("./dataset/ratings_test_" + str(train_size) + ".npy")
# added_values_mask = np.load("./dataset/completed_mask_" + str(train_size) + ".npy")

# Replace NaN values with 0
# ratings_train[np.isnan(ratings_train)] = 0
ratings_train2[np.isnan(ratings_train2)] = 0
ratings_test[np.isnan(ratings_test)] = 0

weights = np.zeros(ratings_test.shape)
# weights[added_values_mask is True] = 0.005
# weights[ratings_train != 0] = 1

weights2 = np.zeros(ratings_train2.shape)
weights2[ratings_train2 != 0] = 1

weight2 = torch.FloatTensor(ratings_train2)
weights2[weights2 != 0] = 1
weights2[weights2 == 0] = 0.0

# normalized_train_data = ratings_train / np.max(ratings_train)
normalized_train_data2 = ratings_train2 / np.max(ratings_train2)

normalized_test_data = ratings_test / np.max(ratings_test)

# Add genres of movies
A = np.load("./dataset/namesngenre.npy")
genres_train = rework_metadata(A)
genres_train = genres_train.to_numpy()

# Create an instance of the model
input_size = ratings_train2.shape
hidden_size_row = 16
hidden_size_col = 64

# Training the model
optimizers = []
"""model = ParallelLayersModel(input_size, hidden_size_row, hidden_size_col)
optimizers.append((model, optim.Adam(model.parameters(), lr=0.001)))
model = ParallelLayersModel(input_size, hidden_size_row, hidden_size_col)
optimizers.append((model, optim.RAdam(model.parameters(), lr=0.001)))
model = ParallelLayersModel(input_size, hidden_size_row, hidden_size_col)
optimizers.append((model, optim.AdamW(model.parameters(), lr=0.001)))
model = ParallelLayersModel(input_size, hidden_size_row, hidden_size_col)
optimizers.append((model, optim.Rprop(model.parameters(), lr=0.001)))"""
for i in range(10):
    model = dmf.ParallelLayersModel(input_size, hidden_size_row, hidden_size_col, genres_train.shape[0], encoded_dim=32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)

    model2 = dmf.ParallelLayersModel(input_size, hidden_size_row, hidden_size_col, encoded_dim=32)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=0.001, weight_decay=0.0001)

    dmf.train_model(model=model, optimizer=optimizer, input_data=torch.FloatTensor(normalized_train_data2),
                    weights=torch.FloatTensor(weights2), genres = torch.FloatTensor(genres_train.T), num_epochs=200, test_data=ratings_test)
    dmf.train_model(model=model2, optimizer=optimizer2, input_data=torch.FloatTensor(normalized_train_data2),
                    weights=torch.FloatTensor(weights2), num_epochs=200, test_data=ratings_test)
    # Pass the new input data through the trained model to get predictions
    predicted = model(torch.FloatTensor(normalized_train_data2),
                      torch.FloatTensor(normalized_train_data2).T, torch.FloatTensor(genres_train.T))
    predicted2 = model2(torch.FloatTensor(normalized_train_data2),
                       torch.FloatTensor(normalized_train_data2).T)
    if i == 0:
        plt.plot(model.rmse_test_hist, color="r", label="Model with genres")
        plt.plot(model2.rmse_test_hist, color="b", label="Normal deepMF")
    else:
        plt.plot(model.rmse_test_hist, color="r")
        plt.plot(model2.rmse_test_hist, color="b")
plt.xlabel("Iterations")
plt.ylabel("RMSE")
plt.title("RMSE comparaison between deepMF with and without taking into account genres of movies")
plt.legend(loc="upper right")
plt.show()