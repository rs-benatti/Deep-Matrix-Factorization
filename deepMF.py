import torch
import torch.nn as nn
import numpy as np


# Define the model
class ParallelLayersModel(nn.Module):
    def __init__(self, input_size, hidden_size_row, hidden_size_col, genres_size = 0, encoded_dim=10):
        super(ParallelLayersModel, self).__init__()
        self.rmse_train_hist = []
        self.rmse_test_hist = []
        self.genres_size = genres_size
        self.row_layer = nn.Linear(input_size[1], hidden_size_row * 2)
        self.row_layer2 = nn.Linear(hidden_size_row * 2, hidden_size_row)
        self.row_layer3 = nn.Linear(hidden_size_row, int(hidden_size_row / 2))

        self.col_layer = nn.Linear(input_size[0] + genres_size, hidden_size_col * 2)
        self.col_layer2 = nn.Linear(hidden_size_col * 2, hidden_size_col)
        self.col_layer3 = nn.Linear(hidden_size_col, int(hidden_size_col / 2))

        self.row_output_layer = nn.Linear(int(hidden_size_row/2), encoded_dim)
        self.col_output_layer = nn.Linear(int(hidden_size_col/2), encoded_dim)
        self.num_epochs = 0 # useful for plotting analysis later
        self.encoded_dim  = encoded_dim

    def forward(self, rows, cols, genres=None):
        if genres is None:
            genres = torch.FloatTensor(np.zeros((cols.shape[0], self.genres_size)))

        rows_output = torch.relu(self.row_layer(rows))
        rows_output = torch.relu(self.row_layer2(rows_output))
        rows_output = torch.relu(self.row_layer3(rows_output))
        rows_output = torch.relu(self.row_output_layer(rows_output))

        cols_output = torch.relu(self.col_layer(torch.cat((cols, genres), dim = 1)))
        cols_output = torch.relu(self.col_layer2(cols_output))
        cols_output = torch.relu(self.col_layer3(cols_output))
        cols_output = torch.relu(self.col_output_layer(cols_output))

        Y_hat = torch.mm(rows_output, cols_output.T)

        cols_output = torch.clamp(cols_output, min=0.0000001)
        rows_output = torch.clamp(rows_output, min=0.0000001)

        row_norms = torch.norm(rows_output, dim=1)
        cols_norms = torch.norm(cols_output, dim=1)
        # Compute the matrix of products using broadcasting
        product_matrix = torch.mm(row_norms[:, None], cols_norms[None, :])

        Y_hat = Y_hat/product_matrix
        Y_hat = torch.clamp(Y_hat, max = 0.99999, min = 0.00001)
        return Y_hat, row_norms, cols_norms
    
    def numpy_and_round(self, Y_hat):# Y_hat normalized
        return np.round(Y_hat.detach().numpy()*10)/2
        
    def RMSE(self, Y, Y_hat):
        return torch.sqrt(torch.mean((Y_hat[Y != 0] - Y[Y != 0]) ** 2)).item()

    def to_numpy(self, Y_hat):
        return Y_hat.detach()*5


# Define the training function
def train_model(model, optimizer, input_data, weights, genres = None, num_epochs=250,
                test_data=False, lambda_=0, mu_=0):  # Obs.: test_data must not be normalized
    target_train = torch.FloatTensor(input_data * 5)

    if test_data is not False:
        target_test = torch.FloatTensor(test_data)
    loss_fn = nn.BCELoss(weight=weights, reduction='mean')
    model.num_epochs = num_epochs
    #loss_fn = nn.MSELoss()
    rmse_train = []
    rmse_test = []
    times = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        Y_hat, row_norms, cols_norms = model(input_data, input_data.T, genres)
        loss = loss_fn(Y_hat, input_data) + lambda_*torch.norm(row_norms, "fro").item() + mu_*torch.norm(cols_norms, "fro").item()
        loss.backward()
        optimizer.step()

        rmse_train.append(model.RMSE(target_train, Y_hat * 5))
        if test_data is not False:
            rmse_test.append(model.RMSE(target_test, Y_hat * 5))

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

    model.rmse_train_hist = rmse_train
    model.rmse_test_hist = rmse_test

    print('Training complete.')
