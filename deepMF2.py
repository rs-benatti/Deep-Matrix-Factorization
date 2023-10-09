import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np





# Define the model
class ParallelLayersModel(nn.Module):
    def __init__(self, input_size, hidden_size_row, hidden_size_col, encoded_dim):
        super(ParallelLayersModel, self).__init__()
        self.row_layer = nn.Linear(input_size[1], hidden_size_row)
        self.row_layer2 = nn.Linear(hidden_size_row, hidden_size_row)

        self.col_layer = nn.Linear(input_size[0], hidden_size_col)
        self.col_layer2 = nn.Linear(hidden_size_col, hidden_size_col)

        self.row_output_layer = nn.Linear(int(hidden_size_row), encoded_dim)
        self.col_output_layer = nn.Linear(int(hidden_size_col), encoded_dim)

        
    def forward(self, rows, cols):
        rows_output = torch.relu(self.row_layer(rows))
        rows_output = torch.relu(self.row_layer2(rows_output))
        rows_output = torch.relu(self.row_output_layer(rows_output))
        #print(rows_output.shape) 
        cols_output = torch.relu(self.col_layer(cols))
        cols_output = torch.relu(self.col_layer2(cols_output))
        cols_output = torch.relu(self.col_output_layer(cols_output))
        #print(cols_output.shape)
        Y_hat = torch.mm(rows_output, cols_output.T)

        cols_output = torch.clamp(cols_output, min=0.0000001)
        rows_output = torch.clamp(rows_output, min=0.0000001)

        row_norms = torch.norm(rows_output, dim=1)
        cols_norms = torch.norm(cols_output, dim=1)
        # Compute the matrix of products using broadcasting
        #product_matrix = row_norms[:, None] * cols_norms
        product_matrix = torch.mm(row_norms[:, None], cols_norms[None, :])

        #print(f"min product {torch.min(product_matrix)}")
        Y_hat = Y_hat/product_matrix
        Y_hat = torch.clamp(Y_hat, max=0.99999, min=0.00001)
        return Y_hat



# Define the training function
def train_model(model, input_data, num_epochs=125, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    weight = input_data.clone()
    weight[weight!=0] = 1
    weight[weight==0] = 0.0000
    loss_fn = nn.BCELoss(weight=weight, reduction='mean')
    #loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
       
        Y_hat = model(input_data, input_data.T)
        #print(similarity_scores.shape)
        #print(torch.max(similarity_scores))
        #print(torch.min(similarity_scores))
        #print(torch.max(labels))
        #print(torch.min(labels))
        loss = loss_fn(Y_hat, input_data)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')
            
    print('Training complete.')


