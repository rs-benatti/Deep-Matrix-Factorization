
import numpy as np
import os
from tqdm import tqdm, trange
import argparse
import deepMF2
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a completed ratings table.')
    parser.add_argument("--name", type=str, default="ratings_eval.npy",
                      help="Name of the npy of the ratings table to complete")

    args = parser.parse_args()



    # Open Ratings table
    print('Ratings loading...') 
    table = np.load(args.name) ## DO NOT CHANGE THIS LINE
    print('Ratings Loaded.')
    

    # Any method you want
    """average = np.nanmean(table)
    table = np.nan_to_num(table, nan=average)"""

    # Replace NaN values with 0
    table[np.isnan(table)] = 0
    normalized_input_data = table/np.max(table)

    encoded_dim = 32
    input_size = table.shape 
    hidden_size_row = 16
    hidden_size_col = 64
    model = deepMF2.ParallelLayersModel(input_size, hidden_size_row, hidden_size_col, encoded_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    deepMF2.train_model(model, optimizer, torch.FloatTensor(normalized_input_data), num_epochs=130)
    predicted = model(torch.FloatTensor(normalized_input_data), torch.FloatTensor(normalized_input_data).T)
    table = predicted.detach().numpy()

    # Save the completed table 
    np.save("output.npy", table) ## DO NOT CHANGE THIS LINE


        
