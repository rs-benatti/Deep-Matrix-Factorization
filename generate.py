
import numpy as np
import os
from tqdm import tqdm, trange
import argparse
import MF2

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
    #average = np.nanmean(table)
    #table = np.nan_to_num(table, nan=average)
    k = 1

    factorization = MF2.MF(table, l=0.01, mu=0.01, k=k)
    factorization.fit(lr_I=0.0001, lr_U=0.0001, num_iterations=400)
    table = factorization.predict()

    

    # Save the completed table 
    np.save("output.npy", table) ## DO NOT CHANGE THIS LINE


        
