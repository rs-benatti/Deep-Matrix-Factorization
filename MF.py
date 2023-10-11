import numpy as np
from scipy.stats import norm, fit
from scipy.optimize import fsolve
from math import sqrt


class Entry:
    def __init__(self, i, j, value):
        self.i = i
        self.j = j
        self.value = value

class MF:
    def __init__(self, R, l, mu, k):
        self.l = l
        self.mu = mu
        self.k = k
        self.R = R
        
        self.idict = dict()
        self.jdict = dict()
        self.validIndicesR = np.where(~np.isnan(R))
        self.validIndicesIter = []
        for i in range(len(self.validIndicesR[0])):
            self.validIndicesIter.append((self.validIndicesR[0][i], self.validIndicesR[1][i]))
        for i, j in self.validIndicesIter:
            value = R[i, j]
            entry = Entry(i, j, value)
            if not(i in self.idict):
                self.idict[i] = []
            self.idict[i].append(entry)
            if not(j in self.jdict):
                self.jdict[j] = []
            self.jdict[j].append(entry)
        
        mean = self.R[~np.isnan(self.R)].mean()
        self.I = (np.ones((self.R.shape[0], self.k)) / self.k**0.5) * mean**0.5
        self.U = (np.ones((self.R.shape[1], self.k)) / self.k**0.5) * mean**0.5
        '''
        for i in range(self.I.shape[0]):
            for j in range(k):
                self.I[i, j] = self.I[i, j] * 2*((j+1)/k)
        for i in range(self.U.shape[0]):
            for j in range(k):
                self.U[i, j] = self.U[i, j] * 2*((j+1)/k)
        '''
    def getReleventJs(self, i):
        return self.idict[i]
    
    def getReleventIs(self, j):
        return self.jdict[j]
    
    def Cost(self):
        '''
        C =  # ∥R - IU^T∥² + λ∥I∥² + µ∥U∥²
        '''
        estimated = self.I @ (self.U.T)
        loss_term = 0
        for i, j in self.validIndicesIter:
            loss_term += (estimated[i, j] - self.R[i, j])**2
        I_penalization_term = self.l * np.trace(self.I.T @ self.I)
        U_penalization_term = self.mu * np.trace(self.U.T @ self.U)
        return loss_term + I_penalization_term + U_penalization_term
        
        
    def predict(self):
        return self.I @ self.U.T

    def RMSE(self):
        """
        Calculate the Root Mean Squared Error (RMSE) between two vectors.

        Parameters:
        - predictions: A NumPy array or list of predicted values.
        - targets: A NumPy array or list of target (true) values.

        Returns:
        - rmse_value: The RMSE between the two vectors.
        """
        # Ensure predictions and targets are NumPy arrays
        predictions = self.predict()[self.validIndicesR]
        targets = self.R[self.validIndicesR]

        # Calculate the squared differences between predictions and targets
        squared_errors = (predictions - targets) ** 2

        # Calculate the mean of squared errors
        mean_squared_error = squared_errors.mean()

        # Calculate the square root to get RMSE
        rmse_value = sqrt(mean_squared_error)

        return rmse_value
    
    def RMSE_(self, predictions):
        """
        Calculate the Root Mean Squared Error (RMSE) between two vectors.

        Parameters:
        - predictions: A NumPy array or list of predicted values.
        - targets: A NumPy array or list of target (true) values.

        Returns:
        - rmse_value: The RMSE between the two vectors.
        """
        # Ensure predictions and targets are NumPy arrays
        predictions = predictions[self.validIndicesR]
        targets = self.R[self.validIndicesR]

        # Calculate the squared differences between predictions and targets
        squared_errors = (predictions - targets) ** 2

        # Calculate the mean of squared errors
        mean_squared_error = squared_errors.mean()

        # Calculate the square root to get RMSE
        rmse_value = sqrt(mean_squared_error)

        return rmse_value
    
    def accuracy(self, predictions):
        """
        Calculate the accuracy between predicted values and target (true) values.

        Parameters:
        - predictions: A NumPy array or list of predicted values.
        - threshold: A threshold value (default is 0.5) for binary classification.

        Returns:
        - accuracy_value: The accuracy between the predicted and target values.
        """
        # Ensure predictions and targets are NumPy arrays
        predictions = predictions[self.validIndicesR]
        targets = self.R[self.validIndicesR]
        
        diff = predictions - targets
        # Calculate the number of correct predictions
        correct_predictions =  len(diff) - np.count_nonzero(diff)

        # Calculate total number of predictions
        total_predictions = len(targets)

        # Calculate accuracy
        accuracy_value = correct_predictions / total_predictions

        return accuracy_value

    
    def calculate_gradient_iiq(self, i, q):
        releventJs = self.getReleventJs(i)
        gradient = 0
        for entry in releventJs:
            stepValue = entry.value
            for s in range(self.k):
                stepValue -= self.I[i, s] * self.U[entry.j, s]
            stepValue *= -self.U[entry.j, q]
            gradient += 2*stepValue
        gradient += 2*self.l*self.I[i, q]
        return gradient
    
    def calculate_gradient_ujq(self, j, q):
        releventIs = self.getReleventIs(j)
        gradient = 0
        for entry in releventIs:
            stepValue = entry.value
            for s in range(self.k):
                stepValue -= self.I[entry.i, s] * self.U[j, s]
            stepValue *= -self.I[entry.i, q]
            gradient += 2*stepValue
        gradient += 2*self.mu*self.U[j, q]
        return gradient

    
    def fit(self, lr_I, lr_U, num_iterations):
        '''
        for iteration in range(num_iterations):
            U = self.U
            I = self.I
            U_grad = -2 * np.dot(R.T, I) + 2 * np.dot(U, np.dot(I.T, I)) + 2 * mu * U
            I_grad = -2 * np.dot(R, U) + 2 * np.dot(I, np.dot(U.T, U)) + 2 * l * I
            self.I -= lr_I*I_grad
            self.U -= lr_U*U_grad
            
            cost = self.C(R, I, U, l, mu)

            print(f"Iteration {iteration + 1}: Cost = {cost}")
        '''
        num_users = self.R.shape[0]
        num_items = self.R.shape[1]
        for it in range(num_iterations):
            gradients_I = np.zeros_like(self.I)  # Initialize gradients for I matrix
            gradients_U = np.zeros_like(self.U)  # Initialize gradients for U matrix
            
            # Calculate gradients for I matrix
            for i in range(num_users):
                for q in range(self.k):
                    gradient_iiq = self.calculate_gradient_iiq(i, q)
                    gradients_I[i, q] = gradient_iiq

            # Calculate gradients for U matrix
            for j in range(num_items):
                for q in range(self.k):
                    gradient_ujq = self.calculate_gradient_ujq(j, q)
                    gradients_U[j, q] = gradient_ujq
            self.U -= lr_U * gradients_U
            self.I -= lr_I * gradients_I

            print(f"Iteration {it + 1}: Cost = {self.Cost()}. RMSE = {self.RMSE()}")
        #print("I:",self.I)
        #print("U:",self.U)
        