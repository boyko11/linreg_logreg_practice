import numpy as np 

class LogisticRegressor():

    def __init__(self, initial_weights = None, learning_rate = 0.1, convergence_threshold = 0.0001):
        self.weights = initial_weights
        self.learning_rate = learning_rate
        self.convergence_threshold = convergence_threshold

    def init_bias_and_weights(self, X):
        # add 1 as a first column to account for bias
        X = np.insert(X, 0, 1, axis=1)
        if self.weights is None:
            self.weights = np.random.rand(X.shape[1], 1)

        return X

    def train(self, X, y):

        X = self.init_bias_and_weights(X)
        converged = False
        count = 0
        while not converged :

            y_hat = self.pure_predict(X)

            y_hat_minus_y = y_hat - y.reshape((X.shape[0], 1)) 

            gradient = np.mean(y_hat_minus_y * X, axis = 0)
            
            previous_weights = self.weights.copy()
            self.weights = self.weights - self.learning_rate * gradient[:, np.newaxis]
            
            diff = np.sum(np.abs(self.weights - previous_weights))
            
            if(count % 20000 == 0):
                print(f'Iteration {count} convergence diff: {diff}')

            if(diff < self.convergence_threshold):
                print(f'Converged in {count} iterations')
                print(f"Weights: {self.weights}")
                converged = True
            count += 1

    def predict(self, X):
        # add 1 as a first column to account for bias
        X = np.insert(X, 0, 1, axis=1)

        prob = self.pure_predict(X)

        return (prob >= .5).astype(int)
    
    def pure_predict(self, X):

        return 1 / (1 + np.exp(-np.dot(X, self.weights)))



if __name__ == "__main__":
    
    # any point to the left of 5.5 is a positive instance
    # any point to the right of 5.5 is a negative instance
    # there would be ifinitely many lines for this toy dataset
    # but any line that separates them would do - just to verify the gradient descent implementation
    logreg = LogisticRegressor(learning_rate=0.5, convergence_threshold=1e-4)

    X = np.array([
        [-1],
        [11],
        [0],
        [10],
        [1],
        [9],
        [2],
        [8],        
        [3],
        [7],
        [4],
        [6],        
        [5]
    ])

    y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    logreg.train(X, y)

    print("Predict Train: ")

    print(logreg.predict(X))

    print("Predict Test: ")

    X_test = np.array([
        [2.2],        
        [5.45],
        [5.5],
        [5.55],
        [9.7]    
    ])
    print(logreg.predict(X_test))