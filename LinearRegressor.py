import numpy as np 

class LinearRegressor():

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

    def train_normal_equation(self, X, y):

        X = self.init_bias_and_weights(X)
        self.weights = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
        print(f'weights: {self.weights}')

    def train(self, X, y):

        X = self.init_bias_and_weights(X)
        converged = False
        count = 0
        while not converged :

            y_hat = np.dot(X, self.weights)

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

        return np.dot(X, self.weights)

if __name__ == "__main__":
    linreg = LinearRegressor(learning_rate=0.0107, convergence_threshold=1e-14)

    X = np.array([
        [11],
        [12],
        [13],
        [14],
        [15],
        [16]
    ])

    y = np.array([1, 2, 3, 4, 5, 6])

    linreg.train(X, y)

    print("Predict: ")

    X_test = np.array([
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [11],
        [12],
        [13],
        [14],
        [15],
        [16],
        [77],
        [777],
        [15.5]
    ])
    print(linreg.predict(X_test))

    linreg = LinearRegressor()
    linreg.train_normal_equation(X, y)

    print("Predict Normal Eq: ")

    print(linreg.predict(X_test))