# Import libraries
import numpy as np

def sigmoid(z):
    """
    Compute the sigmoid activation function.

    Args:
        z (numpy.ndarray): Input array.
    
    Returns:
        numpy.ndarray: Sigmoid activation applied element-wise.
    """
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    """
    Compute the derivative of the sigmoid function.

    Args:
        z (numpy.ndarray): Input array.
    
    Returns:
        numpy.ndarray: Element-wise derivative of the sigmoid function.
    """
    s = sigmoid(z)
    return s * (1 - s)
    
def softmax(z):
    """
    Compute the softmax activation function.

    Args:
        z (numpy.ndarray): Input array.
    
    Returns:
        numpy.ndarray: Softmax activation applied along the last axis.
    """
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

class MLP():
    """
    Implementation of Multi-Layer Perceptron (MLP) using only NumPy.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.01, epochs=10000):
        """
        Initialize the MLP model with random weights and biases.
        
        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of hidden layer neurons.
            output_dim (int): Number of output classes.
            learning_rate (float, optional): Learning rate for gradient descent. Defaults to 0.01.
            epochs (int, optional): Number of training iterations. Defaults to 100.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        self.w1 = np.random.rand(self.input_dim, self.hidden_dim) * 0.01
        self.w2 = np.random.rand(self.hidden_dim, self.output_dim) * 0.01
        self.b1 = np.zeros((1, self.hidden_dim))
        self.b2 = np.zeros((1, self.output_dim))
    
    def forward(self, X):
        """
        Perform forward propagation.

        Args:
            X (numpy.ndarray): Input data of shape (num_samples, input_dim).

        Returns:
            tuple: Output probabilities and intermediate activations.
        """
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = softmax(self.z2)    
        return self.a2
    
    def compute_loss(self, Y_true, Y_pred):
        """
        Compute the cross-entropy loss.

        Args:
            Y_true (numpy.ndarray): True labels (one-hot encoded).
            Y_pred (numpy.ndarray): Predicted probabilities.

        Returns:
            float: Computed loss value.
        """
        m = Y_true.shape[0]
        loss = -np.sum(Y_true * np.log(Y_pred + 1e-15)) / m  # Adding epsilon for numerical stability
        return loss
    
    def backward(self, X, Y_true, Y_pred):
        """
        Perform backward propagation and update weights.

        Args:
            X (numpy.ndarray): Input data.
            Y_true (numpy.ndarray): True labels (one-hot encoded).
            Y_pred (numpy.ndarray): Predicted probabilities.
        """
        m = X.shape[0]

        # Compute gradients
        dZ2 = Y_pred - Y_true  # Softmax derivative (cross-entropy loss gradient)
        dW2 = np.dot(self.a1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.w2.T)
        dZ1 = dA1 * sigmoid_derivative(self.z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Update parameters using gradient descent
        self.w1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.w2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    
    def train(self, X_train, Y_train):
        """
        Train the MLP using gradient descent.

        Args:
            X_train (numpy.ndarray): Training input data.
            Y_train (numpy.ndarray): One-hot encoded training labels.
        """
        for epoch in range(self.epochs):
            Y_pred = self.forward(X_train)
            loss = self.compute_loss(Y_train, Y_pred)
            self.backward(X_train, Y_train, Y_pred)
            
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}/{self.epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        """
        Predict the class labels for input data.

        Args:
            X (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Predicted class labels.
        """
        Y_pred = self.forward(X)
        return np.argmax(Y_pred, axis=1)
    

