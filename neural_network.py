import numpy as np 

np.random.seed(51)

"""
You can specify the parameters for "num_layers" with a tuple. Ex: (5, 3, 5, 5) - input layer with 5 nodes,
and two hidden layer with 3 and 5 nodes correspondingly, and an output layer with 5 nodes.
"""


def sigmoid(z):
        value =  1/(1+np.exp(-z))
        return value

def ReLU(z):
        return np.maximum(0,z)

def tanh(z):
        sinh = 0.5 * (np.exp(z) - np.exp(-z))
        cosh = 0.5 * (np.exp(z) + np.exp(-z))
        return sinh/cosh 
def ReLU_derivative(z):
        return np.where(z > 0, 1, 0)  # Correct way to implement ReLU derivative

def sigmoid_derivative(z):
        sigmoid_values =  sigmoid(z)
        return sigmoid_values * (1 - sigmoid_values)
def tanh_derivative(z):
        tanh_values = tanh(z)
        return 1 - tanh_values**2

class NeuralNetwork():
    def __init__(self, num_layers):
        """Num_layers will be a list (length describes dimensions, values describe nodes)"""
        self.num_layers = num_layers
        self.caches = []
    
    def init_params(self):
        self.params = {}
        self.layers = len(self.num_layers) 
        
        for l in range(1, self.layers):
            self.params["W" + str(l)] = 0.1 * np.random.randn(self.num_layers[l], self.num_layers[l - 1])
            self.params["b" + str(l)] = np.zeros((self.num_layers[l], 1))
        
        return self.params
    def forward_propagation(self, A, W, b):
        Z = np.dot(W, A) + b
        linear_cache = (A, W, b)
        return Z, linear_cache
    def forward_activation(self, A_prev, W, b, activation_function):
        Z, self.linear_cache = self.forward_propagation(A_prev, W, b)
        if activation_function == "sigmoid":
            A = sigmoid(Z) 
        elif activation_function == "relu":
            A = ReLU(Z)
        elif activation_function == "tanh":
            A = tanh(Z)
        else: 
            raise ValueError("No valid function. Choose between sigmoid ('sigmoid', rectified linear unit ('relu'), and hyperbolic tangent ('tanh')")
        self.activation_caches = Z
        cache = self.linear_cache, self.activation_caches
        return A, cache
    def forward_implementation(self, X, activation_function):
        A = X 
        L = int(len(self.params)/2)
        for l in range(1, L):
            W = self.params['W' + str(l)]
            b = self.params['b' + str(l)]
            A_prev = A 
            A, cache = self.forward_activation(A_prev, W, b, activation_function) # why do we cache the z though?
            self.caches.append(cache)
        W = self.params['W' + str(L)]
        b = self.params['b' + str(L)]
        AL, cache = self.forward_activation(A, W, b, "sigmoid")
        self.caches.append(cache)
        return AL, self.caches
    def backward_propagation(self, A, activation_function):
        if activation_function == "sigmoid":
            dA = sigmoid_derivative(A)
        elif activation_function == "relu":
            dA = ReLU_derivative(A)
        elif activation_function == "tanh":
            dA = tanh_derivative(A)
        else:
            raise ValueError("No valid function. Choose between sigmoid ('sigmoid', rectified linear unit ('relu'), and hyperbolic tangent ('tanh')")
        return dA
    def compute_cost(self, AL, Y):
        cost = -1/m * np.sum(Y * np.log(AL) + (1-Y)*np.log(1-AL))
        cost = np.squeeze(cost)
        return cost 
    def backward_activation(self, AL, Y, activation_function):
        m = Y.shape[1]
        L = len(self.caches)
        gradients = {}
        
        current_cache = self.caches[L-1]
        linear_cache, activation_cache = current_cache
        Z = activation_cache
        
        # Initialize backpropagation for the last layer (Lth layer)
        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        dZ = dAL * self.backward_propagation(Z, "sigmoid")
        
        A_prev, W, _ = linear_cache
        dW = 1 / m * np.dot(dZ, A_prev.T)
        db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        
        gradients["dZ" + str(L-1)] = dZ
        gradients["dW" + str(L)] = dW
        gradients["db" + str(L)] = db
        
        # Loop backward through layers L-1 to 1
        for l in reversed(range(L - 1)):
            linear_cache, activation_cache = self.caches[l]
            A_prev, W, _ = linear_cache
            Z = activation_cache
            
            dZ = dA_prev * self.backward_propagation(Z, activation_function)
            dW = 1 / m * np.dot(dZ, A_prev.T)
            db = 1 / m * np.sum(dZ, axis=1, keepdims=True)
            dA_prev = np.dot(W.T, dZ)
            
            gradients["dZ" + str(l)] = dZ
            gradients["dW" + str(l + 1)] = dW
            gradients["db" + str(l + 1)] = db
            
            print(gradients["dZ"+str(l)])
            print(gradients["dW"+str(l + 1)])
            print(gradients["db" + str(l + 1)])
        
        return gradients
    def update_parameters(self, gradients, learning_rate):
        parameters = self.params
        L = (len(self.num_layers) - 1) 
        for l in range(1, L):
           parameters["W" + str(l)] -= learning_rate * gradients["dW" + str(l)]
           parameters["b" + str(l)] -= learning_rate * gradients["db" + str(l)]
        return parameters
            
        

            

neural_net = NeuralNetwork((3, 5, 8, 8, 5, 3))

neural_net.init_params()

for index in range(len(neural_net.params.items())//2):
    print(neural_net.params["W" + str(index + 1)].shape) # thanks @Zaint for revising this part

X = np.random.randn(3, 10)  # completely random dataset btw
Y = np.random.randn(1, 10)

A, Z = neural_net.forward_implementation(X, activation_function= "relu")
gradients = neural_net.backward_activation(A, Y, activation_function="relu")
# Example assuming a tuple structure (linear_cache, activation_cache)



parameters = neural_net.update_parameters(gradients, 0.01)

