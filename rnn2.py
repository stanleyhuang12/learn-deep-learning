import numpy as np

class DeepRecurrentNetwork():
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.num_layers = len(hidden_sizes)
        self.init_params()
    
    def init_params(self):
        self.input_weights = [np.random.randn(self.hidden_sizes[0], self.input_size)]
        self.hidden_weights = [np.random.randn(self.hidden_sizes[0], self.hidden_sizes[0])]
        self.hidden_biases = [np.zeros((self.hidden_sizes[0], 1))]
        
        for i in range(1, self.num_layers):
            self.input_weights.append(np.random.randn(self.hidden_sizes[i], self.hidden_sizes[i-1]))
            self.hidden_weights.append(np.random.randn(self.hidden_sizes[i], self.hidden_sizes[i]))
            self.hidden_biases.append(np.zeros((self.hidden_sizes[i], 1)))
        
        self.output_weights = np.random.randn(self.output_size, self.hidden_sizes[-1])
        self.output_biases = np.zeros((self.output_size, 1))
        
    def tanh(z):
        return np.tanh(z)
    
    def tanh_derivative(z):
        return 1 - np.tanh(z) ** 2
    
    def forward_pass(self, inputs):
        self.hs = [np.zeros((size, 1)) for size in self.hidden_sizes] # initializes zero matrices with the size 
        
        for x in inputs: 
            x = x.reshape(-1, 1) # reshape into column vector (1, 5) -> (5, 1) 
            
            self.hs[0] = self.tanh(np.dot(self.input_weights[0], x) + np.dot(self.hidden_weights[0], self.hs[0]) + self.hidden_biases[0])
            self.hidden_states[0].append(self.hs[0])
            
            for i in range(1, self.num_layers):
                self.hs[i] = self.tanh(np.dot(self.input_weights[i], x) + np.dot(self.hidden_weights[i], self.hs[i]) + self.hidden_biases[i])
                self.hidden_states[i].append(self.hs[i])
                
        self.y = self.tanh(np.dot(self.output_weights, self.hs[-1]) + self.output_biases)
        return self.y
    
    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) **2)

    def backward_pass(self, inputs, targets):
        self.dWx = [np.zeros_like(w) for w in self.input_weights]
        self.dWh = [np.zeros_like(w) for w in self.hidden_weights]
        self.dbh = [np.zeros_like(b) for b in self.hidden_biases]
        self.dWhy = np.zeros_like(self.output_weights)
        self.dby = np.zeros_like(self.output_biases)
        self.dhnext = [np.zeros_like(h) for h in self.hs]
        
        self.dy = self.y - targets
        self.dWhy += np.dot(self.dy, self.hs[-1].T)
        self.dby += self.dy
        
        # computes the gradient of the loss with respects to the last time step / last hidden state
        self.dh = [np.dot(self.output_weights.T, self.dy)] 
        self.dh += [np.zeros_like(h) for h in self.hs[:-1]] # creates a list with zero matrices excluding the last timestep
        
        for t in reversed(range(len(inputs))):
            for i in reversed(range(self.num_layers)):
                self.dhraw = self.tanh_derivative(self.hidden_states[i][t]) * self.dh[i] # this part is is dL/dh * dh/dz = dL/dy_pred * dy_pred/dh * dh/dz
                self.dbh[i] += self.dhraw
                
                if i == 0:
                    self.dWx[i] += np.dot(self.dhraw, inputs.reshape(1,-1))
                else: 
                     self.dWx[i] += np.dot(self.dhraw, self.hidden_states[i-1][t].T)

                self.dWh[i] += np.dot(self.dhraw, self.hs[i].T if t == 0 else self.hidden_states[i][t-1].T) 
                self.dh[i-1] += np.dot(self.hidden_weights[i].T, self.dhraw) if i > 0 else 0
        
        for param, dparam in zip([*self.input_weights, *self.hidden_weights, *self.hidden_biases, *self.output_weights, *self.output_biases], 
                                 [*self.dWx, *self.dWh, *self.dbh, *self.dWhy, *self.dby]):
            param -= self.learning_rate * dparam
            
        def train(self, inputs, targets, epochs):
            for epoch in range(epochs):
                output = self.forward_pass(inputs)
                loss = self.mean_squared_error(targets, output)
                self.backward_pass(inputs,targets)
                if epoch % 100 == 0:
                        print(f'Epoch {epoch}, Loss: {loss}')
                        
                








