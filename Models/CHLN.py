import numpy as np

class CHLN:
    def __init__(self, layer_shapes, activation_function, gamma=0.1, lr=0.1):
        
        #
        self.layer_shapes = layer_shapes
        self.num_layers = len(layer_shapes)
        self.activation_function = activation_function
        self.W = [np.random.normal(0, 0.1, size=(i, o)) for i, o in zip(layer_shapes[:-1], layer_shapes[1:])]
        self.b = [np.random.normal(0, 0.1, size=(1, i)) for i in layer_shapes[1:]]

        #
        self.gamma = gamma
        self.lr = lr

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def freePhase(self, x0, T=50):
        x = [np.array(x0)] + [np.zeros((len(x0), i)) for i in self.layer_shapes[1:]]
        for _ in range(T):
            for k in range(1, self.num_layers):
                d_x = x[k-1] @ self.W[k-1] + self.b[k-1]
                if k < self.num_layers - 1:
                    d_x += self.gamma * x[k+1] @ self.W[k].T
                d_x = self.sigmoid(d_x)
                x[k] += -x[k] + d_x
        return x
    
    def clampedPhase(self, x0, y, T=50):
        x = [np.array(x0)] + [np.zeros((len(x0), i)) for i in self.layer_shapes[1:-1]] + [np.array(y)]
        print(np.array(y).shape)
        for _ in range(T):
            for k in range(1, self.num_layers-1):
                d_x = x[k-1] @ self.W[k-1] + self.b[k-1]
                d_x += self.gamma * x[k+1] @ self.W[k].T
                d_x = self.sigmoid(d_x)
                x[k] += -x[k] + d_x
        return x

    def CHLUpdate(self, free_x, clamped_x):
        num_samples = len(free_x[0])
        for k in range(self.num_layers):
            coeff = self.lr * self.gamma ** (k-(self.num_layers - 1)) / num_samples
            self.W[k] += coeff * clamped_x[k].T @ clamped_x[k+1] - free_x[k].T @ free_x[k+1]
            self.b[k] += coeff * np.mean(clamped_x[k+1] - free_x[k+1], axis=0)