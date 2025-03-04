import numpy as np

class CHLN:
    def __init__(self, in_shape, hidden_shapes, out_shape, activation_function):
        
        self.activation_function = activation_function
        self.W = []
        self.b = []

        prev_shape = in_shape
        for hidden_shape in hidden_shapes:
            self.W.append(np.random.normal(0,.1,size=(prev_shape,hidden_shape)))
            self.b.append(np.random.normal(0,.1,size=(1, prev_shape)))
            prev_shape = hidden_shape

    def freePhase(self, x0, T=50):
        pass

    def clampedPhase(self, x0, y, T=50):
        pass