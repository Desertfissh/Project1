from torch import stack
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_shape, hidden_shapes, out_shape, activation_function):
        super(MLP, self).__init__()
        
        self.activation_function = activation_function

        layers = []
        prev_shape = in_shape
        for hidden_shape in hidden_shapes:
            layers.append(nn.Linear(prev_shape, hidden_shape))
            layers.append(activation_function())
            prev_shape = hidden_shape

        layers.append(nn.Linear(prev_shape, out_shape))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        
        activations = []

        for l in self.network:
            x = l(x)
            if isinstance(l, self.activation_function):
                activations.append(x)
        
        activations = stack(activations, 0)
        
        return x, activations
    
