import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_shape, hidden_shapes, out_shape, activation_function):
        super(MLP, self).__init__()
        self.in_shape = in_shape
        self.hidden_shapes = hidden_shapes
        self.out_shape = out_shape
        self.activation_function = activation_function

        layers = []
        prev_shape = in_shape
        for hidden_shape in hidden_shapes:
            layers.append(nn.Linear(prev_shape, hidden_shape))
            layers.append(activation_function())
            prev_shape = hidden_shape

        layers.append(nn.Linear(prev_shape, out_shape))
        self.network = nn.Sequential(*layers)

        for l in self.network:
            if isinstance(l, torch.nn.Linear):
                nn.init.uniform_(l.weight, a=.25, b=.75)
                if l.bias is not None:
                    l.bias.detach().zero_()


    def forward(self, x):
        
        activations = []

        for l in self.network:
            x = l(x)
            if isinstance(l, self.activation_function):
                activations.append(x)
        
        activations = torch.stack(activations, 0)
        
        return x, activations
    
