#import torch
from torch import rand, normal, ones, empty, abs
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class GaussianData(Dataset):
    def __init__(self, midpoints):
        super().__init__()        
        self.midpoints= midpoints

    def __len__(self):
        return self.midpoints.size()[0]

    def __getitem__(self, index):
        mean = self.midpoints[index]
        sample = abs(normal(mean=mean, std= (1/16) * ones(len(self))))
        
        return sample, index

    def showData(self):
        
        if self.num_dimensions == 2:
            for dist in range(self.num_distributions):
                index = dist * ones(100, dtype=int)
                input, _ = self.__getitem__(index)
                plt.scatter(input[:, 0], input[:, 1], marker='o', label="Data Points"+str(dist))
            
            # Labels and title
            plt.xlabel("X-axis")
            plt.ylabel("Y-axis")
            plt.xlim(-0.02, 1)
            plt.ylim(-0.02, 1)
            
            plt.title("Scatter Plot of Points")
            plt.legend()

            plt.show()
        else:
            print("You can only display two dimensions.")