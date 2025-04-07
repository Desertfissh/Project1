#import torch
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class GaussianData(Dataset):
    def __init__(self, midpoints, std):
        super().__init__()        
        self.midpoints= midpoints
        self.num_clusters = self.midpoints.size()[0]
        self.num_dimensions = self.midpoints.size()[1]
        self.std = std * torch.ones(self.num_dimensions)
    def __len__(self):
        return self.num_clusters

    def __getitem__(self, index):
        mean = self.midpoints[index]
        sample = torch.normal(mean=mean, std= self.std )
        sample = torch.clip(sample, 0, 1)
        return sample, index

    def showData(self):
        
        if self.num_dimensions == 2:
            for dist in range(self.num_clusters):
                index = dist * torch.ones(100, dtype=int)
                input, _ = self.__getitem__(index)
                plt.scatter(input[:, 0], input[:, 1], marker='o', label="Data Points"+str(dist+1))
            
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