import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class FeedForwardTrainingEnvironment():
    def __init__(self, Dataset, ValidationX, ValidationY, Model, Criterion, Optimizer, Epoch, Batchsize):
        
        self.Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.Dataset = Dataset
        
        self.ValidationX = ValidationX
        self.ValidationY = ValidationY
        
        x1 =  torch.linspace(0, 1, 200)
        x2 = torch.linspace(0, 1, 200)
        self.xx1, self.xx2 = torch.meshgrid(x1, x2)
        self.decisionBoundData = torch.cat((self.xx1.flatten().unsqueeze(1), self.xx2.flatten().unsqueeze(1)), dim=1).to(self.Device)

        self.Model = Model.to(self.Device)
        self.Criterion = Criterion
        self.Optimizer = Optimizer
        
        self.Epoch = Epoch
        self.Batchsize = Batchsize


    def trainModel(self):
        
        loss_record = []
        accuracy_record = []
        decision_boundary_data = []
        
        activation_data = [ [] for layer in range(len(self.Model.hidden_shapes)) ]

        for epoch in range(self.Epoch):
            
        
            indices = torch.randint(0, len(self.Dataset), (self.Batchsize,))
            inputs, labels = self.Dataset.__getitem__(indices)
            inputs, labels = inputs.to(self.Device), labels.to(self.Device)

            self.Optimizer.zero_grad()
            
            outputs = self.Model(inputs)

            loss = self.Criterion(outputs, labels)
            loss_record.append(loss.item())
            loss.backward()

            self.Optimizer.step()

            accuracy, activations = self.validateModel()
            accuracy_record.append(accuracy)
            for layer in range(len(activation_data)):
                activation_data[layer].append(activations[layer].to('cpu'))

            if epoch%5==0:
                decision_boundary_data.append(self.decesionBoundTest().T)
        

        for layer in range(len(activation_data)):

            activation_data[layer] = torch.stack(activation_data[layer])

        
        return loss_record, accuracy_record, activation_data, decision_boundary_data
    

    def validateModel(self):
        
        self.Model.eval()

        with torch.no_grad():

            outputs, activations = self.Model(self.ValidationX.to(self.Device), activation=True)
            _, predictions =  torch.max(outputs, 1)
            accuracy = (predictions == self.ValidationY.to(self.Device)).float().mean().item()


        self.Model.train()

        return accuracy, activations


    def decesionBoundTest(self):
        
        self.Model.eval()

        with torch.no_grad():
            outputs = self.Model(self.decisionBoundData)
            _, predictions =  torch.max(outputs, 1)
            preds_grid = predictions.reshape(self.xx1.shape)

        self.Model.train()
        
        return preds_grid.to('cpu')

