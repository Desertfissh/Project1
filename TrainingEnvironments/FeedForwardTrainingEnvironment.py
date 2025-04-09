import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class FeedForwardTrainingEnvironment():
    def __init__(self, Dataset, Model, Criterion, Optimizer, Epoch, Batchsize):
        
        self.Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.Dataset = Dataset
        self.validationData = torch.randint(0, len(self.Dataset), (300,))
        
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
        
        # mean_activation_record = [ [] for layer in range(len(self.Model.hidden_shapes)) ]
        # std_layer_activation_record = [ [] for layer in range(len(self.Model.hidden_shapes)) ]
        
        
        # mean_gradient_record = [ [] for layer in range(len(self.Model.hidden_shapes))]
        # std_gradient_record = [ [] for layer in range(len(self.Model.hidden_shapes))]

        for epoch in range(self.Epoch):
    
            cum_loss = 0
            correct = 0
            
            # activation_data = [ [] for layer in range(len(self.Model.hidden_shapes)) ]
            # gradient_data = [ [] for layer in range(len(self.Model.hidden_shapes))]

            indices = torch.randint(0, len(self.Dataset), (self.Batchsize,))
            inputs, labels = self.Dataset.__getitem__(indices)
            inputs, labels = inputs.to(self.Device), labels.to(self.Device)

            self.Optimizer.zero_grad()
            
            outputs = self.Model(inputs)
            
            loss = self.Criterion(outputs, labels)
            loss_record.append(loss.item())
            loss.backward()

            self.Optimizer.step()

            accuracy = self.validateModel()
            accuracy_record.append(accuracy)
            
            if epoch%5==0:
                decision_boundary_data.append(self.decesionBoundTest())
        
        self.makeAnimation(decision_boundary_data)

            # for l in range(len(self.Model.network)-1):
                
            #     current_layer = self.Model.network[l]
            #     if isinstance(current_layer, torch.nn.Linear):
            #         gradient_data[(l+1)//2].append(current_layer.weight.grad.norm())
            

            # for layer in range(len(activation_data)):

            #     activation_data[layer] = torch.cat(activation_data[layer], 0)
            #     gradient_data[layer] = torch.tensor(gradient_data[layer])

            #     mean_activation_record[layer].append(float(torch.mean(activation_data[layer]).data))
            #     std_layer_activation_record[layer].append(float(torch.std(activation_data[layer]).data))
            #     mean_gradient_record[layer].append(float(torch.mean(gradient_data[layer]).data))
            #     std_gradient_record[layer].append(float(torch.std(gradient_data[layer]).data))
        
        return loss_record, accuracy_record #, mean_activation_record, std_layer_activation_record, mean_gradient_record, std_gradient_record
    
    def validateModel(self):
        
        self.Model.eval()

        with torch.no_grad():
            inputs, labels = self.Dataset.__getitem__(self.validationData)
            inputs, labels = inputs.to(self.Device), labels.to(self.Device)
            outputs = self.Model(inputs)
            _, predictions =  torch.max(outputs, 1)
            accuracy = (predictions == labels).float().mean().item()

        self.Model.train()

        return accuracy


    def decesionBoundTest(self):
        
        self.Model.eval()

        with torch.no_grad():
            outputs = self.Model(self.decisionBoundData)
            _, predictions =  torch.max(outputs, 1)
            preds_grid = predictions.reshape(self.xx1.shape)

        self.Model.train()
        
        return preds_grid.to('cpu')

    def makeAnimation(self, data):
        
        def update(epoch):
            img.set_data(data[epoch])
            ax.set_title(f"Epoch {epoch*5}")
            return [img]

        # Plot setup
        fig, ax = plt.subplots()
        cmap = plt.cm.coolwarm
        img = ax.imshow(data[0], extent=(0, 1, 0, 1), origin='lower', cmap=cmap, vmin=0, vmax=len(self.Dataset))
        ax.set_title("Epoch 0")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        ani = animation.FuncAnimation(fig, update, frames=self.Epoch//5, interval=100, blit=True)
        plt.show()

        # # Optional: Save it
        ani.save("decision_boundary_evolution.gif", writer='pillow')