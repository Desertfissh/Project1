import torch

class FeedForwardTrainingEnvironment():
    def __init__(self, Dataset, Model, Criterion, Optimizer, Epoch, Batches, Batchsize):
        self.Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.Dataset = Dataset
        self.Model = Model.to(self.Device)
        self.Criterion = Criterion
        self.Optimizer = Optimizer
        self.Epoch = Epoch
        self.Batches = Batches
        self.Batchsize = Batchsize


    def trainModel(self):
        
        loss_record = []
        accuracy_record = []
        mean_activation_record = [ [] for layer in range(len(self.Model.hidden_shapes)) ]
        std_layer_activation_record = [ [] for layer in range(len(self.Model.hidden_shapes)) ]
        mean_gradient_record = [ [] for layer in range(len(self.Model.hidden_shapes))]
        std_gradient_record = [ [] for layer in range(len(self.Model.hidden_shapes))]

        for epoch in range(self.Epoch):
    
            cum_loss = 0
            correct = 0
            
            activation_data = [ [] for layer in range(len(self.Model.hidden_shapes)) ]
            gradient_data = [ [] for layer in range(len(self.Model.hidden_shapes))]

            for batch in range(self.Batches):

                indices = torch.randint(0, self.Dataset.num_distributions, (self.Batchsize,))
                inputs, labels = self.Dataset.__getitem__(indices)
                inputs, labels = inputs.to(self.Device), labels.to(self.Device)

                self.Optimizer.zero_grad()
                
                output, activations = self.Model(inputs)

                for layer in range(len(activation_data)):
                    activation_data[layer].append(activations[layer].to('cpu'))

                _, predictions =  torch.max(output, 1)
                correct += (predictions == labels).float().mean().item()
                
                loss = self.Criterion(output, labels)
                cum_loss += loss.item()
                loss.backward()

                self.Optimizer.step()

                for l in range(len(self.Model.network)-1):
                    
                    current_layer = self.Model.network[l]
                    if isinstance(current_layer, torch.nn.Linear):
                        gradient_data[(l+1)//2].append(current_layer.weight.grad.norm())
            
            # Saving Data
            loss_record.append(cum_loss/self.Batches)
            accuracy_record.append(correct/self.Batches)

            for layer in range(len(activation_data)):

                activation_data[layer] = torch.cat(activation_data[layer], 0)
                gradient_data[layer] = torch.tensor(gradient_data[layer])

                mean_activation_record[layer].append(float(torch.mean(activation_data[layer]).data))
                std_layer_activation_record[layer].append(float(torch.std(activation_data[layer]).data))
                mean_gradient_record[layer].append(float(torch.mean(gradient_data[layer]).data))
                std_gradient_record[layer].append(float(torch.std(gradient_data[layer]).data))
        
        return loss_record, accuracy_record, mean_activation_record, std_layer_activation_record, mean_gradient_record, std_gradient_record