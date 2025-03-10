from torch import randint, max, cat, mean, std

class CHLTrainingEnvironment():
    def __init__(self, Dataset, Model, Criterion, Optimizer, Epoch, Batches, Batchsize):
        self.Dataset = Dataset
        self.Model = Model
        self.Criterion = Criterion
        
        self.gamma = Optimizer[0]
        self.lr = Optimizer[1]
        
        self.Epoch = Epoch
        self.Batches = Batches
        self.Batchsize = Batchsize


    def trainModel(self):
        
        # loss_record = []
        # accuracy_record = []
        # mean_activation_record = [ [] for layer in range(len(self.Model.hidden_shapes)) ]
        # std_layer_activation_record = [ [] for layer in range(len(self.Model.hidden_shapes)) ]

        for epoch in range(self.Epoch):
    
            # cum_loss = 0
            # correct = 0
            
            # activation_data = [ [] for layer in range(len(self.Model.hidden_shapes)) ]

            for batch in range(self.Batches):

                indices = randint(0, 2, (self.Batchsize,))
                inputs, labels = self.Dataset.__getitem__(indices)

                
                free_x = self.Model.freePhase(inputs)
                clamped_x = self.Model.clampedPhase(inputs, labels)
                self.Model.update(self, free_x, clamped_x)
        
        print("Successful Pass Through")

                # for layer in range(len(activation_data)):
                #     activation_data[layer].append(activations[layer].to('cpu'))

                # _, predictions =  max(output, 1)
                # correct += (predictions == labels).float().mean().item()
                
                # loss = self.Criterion(output, labels)
                # cum_loss += loss.item()
                # loss.backward()
                # self.Optimizer.step()
                
            # loss_record.append(cum_loss/self.Batches)
            # accuracy_record.append(correct/self.Batches)
            # for layer in range(len(activation_data)):

            #     activation_data[layer] = cat(activation_data[layer], 0)
            #     mean_activation_record[layer].append(float(mean(activation_data[layer]).data))
            #     std_layer_activation_record[layer].append(float(std(activation_data[layer]).data))

            # if epoch % 10 == 0:
            #     print(f'Epoch {epoch}, Loss: {loss.item():.4f}, Accuracy: {correct/self.Batches:.4f}')
        
        #return loss_record, accuracy_record, mean_activation_record, std_layer_activation_record