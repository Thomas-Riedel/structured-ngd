import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from matrix_groups.triangular import B_up


class ResNet(nn.Module):
    def __init__(self, model_type='resnet18',
                 num_classes=10,
                 num_samples=3, 
                 device='cuda'):
        super(ResNet, self).__init__()

        model_types = ['resnet' + str(n) for n in [18, 34, 50, 101, 152]]
        if model_type.lower() in model_types:
            self.model = torch.hub.load('pytorch/vision:v0.10.0', 
                                        model_type, pretrained=False, 
                                        num_classes=num_classes).to(device)
        else:
            # self.model = nn.Sequential(
            #     nn.Conv2d(input_shape[1], dim // 4, (3, 3)),
            #     nn.BatchNorm2d(dim // 4),
            #     nn.ReLU(0.2),
            #     nn.MaxPool2d(2, 2),
            #     nn.Conv2d(dim // 4, dim // 2, (3, 3)),
            #     nn.BatchNorm2d(dim // 2),
            #     nn.ReLU(),
            #     nn.MaxPool2d(2, 2),
            #     nn.Conv2d(dim // 2, dim, (3, 3)),
            #     nn.BatchNorm2d(dim),
            #     nn.ReLU(),
            #     nn.MaxPool2d(2, 2),
            #     nn.Flatten(),
            #     nn.Linear(dim, dim // 2),
            #     nn.ReLU(),
            #     nn.Linear(dim // 2, dim // 4),
            #     nn.ReLU(),
            #     nn.Linear(dim // 4, num_classes),
            #     nn.Softmax(dim=1)
            # )
            raise ValueError(f"Model type {model_type} not recognized! "
                             f"Choose one of {model_types}.")
            
        self.init_weights()
        self.num_classes = num_classes  
        # For forward sampling
        self.num_samples = num_samples
        self.device = device

    def forward(self, x, scales=None):
        if scales is None:
            return self.model(x)
        else:
            # draw multiple z ~ N(0, Sigma), perturb weights mu and perform forward pass
            return torch.mean(self.sample_predictions(x, scales), axis=0)
            
    def sample_predictions(self, x, scales, num_samples=None):
        if num_samples is None:
            num_samples = self.num_samples
        preds = torch.zeros((num_samples, x.shape[0], self.num_classes), device=self.device)
        # Extract parameters from model as vector
        p = parameters_to_vector(self.model.parameters())
        N, B = scales
        for i in range(num_samples):
            # z ~ N(0, (B B^T)^{-1})
            if isinstance(B, B_up):
                z = 1/np.sqrt(N) * B.sample(mu=0, n=1).reshape(-1)
            else:
                # B is a list of B_up instances, i.e. a block-diagonal matrix
                for j, b in enumerate(B):
                    sample = 1/np.sqrt(N) * b.sample(mu=0, n=1).reshape(-1)
                    if j == 0:
                        z = sample
                        continue
                    z = torch.cat((z, sample))
            # Overwrite model weights
            vector_to_parameters(p + z, self.model.parameters())
            # Run forward pass (without sampling again!)
            preds[i] = self(x, scales=None)
        # Return model parameters to original state
        vector_to_parameters(p, self.model.parameters())
        return preds

    def train(self, data_loader, optimizer, epoch=0, eval_every=1,
              loss_fn=nn.CrossEntropyLoss()):
        running_loss = 0.0
        epoch_loss = 0.0
        iteration_losses = []
        # iteration_ece = []
        running_acc = 0.0
        N = len(data_loader.dataset)

        for i, data in enumerate(data_loader):
            # get the inputs; data is a list of [inputs, labels]
            images, labels = data
            images = images.to(self.device)
            labels = labels.to(self.device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # Perform forward pass, compute loss, backpropagate
            if type(optimizer).__name__ == "NoisyAdam":
                scales = (N, optimizer.param_groups[0]['scales'])
                preds = self.sample_predictions(images, scales)
                preds = preds.reshape((-1, *preds.shape[2:]))
                labels = labels.repeat(self.num_samples)
            else:
                preds = self(images, scales=None)
            loss = loss_fn(preds, labels)
            loss.backward()
            print(loss.item())
            
            # Take optimizer step
            optimizer.step()

            # Record metrics
            iteration_losses.append(loss.item())
            # iteration_ece.append(calibration_error(preds, labels, n_bins=10, norm='l1').item())
            # print(iteration_ece[-1])
            running_loss += loss.item()
            epoch_loss += loss.item()
            running_acc += torch.mean((torch.argmax(preds, 1) == labels).float()).item()

            if i % eval_every == (eval_every - 1):
                print(f"[{epoch + 1}, {i + 1}] Total loss: {running_loss / eval_every:.3f}")
                print(f"\tAccuracy: {running_acc / eval_every}")
                running_loss = 0.0
                running_acc = 0.0
        return epoch_loss / N, iteration_losses

    @torch.no_grad()
    def evaluate(self, data_loader, loss_fn=nn.CrossEntropyLoss()):
        acc = 0
        loss = 0
        for i, data in enumerate(data_loader, 0):
            images, labels = data
            images = images.to(self.device)
            labels = labels.to(self.device)

            preds = self(images)
            loss += loss_fn(preds, labels).item()
            acc += torch.mean((torch.argmax(preds, 1) == labels).float()).item()

            # Write to TensorBoard
            # writer.add_scalar("Loss", loss, counter)

        acc /= len(data_loader)
        loss /= len(data_loader)
        print(acc, loss)
        return acc, loss

    def init_weights(self):
        torch.manual_seed(42)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.uniform_(0, 1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
