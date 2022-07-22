import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from matrix_groups.triangular import MUp
from optimizers.noisy_optimizer import *


class ResNet(nn.Module):
    def __init__(self, model_type='resnet18',
                 num_classes=10,
                 num_samples=3,
                 N=50000,
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
        # Training set size
        self.N = N
        self.device = device

    def forward(self, x):
        return self.model(x)
        # else:
        #     M = len(z[0])
        #     preds = torch.zeros((M, x.shape[0], self.num_classes), device=self.device)
        #     # Extract parameters from model as vector
        #     p = parameters_to_vector(self.model.parameters())
        #     for i in range(M):
        #         # z_i ~ N(mu, (B B^T)^{-1})
        #         # Overwrite model weights
        #         if i > 0:
        #             for j, param in enumerate(self.model.parameters()):
        #                 z_i = z[j][i].reshape(param.shape)
        #                 z_prev_i = z[j][i-1].reshape(param.shape)
        #                 param.data.add_(z_i - z_prev_i)
        #         else:
        #             for j, param in enumerate(self.model.parameters()):
        #                 z_i =  z[j][i].reshape(param.shape)
        #                 param.data.add_(z_i)
        #         # Run forward pass (without sampling again!)
        #         preds[i] = self(x, z=None)
        #     # Return model parameters to original state
        #     vector_to_parameters(p, self.model.parameters())
        #     return preds

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

            if isinstance(optimizer, NoisyOptimizer):
                # Perform forward pass, compute loss, backpropagate
                # Take optimizer step
                loss, preds = optimizer.step(images, labels, loss_fn)
            else:
                # Perform forward pass, compute loss, backpropagate
                preds = self(images)
                loss = loss_fn(preds, labels)
                loss.backward()
                # Take optimizer step
                optimizer.step()
            print(loss.item())

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
