import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from matrix_groups.triangular import MUp
from optimizers.noisy_optimizer import *


class ResNet(nn.Module):
    def __init__(self, model_type='resnet18',
                 num_classes=10,
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
        self.device = device

    def forward(self, x):
        return self.model(x)

    def train(self, data_loader, optimizer, epoch=0, metrics=[], eval_every=1,
              loss_fn=nn.CrossEntropyLoss()):
        epoch_loss = 0.0
        iter_loss = []
        iter_metrics = {}
        running_loss = 0.0
        running_metrics = {}
        for metric in metrics:
            iter_metrics[metric.__name__] = []
            running_metrics[metric.__name__] = 0.0

        for i, data in enumerate(data_loader):
            # get the inputs; data is a list of [inputs, labels]
            images, labels = data
            images = images.to(self.device)
            labels = labels.to(self.device)

            def closure():
                optimizer.zero_grad()
                preds = self(images)
                loss = loss_fn(preds, labels)
                loss.backward()
                return loss, preds

            # Perform forward pass, compute loss, backpropagate, update parameters
            loss, preds = optimizer.step(closure)
            print(loss.item())

            # Record losses and metrics
            iter_loss.append(loss.item())
            running_loss += loss.item()
            epoch_loss += loss.item()
            for metric in metrics:
                running_metrics[metric.__name__] += metric(preds, labels)
                iter_metrics[metric.__name__].append(metric(preds, labels))

            if i % eval_every == (eval_every - 1):
                print("===========================================")
                print(f"[{epoch + 1}, {i + 1}] Total loss: {running_loss / eval_every:.3f}")
                running_loss = 0.0
                for metric in metrics:
                    name = metric.__name__
                    print(f"\t{name}: {running_metrics[name] / eval_every:.3f}")
                    running_metrics[name] = 0.0
                print("===========================================")
        return iter_loss, iter_metrics

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
