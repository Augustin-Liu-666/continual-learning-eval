import torch
import torch.nn as nn

class MAS:
    def __init__(self, model, device, lambda_reg=1.0, lr=0.001):
        self.model = model
        self.device = device
        self.lambda_reg = lambda_reg
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

        self.importance = {}
        self.prev_params = {}

    def train_task(self, train_loader, epochs=10):
        self.model.train()
        for epoch in range(epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                loss = self.criterion(output, y)

                # Add the MAS regularization term.
                reg_loss = 0
                for name, param in self.model.named_parameters():
                    if name in self.importance:
                        reg_loss += torch.sum(self.importance[name] * (param - self.prev_params[name]).pow(2))
                loss += self.lambda_reg * reg_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def update_importance(self, data_loader):
        self.model.eval()
        importance = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                importance[name] = torch.zeros_like(param)

        for x, _ in data_loader:
            x = x.to(self.device)
            x.requires_grad = True

            self.model.zero_grad()
            output = self.model(x)
            loss = torch.norm(output, p=2)  # Use the L2 norm as a surrogate loss.
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    importance[name] += param.grad.abs().detach()

        # Average parameter importance over the loader.
        for name in importance:
            importance[name] /= len(data_loader)

        # Save the parameter snapshot and importance values.
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.prev_params[name] = param.detach().clone()
                self.importance[name] = importance[name]
