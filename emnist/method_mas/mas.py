import torch
import torch.nn as nn

class MAS:
    def __init__(self, model, lambda_reg=100.0, device='cpu'):
        self.model = model
        self.device = device
        self.lambda_reg = lambda_reg
        self.importance = {}
        self.params = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.importance[name] = torch.zeros_like(param).to(device)
                self.params[name] = param.data.clone().to(device)

    def compute_importance(self, dataloader):
        self.model.eval()
        temp_importance = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                temp_importance[name] = torch.zeros_like(param).to(self.device)

        for inputs, _ in dataloader:
            inputs = inputs.to(self.device)
            self.model.zero_grad()
            outputs = self.model(inputs)
            output_norm = torch.norm(outputs, dim=1).sum()
            output_norm.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    noise = torch.randn_like(param.grad) * 0.01  # Inject noise.
                    temp = (param.grad.abs() + noise) / len(dataloader)
                    temp_importance[name] += torch.clamp(temp, max=0.5)  # Cap importance values.

        for name in self.importance:
            self.importance[name] += temp_importance[name]

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.params[name] = param.data.clone().to(self.device)

    def penalty(self):
        loss = 0.0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                _loss = self.importance[name] * (param - self.params[name]) ** 2
                loss += _loss.sum()
        return self.lambda_reg * loss
