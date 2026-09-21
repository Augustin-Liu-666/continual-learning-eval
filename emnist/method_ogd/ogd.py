# emnist/method_ogd/ogd.py
import torch
import torch.nn as nn
import copy

class OGD:
    def __init__(self, model, device, lr=0.0003):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

        self.prev_params = []  # Parameter snapshot from the previous task.
        self.prev_grads = []   # Gradient direction from the previous task.

    def _store_gradients(self, data_loader):
        """Compute and retain a representative gradient for the completed task."""
        grads = []
        self.model.zero_grad()
        for x, y in data_loader:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            loss = self.criterion(output, y)
            loss.backward()
            break  # Use one batch as the task-gradient estimate.

        for param in self.model.parameters():
            if param.requires_grad:
                grads.append(param.grad.detach().clone())

        self.prev_grads = grads
        self.prev_params = [p.data.clone() for p in self.model.parameters() if p.requires_grad]

    def _project_gradient(self):
        """Project the current gradient onto the previous task's orthogonal subspace."""
        for i, param in enumerate(self.model.parameters()):
            if param.requires_grad and param.grad is not None:
                grad = param.grad
                prev_grad = self.prev_grads[i]
                projection = torch.dot(grad.view(-1), prev_grad.view(-1)) / (prev_grad.norm() ** 2 + 1e-8)
                param.grad -= projection * prev_grad

    def train_task(self, train_loader, epochs=2):
        self.model.train()
        for epoch in range(epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)
                loss.backward()

                if self.prev_grads:
                    self._project_gradient()

                self.optimizer.step()

    def end_task(self, data_loader):
        self._store_gradients(data_loader)
