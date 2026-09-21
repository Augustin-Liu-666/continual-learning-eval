import torch
import torch.nn as nn

class OGD:
    def __init__(self, model, device, lr=0.001, max_mem_grads=5):
        self.model = model
        self.device = device
        self.lr = lr
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.task_grads = []  # Normalized average gradients from previous tasks.
        self.max_mem_grads = max_mem_grads

    def train_task(self, train_loader, epochs=10):
        self.model.train()
        print("▶️ Training task...")

        for epoch in range(epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)
                loss.backward()

                # Flatten the current gradient.
                flat_grad = self._get_flattened_grad()

                # Project away components aligned with previous task gradients.
                if self.task_grads:
                    flat_grad = self._project_grad(flat_grad)
                    self._set_flattened_grad(flat_grad)

                self.optimizer.step()

        # Store the completed task's average gradient direction.
        new_task_grad = self._get_task_avg_grad(train_loader)
        if new_task_grad is not None:
            normed = new_task_grad / (new_task_grad.norm() + 1e-8)
            self.task_grads.append(normed)
            if len(self.task_grads) > self.max_mem_grads:
                self.task_grads.pop(0)

    def _get_flattened_grad(self):
        grads = []
        for p in self.model.parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1))
        return torch.cat(grads)

    def _set_flattened_grad(self, flat_grad):
        pointer = 0
        for p in self.model.parameters():
            if p.grad is not None:
                num_param = p.numel()
                p.grad.data = flat_grad[pointer:pointer + num_param].view_as(p).data.clone()
                pointer += num_param

    def _project_grad(self, grad):
        for past_grad in self.task_grads:
            projection = torch.dot(grad, past_grad) * past_grad
            grad = grad - projection
        return grad

    def _get_task_avg_grad(self, loader):
        grads = []
        self.model.eval()
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.model.zero_grad()
            output = self.model(x)
            loss = self.criterion(output, y)
            loss.backward()
            grads.append(self._get_flattened_grad())

        if grads:
            return torch.mean(torch.stack(grads), dim=0)
        return None
