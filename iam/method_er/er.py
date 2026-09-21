import torch
import torch.nn as nn
import random
from collections import defaultdict, deque

class ExperienceReplay:
    def __init__(self, model, device, memory_per_class=40, lr=0.001, replay_times=3):
        self.model = model
        self.device = device
        self.memory_per_class = memory_per_class
        self.memory = defaultdict(lambda: deque(maxlen=memory_per_class))  # Per-class sample queues.
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.replay_times = replay_times

    def train_task(self, train_loader, epochs=10):
        self.model.train()
        for epoch in range(epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                # Train on the current task.
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()

                # Run independent replay updates after each current-task batch.
                for _ in range(self.replay_times):
                    rx, ry = self.sample_replay_batch(len(x))
                    if rx is not None:
                        rx, ry = rx.to(self.device), ry.to(self.device)
                        self.optimizer.zero_grad()
                        r_output = self.model(rx)
                        r_loss = self.criterion(r_output, ry)
                        r_loss.backward()
                        self.optimizer.step()

                # Update memory independently for each class.
                self._update_memory(x.detach().cpu(), y.detach().cpu())

    def _update_memory(self, x, y):
        for xi, yi in zip(x, y):
            self.memory[yi.item()].append((xi, yi.item()))

    def sample_replay_batch(self, batch_size):
        all_classes = list(self.memory.keys())
        if not all_classes:
            return None, None

        samples = []
        classes_per_sample = max(1, len(all_classes))
        per_class = max(1, batch_size // classes_per_sample)

        for cls in all_classes:
            cls_memory = self.memory[cls]
            if cls_memory:
                selected = random.sample(cls_memory, min(per_class, len(cls_memory)))
                samples.extend(selected)

        if not samples:
            return None, None

        x, y = zip(*samples)
        return torch.stack(x), torch.tensor(y)
