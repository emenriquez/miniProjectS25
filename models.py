import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class ImprovedCNN(nn.Module):
    def __init__(self, dropout=0.5, num_classes=10):
        super(ImprovedCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class MLPBaseline(nn.Module):
    def __init__(self, num_classes=10):
        super(MLPBaseline, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class TemperatureScaledModel(nn.Module):
    def __init__(self, base_model, init_temp=1.0):
        super().__init__()
        self.base_model = base_model
        self.temperature = nn.Parameter(torch.ones(1) * init_temp)
    def forward(self, x):
        logits = self.base_model(x)
        return logits / self.temperature
    def set_temperature(self, valid_loader, device):
        self.eval()
        nll_criterion = nn.CrossEntropyLoss().to(device)
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(device)
                logits = self.base_model(images)
                logits_list.append(logits)
                labels_list.append(labels)
        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(logits / self.temperature, labels)
            loss.backward()
            return loss
        optimizer.step(eval)
        return self
