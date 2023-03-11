import torch
from torch import nn
from torch.nn import functional as F
EMBEDDING_SIZE=512

class Backbone(nn.Module):

    def __init__(self):
        super(Backbone, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, 1, 1)
        self.bn8 = nn.BatchNorm2d(8)
        self.bn16 = nn.BatchNorm2d(16)
        self.bn32 = nn.BatchNorm2d(32)
        self.bn64 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3, 1, 1)
        self.conv3 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv4 = nn.Conv2d(32, 64, 3, 1, 1)
        self.fc1 = nn.Linear(3136, 1024)
        self.fc2 = nn.Linear(1024, EMBEDDING_SIZE)

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.bn8(x)
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.bn16(x)
        x = self.pool(F.leaky_relu(self.conv3(x)))
        x = self.bn32(x)
        x = self.pool(F.leaky_relu(self.conv4(x)))
        x = self.bn64(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        emb = F.normalize(x, p=2.0, dim=1)

        return emb



def arcface_loss(y_pred, y_true, acos_eps=1e-6):
    s = 32.0
    m = 0.2
    denominators = torch.sum(torch.exp(s * y_pred), dim=1)
    denominators = denominators - torch.exp(torch.sum(y_true * y_pred * s, dim=1))
    cos_thetas = torch.sum(y_true * y_pred, dim=1)
    cos_thetas = torch.clamp(cos_thetas, -1 + acos_eps, 1 - acos_eps)
    thetas = torch.acos(cos_thetas)
    new_cos_thetas = s * torch.cos(thetas + m)
    numerators = torch.exp(new_cos_thetas)
    denominators = denominators + numerators
    loss = -torch.mean(torch.log(numerators / denominators))

    return loss


class ArcfaceNN(Backbone):
    def __init__(self, n_classes):
        super(ArcfaceNN, self).__init__()
        self.n_classes = n_classes
        self.fc = nn.Linear(EMBEDDING_SIZE, self.n_classes)

    def forward(self, x):
        emb = super(ArcfaceNN, self).forward(x)

        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=0)

        x = self.fc(emb)

        return x

    def make_emb(self, x):
        return super(ArcfaceNN, self).forward(x)