from torch import optim
from torch.nn import functional as F

from model import ArcfaceNN, ArcfaceLoss

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

EPOCHS = 50
n_classes = dataset.targets[-1] + 1
arcface_model = ArcfaceNN(n_classes).to(device)
lr = 0.0001
criterion = ArcfaceLoss
optimizer = optim.Adam(arcface_model.parameters(), lr=lr)
arcface_model.train()

for epoch in range(EPOCHS):
    epoch_loss = 0
    for data, label in dataloader:
        optimizer.zero_grad()
        data = data.to(device)
        label = label.to(device)

        y_pred = arcface_model.forward(data)
        label = F.one_hot(label, num_classes=n_classes)

        loss = criterion(y_pred, label)
        epoch_loss += loss.detach().cpu()
        loss.backward()
        optimizer.step()

    torch.save(arcface_model, f"./arcface_{epoch_loss}.pt")
    print(f'[{epoch}, loss: {epoch_loss}')