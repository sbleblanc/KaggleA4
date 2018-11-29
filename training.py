import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from datasets.DoodleDataset import DoodleDataset
from models.CNN8 import CNN8

num_epochs = 20
num_classes = 31
batch_size = 100
learning_rate = 0.0001
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_ds = DoodleDataset(augment_factor=30)
train_loader = torch.utils.data.DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
test_ds = DoodleDataset(train_test='test')
test_loader = torch.utils.data.DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False)

model = CNN8(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)

max_acc = 0.
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (lbls, imgs) in enumerate(train_loader):
        imgs = imgs.to(device=device, dtype=torch.float)
        lbls = lbls.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, lbls)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for lbls, imgs in train_loader:
            imgs = imgs.to(device=device, dtype=torch.float)
            lbls = lbls.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total += lbls.size(0)
            correct += (predicted == lbls).sum().item()

        print('Train Accuracy of the model on the {} train images: {} %'.format(total, 100 * correct / total))
    with torch.no_grad():
        correct = 0
        total = 0
        for lbls, imgs in test_loader:
            imgs = imgs.to(device=device, dtype=torch.float)
            lbls = lbls.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total += lbls.size(0)
            correct += (predicted == lbls).sum().item()

        accu = correct / total
        print('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))

    # Quick solution to implement early stopping. Save a seperate copy of the model parameters if the results are better
    # When the valid accuracy drops, ca simply stop or continue to monitor
    if accu >= max_acc:
        max_acc = accu
        torch.save(model.state_dict(), 'CNN8_submit_best.ckpt')

    model.train()

torch.save(model.state_dict(), 'CNN8_submit_final.ckpt')