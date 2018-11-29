import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from datasets.DoodleSubmitDataset import DoodleSubmitDataset
from models.CNN5 import CNN5
from models.CNN6 import CNN6
from models.CNN7 import CNN7
from models.CNN8 import CNN8

batch_size = 100
num_classes = 31
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

submit_ds = DoodleSubmitDataset()
submit_loader = torch.utils.data.DataLoader(dataset=submit_ds, batch_size=batch_size, shuffle=False)

model = CNN8(num_classes)
model.load_state_dict(torch.load('CNN8_2_best.ckpt'))
model.to(device)

model2 = CNN8(num_classes)
model2.load_state_dict(torch.load('model9_cnn8.ckpt'))
model2.to(device)

model3 = CNN7(num_classes)
model3.load_state_dict(torch.load('model8_cnn7.ckpt'))
model3.to(device)

model4 = CNN6(num_classes)
model4.load_state_dict(torch.load('model7_cnn6.ckpt'))
model4.to(device)

model5 = CNN5(num_classes)
model5.load_state_dict(torch.load('model6_cnn5.ckpt'))
model5.to(device)

model6 = CNN8(num_classes)
model6.load_state_dict(torch.load('model9_cnn8_2.ckpt'))
model6.to(device)

model.eval()
model2.eval()
model3.eval()
model4.eval()
model5.eval()
model6.eval()

name_dic = {}
lbls = []
count = 0
train_labels = np.genfromtxt("train_labels.csv", delimiter=",", dtype=str, skip_header=True)
for _, lbl in train_labels:
    if lbl not in name_dic:
        name_dic[lbl] = count
        lbls.append(lbl)
        count += 1


model.eval()
model2.eval()
model3.eval()
model4.eval()
model5.eval()
model6.eval()

with torch.no_grad():
    with open('submit_final.csv', 'w') as file:
        file.write('Id,Category\n')
        for index, imgs in submit_loader:
            imgs = imgs.to(device=device, dtype=torch.float)
            outputs = model(imgs)
            outputs2 = model2(imgs)
            outputs3 = model3(imgs)
            outputs4 = model4(imgs)
            outputs5 = model5(imgs)
            outputs6 = model6(imgs)
            output_combined = (outputs + outputs2 + outputs3 + outputs4 + outputs5 + outputs6) / 6
            _, predicted = torch.max(output_combined.data, 1)
            for i, pred in enumerate(predicted):
                file.write('{},{}\n'.format(index[i], lbls[pred]))

