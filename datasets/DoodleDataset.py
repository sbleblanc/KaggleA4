import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class DoodleDataset(Dataset):

    # test here actually refers to validation
    def __init__(self, test_ratio=0.2, train_test='train', augment_factor=30):
        self.to_tensor = transforms.Compose([
            transforms.ToTensor()
        ])
        self.to_tensor_augmentation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation((-45, 45)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(0, scale=(0.8, 1.3)),
            transforms.RandomAffine(0, shear=(-10, 10)),
            transforms.ToTensor()
        ])
        self.augment_factor = augment_factor
        self.cache = {}

        self.state = train_test
        self.imgs = np.load('train_images.npy', encoding='latin1')
        self.train_split = int(self.imgs.shape[0] * (1.0 - test_ratio))
        self.test_split = int(self.imgs.shape[0] * test_ratio)
        self.labels = []
        labels_raw = np.genfromtxt('train_labels.csv', names=True, delimiter=',',
                                   dtype=[('Id', 'i8'), ('Category', 'S5')])

        name_dic = {}
        count = 0
        for _, lbl in labels_raw:
            if lbl not in name_dic:
                name_dic[lbl] = count
                self.labels.append(count)
                count += 1
            else:
                self.labels.append(name_dic[lbl])

    def __getitem__(self, index):
        if self.state == 'train':
            if self.augment_factor > 0:
                correct_i = int(np.floor(index / self.augment_factor))
            else:
                correct_i = index
            img = self.imgs[correct_i, 1]
            lbl = self.labels[correct_i]
            if self.augment_factor > 0:
                if index % self.augment_factor != 0:
                    if index not in self.cache:
                        self.cache[index] = self.to_tensor_augmentation(
                            img.astype(np.float32).reshape(100, 100)[:, :, None])
                    return (lbl, self.cache[index])
        elif self.state == 'test':
            img = self.imgs[self.train_split + index, 1]
            lbl = self.labels[self.train_split + index]
        return (lbl, self.to_tensor(img.astype(np.float32).reshape(100, 100)[:, :, None]))

    def __len__(self):
        if self.state == 'train':
            if self.augment_factor > 0:
                return self.train_split * self.augment_factor
            else:
                return self.train_split
        elif self.state == 'test':
            return self.test_split
