import numpy as np
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class DoodleSubmitDataset(Dataset):

    def __init__(self):
        self.imgs = np.load('test_images.npy', encoding='latin1')

    def __getitem__(self, index):
        return (index, transforms.ToTensor()(self.imgs[index, 1].reshape(100, 100)[:, :, None]))

    def __len__(self):
        return len(self.imgs)
