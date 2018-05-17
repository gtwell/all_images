import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

def default_loader(path):
    return Image.open(path).convert('RGB')


class MyImageFloder(Dataset):
    def __init__(self, csv_file, root, transform=None, loader=default_loader):
        self.files = pd.read_csv(csv_file, names=['images', 'labels'])
        # file = file.sample(frac=1).reset_index(drop=True)
        self.root = root
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_name = self.files.iloc[index,0]
        img = self.loader(os.path.join(self.root, img_name))
        if self.transform is not None:
            img = self.transform(img)
        labels = self.files.iloc[index, 1]
        return img, labels

    def __len__(self):
        return len(self.files)

def create_dataloader():
    r"""
    Args:



    Returns:



    """

if __name__ == '__main__':
    root = '/home/gtwell/all_images/data'
    train_dir = '/home/gtwell/all_images/data/Annotations/train.csv'
    train_transform = transforms.Compose([transforms.Resize((299, 299)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.ColorJitter(0.4, 0.4),
                                          transforms.RandomRotation(40),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    dataset = MyImageFloder(csv_file=train_dir, root=root, transform=train_transform)
    print(dataset[0])
