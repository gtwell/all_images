import os
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms, models
from torch import nn, optim
from torch.autograd import Variable
import time
import copy
from torch.optim import lr_scheduler
import ipdb
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter


def default_loader(path):
    return Image.open(path).convert('RGB')


class MyImageFolder(Dataset):
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


# batch_size = 32
# num_epochs = 50
# learning_rate = 1e-4
# root = '/home/gtwell/all_images/data'
# train_dir = '/home/gtwell/all_images/data/Annotations/train.csv'
# #test_dir = '/home/gtwell/FashionAI/neck/neck_design_labels_test.csv'

def create_dataset(csv_file = './five_folders/fiber_a/{}.csv',
                   root_dir = '/home/gtwell/all_images/dataset',
                   phase = ['train', 'val'],
                   shuffle=True,
                   img_size=224,
                   batch_size=32):
    """Create dataset, dataloader for train and test
    Args: label_type (str): Type of label
        csv_file (str): CSV file pattern for file indices.
        root_dir (str): Root dir based on paths in csv file.
        phase: list of str 'train' or 'test'.
    Returns:
        out (dict): A dict contains image_datasets, dataloaders,
            dataset_sizes
    """

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {}
    dataloaders = {}
    dataset_sizes = {}
    for x in phase:    # ['train', 'val']
        image_datasets[x] = MyImageFolder(csv_file.format(x),
                                          root_dir,
                                          data_transforms[x])
        if x == 'train':
            dataloaders[x] = DataLoader(image_datasets[x],
                                        batch_size=batch_size,
                                        shuffle=shuffle and x=='train',
                                        num_workers=3)
        else:
            dataloaders[x] = DataLoader(image_datasets[x],
                                        batch_size=32,
                                        num_workers=3)
        dataset_sizes[x] = len(image_datasets[x])

    out = {'image_datasets': image_datasets,
           'dataloaders': dataloaders,
           'dataset_sizes': dataset_sizes}
    return out


csv_file = './five_folders/' + 'fiber_a/' + '{}.csv'
root_dir = '/home/gtwell/all_images/dataset'
out = create_dataset(csv_file=csv_file, root_dir=root_dir, batch_size=64)
dataloaders = out['dataloaders']
dataset_sizes = out['dataset_sizes']

# for index, (image, label) in enumerate(out['image_datasets']['train']):
#     print(image, label)
#     if index >=2:
#         break

writer = SummaryWriter('/home/gtwell/all_images/pre_log/loss/resnet18_adm/')
# save_dir = '/home/gtwell/all_images/save_model/inception.pkl'


def train_model(model, criterion, optimizer, scheduler, num_epochs, saved_model_file, load_checkpoint=False):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_iter = 0
    val_iter = 0
    start_epoch = 0

    if load_checkpoint:
        if os.path.isfile(saved_model_file):
            print("=> loading checkpoint '{}'".format(saved_model_file))
            checkpoint = torch.load(saved_model_file)
            start_epoch = checkpoint['epoch']
            train_iter = checkpoint['train_iter']
            val_iter = checkpoint['val_iter']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint from epoch {})"
                  .format(checkpoint['epoch']))

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            batch_num = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train

                outputs = model(inputs)

                _, preds = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                    if batch_num % 100 == 0:
                        print('batch: #{}, loss = {}'.format(batch_num, loss.data[0]))
                    batch_num += 1



                # statistics
                running_loss += loss.data[0] * inputs.size(0)
                # ipdb.set_trace()
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            if phase == 'train':
                train_loss = epoch_loss
                train_acc = epoch_acc
                writer.add_scalar('data/train_loss', train_loss, train_iter)
                writer.add_scalar('data/train_acc', train_acc, train_iter)
                writer.add_scalars('data/scalar_group', {'train_loss': train_loss,
                                                         'train_acc': train_acc,}, train_iter)
                train_iter += 1
            else:
                val_loss = epoch_loss
                val_acc = epoch_acc
                writer.add_scalar('data/val_loss', val_loss, val_iter)
                writer.add_scalar('data/val_acc', val_acc, val_iter)
                writer.add_scalars('data/scalar_group', {'val_loss': val_loss,
                                                         'val_acc': val_acc, }, val_iter)
                val_iter += 1

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, saved_model_file)

                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': best_model_wts,
                    'train_iter': train_iter,
                    'val_iter': val_iter,
                }, saved_model_file)


        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 6)

model_ft.cuda()

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
#                        num_epochs=25, saved_model_file='saved_model_file/resnet18_adm.pkl')

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=9, saved_model_file='pre_training/resnet18_checkpoint.tar', load_checkpoint=True)

writer.close()








