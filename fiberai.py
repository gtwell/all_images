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
                                        batch_size=8,
                                        shuffle=shuffle and x == 'train',
                                        num_workers=3)
        dataset_sizes[x] = len(image_datasets[x])

    out = {'image_datasets': image_datasets,
           'dataloaders': dataloaders,
           'dataset_sizes': dataset_sizes}
    return out


csv_file = './five_folders/' + 'fiber_a/' + '{}.csv'
root_dir = '/home/gtwell/all_images/dataset'
out = create_dataset(csv_file=csv_file, root_dir=root_dir,)
dataloaders = out['dataloaders']
dataset_sizes = out['dataset_sizes']

# for index, (image, label) in enumerate(out['image_datasets']['train']):
#     print(image, label)
#     if index >=2:
#         break

# writer = SummaryWriter('/home/gtwell/all_images/loss/loss0/')
# save_dir = '/home/gtwell/all_images/save_model/inception.pkl'


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
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

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    torch.save(best_model_wts, 'resnet_18.pkl')
    model.load_state_dict(best_model_wts)
    return model


model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 6)

model_ft.cuda()

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)





def train_epoch(learning_rate, num_epochs, is_train=True, loader_defeault_model=True):
    # inception = models.inception_v3(pretrained=False)
    # # model_ft = models.resnet18(pretrained=True)
    # num_ftrs = inception.fc.in_features
    # inception.fc = nn.Linear(num_ftrs, 5)
    # # nn.init.xavier_uniform(resnet.fc.weight)
    # # nn.init.constant(resnet.fc.bias, 0)
    # inception.load_state_dict(torch.load('/home/gtwell/FashionAI/neck/save_model/inception_0.pkl'))
    # inception.cuda()

    inception = creat_model(is_train=is_train, loader_defeault_model=loader_defeault_model)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(inception.parameters(), lr=learning_rate, weight_decay=0.005)
    losses = []
    accuracy = []
    # global learning_rate
    # global optimizer
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images).cuda()
            # labels1 = labels
            labels = torch.squeeze(labels)
            labels = Variable(labels).cuda()

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = inception(images)[0]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            writer.add_scalar('data/losses', loss, epoch*(len(train_dataset)//batch_size) + i)

            if (i + 1) % 50 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))
                _, predicted = torch.max(outputs.data, 1)
                correct = torch.sum(predicted == labels.data)
                acc = correct / images.size(0)
                accuracy.append(acc)
                losses.append(loss.data[0])
                writer.add_scalar('data/accuracy', acc, epoch*((len(train_dataset)//batch_size)//50) + i // 50)
                print('Accuracy is: {:>7.3%}'.format(acc))

        # Decaying Learning Rate
        if (epoch + 1) % 5 == 0:
            learning_rate /= 2
            optimizer = optim.Adam(inception.parameters(), lr=learning_rate, weight_decay=0.01)
            torch.save(inception.state_dict(), save_dir)

    writer.close()
    # def plot():
    #     sns.set_style("darkgrid")
    #
    #     plt.figure()
    #     plt.plot(losses, label='loss curves')
    #     plt.xlabel('iteration')
    #     plt.ylabel('loss')
    #     plt.legend()
    #     plt.savefig('resnet34_transfer.png', dpi=350)
    #
    #     plt.figure()
    #     plt.plot(accuracy, 'r', label='accuracy curves')
    #     plt.xlabel('iteration')
    #     plt.ylabel('accuracy')
    #     plt.legend()
    #     plt.savefig('resnet34_transfer.png', dpi=350)
    #     # plt.show()

    # plot()

    # Save the Trained Model
    torch.save(inception.state_dict(), save_dir)




