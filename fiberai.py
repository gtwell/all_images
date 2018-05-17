import os
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms, models
from torch import nn, optim
from torch.autograd import Variable
import seaborn as sns
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter


def default_loader(path):
    return Image.open(path).convert('RGB')


class myImageFloder(Dataset):
    def __init__(self, csv_file, root, transform=None,
                 target_transform=None,
                 loader=default_loader):
        self.files = pd.read_csv(csv_file, names=['images', 'labels'])
        # file = file.sample(frac=1).reset_index(drop=True)
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
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


batch_size = 32
num_epochs = 50
learning_rate = 1e-4
root = '/home/gtwell/all_images/data'
train_dir = '/home/gtwell/all_images/data/Annotations/train.csv'
#test_dir = '/home/gtwell/FashionAI/neck/neck_design_labels_test.csv'
train_transform = transforms.Compose([transforms.Resize((299, 299)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.ColorJitter(0.4, 0.4),
                                      transforms.RandomRotation(40),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
test_transform = transforms.Compose([transforms.Resize((299, 299)),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

train_dataset = myImageFloder(csv_file=train_dir, root=root, transform=train_transform)
#test_dataset = myImageFloder(csv_file=test_dir, root=root, transform=test_transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
#test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=3)


# for index, (image, label) in enumerate(test_loader):
#     print(image, label)
#     if index >=2:
#         break


def train(learning_rate, optimizer, num_epochs):
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
            outputs = resnet(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            correct = torch.sum(predicted == labels.data)
            # correct = (predicted.cpu() == labels1).sum()
            acc = correct / images.size(0)
            accuracy.append(acc)
            loss.backward()
            losses.append(loss.data[0])
            optimizer.step()

            if (i + 1) % 50 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data[0]))
                print('Accuracy is: {:>7.3%}'.format(acc))

        # Decaying Learning Rate
        if (epoch + 1) % 10 == 0:
            learning_rate /= 3
            optimizer = optim.Adam(resnet.parameters(), lr=learning_rate)

    def plot():
        sns.set_style("darkgrid")

        plt.figure()
        plt.plot(losses, label='loss curves')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig('resnet50_fashionai_loss1.png', dpi=350)

        plt.figure()
        plt.plot(accuracy, 'r', label='accuracy curves')
        plt.xlabel('iteration')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig('resnet50_fashionai_acc1.png', dpi=350)
        # plt.show()

    plot()


writer = SummaryWriter('/home/gtwell/all_images/loss/loss0/')
save_dir = '/home/gtwell/all_images/save_model/inception.pkl'

def creat_model(is_train=True, loader_defeault_model=True):
    if is_train:
        if loader_defeault_model:
            inception = models.inception_v3(pretrained=loader_defeault_model)
            num_ftrs = inception.fc.in_features
            inception.fc = nn.Linear(num_ftrs, 5)
        else:
            inception = models.inception_v3(pretrained=False)
            num_ftrs = inception.fc.in_features
            inception.fc = nn.Linear(num_ftrs, 5)
            inception.load_state_dict(torch.load(save_dir))
    else:
        inception = models.inception_v3(pretrained=False)
        num_ftrs = inception.fc.in_features
        inception.fc = nn.Linear(num_ftrs, 5)
        inception.load_state_dict(torch.load(save_dir))
        inception.eval()
    # model_ft = models.resnet18(pretrained=True)
    nn.init.xavier_uniform(inception.fc.weight)
    nn.init.constant(inception.fc.bias, 0)
    inception.cuda()

    return inception


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


# Test the Model
def test(loader):
    # inception = models.inception_v3(pretrained=False)
    # # model_ft = models.resnet18(pretrained=True)
    # num_ftrs = inception.fc.in_features
    # inception.fc = nn.Linear(num_ftrs, 5)
    # # nn.init.xavier_uniform(resnet.fc.weight)
    # # nn.init.constant(resnet.fc.bias, 0)
    # inception.load_state_dict(torch.load(model_dir))
    # inception.cuda()
    # inception.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

    inception = creat_model(is_train=False, loader_defeault_model=False)

    correct = 0
    total = 0
    for images, labels in loader:
        images = Variable(images).cuda()
        labels = torch.squeeze(labels)
        outputs = inception(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()
        # print('(predicted.cpu() == labels).sum()', (predicted.cpu() == labels).sum())
    print('predicted number correctly is {}, and total number is {}'.format(correct, total))
    print('Test Accuracy of the model on the {1} test images: {0:.3%}'
          .format((correct / total), (len(test_dataset))))


if __name__ == '__main__':
    train_epoch(learning_rate=learning_rate, num_epochs=num_epochs, is_train=True, loader_defeault_model=True)
    #test(loader=test_loader)

