import torch
from tensorboardX import SummaryWriter
import argparse
from torch.autograd import Variable
from data_loader.dataset import create_test_dataset
from torchvision import models
from torch import nn
import matplotlib.pyplot as plt
import ipdb

# parser = argparse.ArgumentParser("test")
# parser.add_argument("--csv_file", type=str, default='fiber_a', help="input file")
# args = parser.parse_args()

csv_file = './dataset/Annotations/test.csv'
# csv_file = './five_folders/fiber_a/val.csv'
root_dir = './dataset/'

out = create_test_dataset(csv_file=csv_file,
                          root_dir=root_dir,
                          img_size=224,
                          batch_size=32)
dataloaders = out['dataloaders']
dataset_sizes = out['dataset_sizes']

model_ft = models.resnet18(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 6)
model_ft.load_state_dict(torch.load('./saved_model_file/fiber_a/resnet18/resnet18_Adam.pkl'))
#nn.init.xavier_uniform(model_ft.fc.weight)
#nn.init.constant(model_ft.fc.bias, 0)
model_ft.cuda()
model_ft.eval()

# use_gpu = torch.cuda.is_available()
# model_conv = create_model(model_key='resnet18',
#                           pretrained='False',
#                           num_of_classes=6,
#                           use_gpu=use_gpu)

# saved_model_file='pre_training/resnet18_checkpoint.tar'
# checkpoint = torch.load(saved_model_file)
# model_conv.load_state_dict(checkpoint['state_dict'])

# model_ft.load_state_dict(torch.load('./pre_training/resnet50.pkl'))
# writer = SummaryWriter('./pre_log_no_eval')
correct = 0
total = 0
global_step = 0

for i, (inputs, labels) in enumerate(dataloaders):
    inputs = Variable(inputs).cuda()
    labels = Variable(labels).cuda()

    x = model_ft.conv1(inputs)
    x = model_ft.bn1(x)
    x = model_ft.relu(x)
    x = model_ft.maxpool(x)
#    x = model_ft.layer1(x)
#    x = model_ft.layer2(x)
    fig = plt.figure()
    for a in range(1, 65):
        y = x[0][a-1][:][:].data
        y = y.cpu()
        y = y.numpy()
        ax = 'ax' + str(a)
        ax = fig.add_subplot(8, 8, a)
        ax.imshow(y, cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0, left=0.20, right=0.80, bottom=0.01, top=0.99)
    ipdb.set_trace()
#    plt.tight_layout()
#    ipdb.set_trace()
    # x = model_ft.layer2(x)
    # x = model_ft.layer3(x)
    # x = model_ft.layer4(x)
    # outputs = model_ft.avgpool(x)
    # outputs = torch.squeeze(outputs)
#     outputs = model_ft(inputs)
    # if i == 0:
    #     out = outputs
    #     label_data = labels
    #     # input_data = inputs
    # else:
    #     out = torch.cat((out, outputs), dim=0)
    #     # input_data = torch.cat((input_data, inputs), dim=0)
    #     label_data = torch.cat((label_data, labels), dim=0)
    # if i == 5:
    #     print('{} has finished'.format(i))

    # writer.add_embedding(outputs.data, metadata=labels.data, global_step=global_step)
    # global_step += 1
#    _, predicted = torch.max(outputs.data, 1)
#    total += labels.size(0)
#    correct += (predicted == labels.data).sum()
#     print('(predicted.cpu() == labels).sum()', (predicted.cpu() == labels).sum())
# print('predicted number correctly is {}, and total number is {}'.format(correct, total))
# print('Test Accuracy of the model on the {1} test images: {0:.3%}'.format((correct / total), (dataset_sizes)))
