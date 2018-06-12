import torch
from tensorboardX import SummaryWriter
import argparse
from torch.autograd import Variable
from data_loader.dataset import create_test_dataset
from model.models import create_model
from torchvision import models
from torch import nn
import matplotlib.pyplot as plt
import ipdb
from model.models import create_model

# parser = argparse.ArgumentParser("test")
# parser.add_argument("--csv_file", type=str, default='fiber_a', help="input file")
# args = parser.parse_args()

csv_file = './dataset/Annotations/test.csv'
# csv_file = './five_folders/fiber_a/val.csv'
root_dir = '/home/gtwell/all_images/dataset'

out = create_test_dataset(csv_file=csv_file,
                          root_dir=root_dir,
                          img_size=299,
                          batch_size=16)
dataloaders = out['dataloaders']
dataset_sizes = out['dataset_sizes']

model_ft = create_model(model_key='inception_v3',
                        pretrained=eval('False'),
                        num_of_classes=6,
                        use_gpu=eval('True'))
#model_ft = models.resnet50(pretrained=False)
#num_ftrs = model_ft.fc.in_features
#model_ft.fc = nn.Linear(num_ftrs, 6)
#model_ft.cuda()

model_ft.load_state_dict(torch.load('/home/gtwell/all_images/saved_model_file/fiber_a/inception_v3/inception_v3_SGD_0611.pkl'))
model_ft.eval()
correct = 0
total = 0

for i, (inputs, labels) in enumerate(dataloaders):
    inputs = Variable(inputs).cuda()
    labels = Variable(labels).cuda()

    outputs = model_ft(inputs)

    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.data).sum()
# print('(predicted.cpu() == labels).sum()', (predicted.cpu() == labels).sum())
print('predicted number correctly is {}, and total number is {}'.format(correct, total))
print('Test Accuracy of the model on the {1} test images: {0:.3%}'
        .format((correct / total), (dataset_sizes)))
