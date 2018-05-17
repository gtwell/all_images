import pandas as pd
import os
import argparse

dirs = ['fiber_a/',
        'fiber_b/',
        'fiber_c/',
        'fiber_d/',
        'fiber_e/']

for folder in dirs:
    if not os.path.exists(folder):
        print('folder: {} is not existed'.format(folder))
        os.makedirs(folder)
    else:
        print('folder: {} is existed'.format(folder))
        print(os.getcwd())

def split_csv(csv_file):
    data = pd.read_csv(csv_file, names=['images', 'labels'])
    step = len(data)//5
    f = dict()
    f[0] = data[0:step]
    f[1] = data[step:2*step]
    f[2] = data[2*step:3*step]
    f[3] = data[3*step:4*step]
    f[4] = data[4*step:]
    
    for i in range(5):
        l = list(range(5))
        train_idxs = l[:i] + l[i+1:]
        val_idxs = f[i]
        train = pd.concat([f[i] for i in train_idxs])
        val = val_idxs
        train.to_csv(os.path.join(dirs[i], 'train.csv'), index=False, header=False)
        val.to_csv(os.path.join(dirs[i], 'val.csv'), index=False, header=False)

if __name__ == '__main__':
    csv_file = '/home/gtwell/all_images/dataset/Annotations/train.csv'
    parser = argparse.ArgumentParser("split_csv")
    parser.add_argument("--csv_file", type=str, default=csv_file, help="input file")
    args = parser.parse_args()
    split_csv(args.csv_file)
