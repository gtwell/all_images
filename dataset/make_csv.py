import os
import pandas as pd
import argparse

def make_csv(Images_path, new_csv):
    r"""
    args:
        path: the directory we need to make csv file
    output:
        generative a new csv
    """

    dirs = []
    for i in os.listdir(Images_path):
        dirs.append(i)

    print(dirs)

    images = []
    labels = []
    for index, folder in enumerate(dirs):
        for file in os.listdir(os.path.join(Images_path, folder)):
            images.append(os.path.join(Images_path, folder, file))
            labels.append(index)
    data = {'images': images, 'labels': labels}
    #data = pd.DataFrame(labels, images)
    data = pd.DataFrame(data)
    # 将数据打乱 frac=1 返回全部,将打混后数据集的index(索引)重新排序
    data = data.sample(frac=1).reset_index(drop=True)
    print(len(data))
    print(data.head())
    data.to_csv(new_csv, index= False, header=False)


if __name__ == '__main__':
    Images_path = 'Images/'
    new_csv = 'Annotations/train.csv'
    parser = argparse.ArgumentParser("make_csv")
    parser.add_argument("--Images_path", type=str, default=Images_path, help="input file")
    parser.add_argument("--new_csv", type=str, default=new_csv, help="output file")
    args = parser.parse_args()
    make_csv(args.Images_path, args.new_csv)
