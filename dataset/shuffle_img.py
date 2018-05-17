import os
import random
from PIL import Image
import argparse

def shuffle_images(path, new_folder):
    r"""
    args:
        path: the directory we need to shuffle
    output:
        a shuffled file.
    """

    if not os.path.exists(new_folder):
        print('new_folder: {} is not existed'.format(new_folder))
        os.makedirs(new_folder)
    else:
        print('new_folder: {} is existed'.format(new_folder))
    file_list = os.listdir(path)
    random.shuffle(file_list)
    for index, img in enumerate(file_list):
        img = os.path.join(path,img)
        img = Image.open(img)
        img.save(new_folder+str(index)+'.jpg')

if __name__ == '__main__':
    path = 'Images/cn_cyan_cashmere/'
    new_folder = 'shuffle_images/cn_cyan_cashmere/'
    parser = argparse.ArgumentParser("shuffle images")
    parser.add_argument("--path", type=str, default=path, help="input file")
    parser.add_argument("--new_folder", type=str, default=new_folder, help="output file")
    args = parser.parse_args()
    shuffle_images(args.path, args.new_folder)
