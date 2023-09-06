import os
import argparse

parser = argparse.ArgumentParser(description='List population')
parser.add_argument('--force_update', '-f', default=False,
                    action=argparse.BooleanOptionalAction)
parser.add_argument('--dataset', '-d', default='./coco/train2014', type=str)
args = parser.parse_args()

dataset = args.dataset
force_update = args.force_update
fs_path = f'{dataset}/../fs.list'

fs_exixts = os.path.exists(fs_path)

if fs_exixts and not force_update:
    # read from file with names to save time
    print("reading fs.list")
    with open(fs_path, 'r') as f:
        imagepaths = f.read().splitlines()
else:
    # one off, create file with image names
    if force_update and fs_exixts:
        print("deleting fs.list")
        os.remove(fs_path)

    print("reading dataset directory")
    fs = os.listdir(dataset)
    imagepaths = []
    for fn in fs:
        base, ext = os.path.splitext(fn)
        if ext == '.jpg' or ext == '.png':
            imagepath = os.path.join(dataset, fn)
            imagepaths.append(imagepath)
    print("saving in fs.list") 
    with open(fs_path, 'w') as tfile:
        tfile.write('\n'.join(imagepaths))

n_data = len(imagepaths)
print('num traning images:', n_data)