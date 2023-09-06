import os
import argparse
import random
import shutil

parser = argparse.ArgumentParser(description='Create test')
parser.add_argument('--dataset_source', '-ds', default='./coco/train2014', type=str)
parser.add_argument('--dataset_destination', '-dd', default='./coco/test', type=str)
parser.add_argument('--size', '-s', default='100', type=int)
args = parser.parse_args()

dataset_source = args.dataset_source
dataset_destination = args.dataset_destination
dataset_size = args.size
force_update = args.force_update
fs_path = f'{dataset_source}/../fs.list'

assert(os.path.exists(dataset_source), "source dataset folder does not exist")
assert(os.path.exists(f'{dataset_destination}/{dataset_size}'), "destination dataset folder already exists")
assert(os.path.exists(fs_path), f"fs.list file not found in {fs_path}")
fs_exixts = os.path.exists(fs_path)

print("reading fs.list")
with open(fs_path, 'r') as f:
    imagepaths = f.read().splitlines()

for count in range(dataset_size):
    source_dataset_size = len(imagepaths)
    rand_num = random.random(0, source_dataset_size - 1)
    popped_img = imagepaths.pop(rand_num)
    fn = popped_img.split('/')[len(popped_img)-1]
    shutil.copyfile(popped_img, f'{dataset_destination}/{fn}')

n_data = len(imagepaths)
print('num traning images:', n_data)