import os
import argparse
import random
import shutil

parser = argparse.ArgumentParser(description='Create test')
parser.add_argument('--dataset_source', '-ds', default='coco/train2014', type=str)
parser.add_argument('--dataset_destination', '-dd', default='coco/test', type=str)
parser.add_argument('--size', '-s', default='100', type=int)
args = parser.parse_args()

dataset_source = args.dataset_source
dataset_destination = args.dataset_destination
dataset_size = args.size
fs_path = f'{dataset_source}/fs.list'

assert os.path.exists(dataset_source), 'source dataset folder does not exist'

assert os.path.exists(fs_path), f'fs.list file not found in {fs_path}'
fs_exixts = os.path.exists(fs_path)

assert not os.path.exists(f'{dataset_destination}{dataset_size}'), 'destination dataset folder already exists'
os.mkdir(f'{dataset_destination}{dataset_size}')
os.mkdir(f'{dataset_destination}{dataset_size}/images')

print('reading source fs.list')
with open(fs_path, 'r') as f:
    imagepaths = f.read().splitlines()

n_data = len(imagepaths)
print(f'num source images: {n_data}')    

for count in range(dataset_size):
    source_dataset_size = len(imagepaths)
    rand_num = random.randint(0, source_dataset_size - 1)
    popped_img = imagepaths.pop(rand_num)
    fn = popped_img.split('/')
    fn = fn[len(fn)-1]
    shutil.copyfile(f'{dataset_source}/images/{fn}', f'{dataset_destination}{dataset_size}/images/{fn}')
    print(f'{count}: copied {fn}')