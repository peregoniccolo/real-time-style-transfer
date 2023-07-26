import os

dataset = './coco/train2014'

if os.path.exists(f'{dataset}/../fs.list'):
    # read from file with names to save time
    print("reading fs.list")
    with open(f'{dataset}/../fs.list') as f:
        imagepaths = f.read().splitlines()
else:
    # one off, create file with image names
    print("reading dataset directory")
    fs = os.listdir(dataset)
    imagepaths = []
    for fn in fs:
        base, ext = os.path.splitext(fn)
        if ext == '.jpg' or ext == '.png':
            imagepath = os.path.join(dataset, fn)
            imagepaths.append(imagepath)
    print("saving in fs.list") 
    with open(f'{dataset}/../fs.list', 'w') as tfile:
        tfile.write('\n'.join(imagepaths))

n_data = len(imagepaths)
print('num traning images:', n_data)