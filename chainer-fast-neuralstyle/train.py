from __future__ import print_function, division
import numpy as np
import os
import argparse
import random
from PIL import Image
import natsort

from chainer import cuda, Variable, optimizers, serializers
from net import *

import glob


def output_name_check(output):
    # character _ generates problems later on so i substitute it
    if (output.__contains__('_')):
        print('found character "_" in the output name, replaced with "-" to avoid problems')
        return output.replace('_', '-')
    else:
        return output


def check_available_models(rel_model_path, output, checkpoint_number):
    query_models = glob.glob(f'{rel_model_path}check_{output}*.model')
    query_states = glob.glob(f'{rel_model_path}check_{output}*.state')

    # check the number of models and states coincide
    assert len(query_models) == len(
        query_states
    ), f'different number of models and states'
    # check number of saved models is lower than the specified amount to keep track of
    assert (
        len(query_models) <= checkpoint_number
    ), f'too much chakpoint saved, delete some or change checkpoint number'
    # check epochs and states match in models and states

    query_models = natsort.natsorted(query_models)
    query_states = natsort.natsorted(query_states)
    for count in np.arange(len(query_models)):
        name_model = (
            query_models[count].split('/')[-1].split('.')[0]
        )  # names without extention
        name_state = query_states[count].split('/')[-1].split('.')[0]
        assert name_model == name_state, 'model and state files do not coincide'

    return query_models, query_states

def get_input_epoch():
    epoch_str = input('insert number of completed epochs (including epoch 0): ')
    while not epoch_str.isdigit():
        epoch_str = input('input wasn\'t a digit, please insert number (integer) of completed epochs: ')
    return int(epoch_str)

def resume_from_epoch(fn, output):
    fn, _ = os.path.splitext(os.path.basename(fn))
    fn_split = fn.split('_')
    if fn_split[0] == 'check' or fn_split[0] == 'oldcheck':
        if len(fn_split) == 4 and fn_split[1] == output and fn_split[2].isdigit() and fn_split[3].isdigit():
            print('initmodel from filetype:\n' +
                  '\tcheck_{output}_{epoch}_{iteration}\n' +
                  '\toldcheck_{output}_{epoch}_{iteration}\n'
                  )
            return int(fn_split[2])  # epoch_to_return, has_ended_int
    elif fn_split[0] == 'final':
        if len(fn_split) == 4 and fn_split[1] == 'ep' and fn_split[2] == output and fn_split[3].isdigit():
            print('initmodel from filetype:\n' +
                  '\tfinal_ep_{output}_{epoch}\n' 
                  )
            return fn_split[3]
        if len(fn_split) == 2 and fn_split[1] == output:
            print('initmodel from filetype:\n' +
                  '\tfinal_{output}\n' 
                  )
            return get_input_epoch()

    print('bad naming convention or no epoch found in the specified model filename\n' +
          'please rename it following one of the possible conventions' +
          '\tcheck_{output}_{epoch}_{iteration}\n' +
          '\toldcheck_{output}_{epoch}_{iteration}\n' +
          '\tfinal_ep_{output}_{epoch}\n'
          )
    return None


def check_resume(query_models, query_states, oldest):
    if not query_models or not query_states:
        print('unmatching state and model found, starting over')
        return None, None, 0, 0

    oldest_it = 0 if oldest else -1

    if query_models:
        info = query_models[oldest_it].split('/')[-1].split('.')[0].split('_')
        start_it_model = int(info[-1])
        start_epoch_model = int(info[-2])

    initmodel = query_models[oldest_it]
    resume = query_states[oldest_it]
    return (
        initmodel,
        resume,
        start_it_model,
        start_epoch_model,
    )  # we know they are the same for state because we checked before


def delete_oldest_model(rel_model_path, output, checkpoint_number):
    curr_query_model = glob.glob(f'{rel_model_path}check_{output}*.model')
    curr_query_state = glob.glob(f'{rel_model_path}check_{output}*.state')
    if len(curr_query_model) == checkpoint_number + 1:  # +1 because i already saved the new one
        curr_query_model = natsort.natsorted(curr_query_model)
        curr_query_state = natsort.natsorted(curr_query_state)
        os.remove(curr_query_model[0])
        os.remove(curr_query_state[0])
    else:
        print('Not enough models, not deleting anything')


def mark_as_old(query_models, query_states):
    for model in query_models + query_states:
        path, model_name = os.path.split(model)
        new_model_name = f'old{model_name}'
        os.rename(model, f'{path}/{new_model_name}')


def load_image(path, size):
    image = Image.open(path).convert('RGB')
    w, h = image.size
    if w < h:
        if w < size:
            image = image.resize((size, size * h // w))
            w, h = image.size
    else:
        if h < size:
            image = image.resize((size * w // h, size))
            w, h = image.size
    image = image.crop(
        ((w - size) * 0.5, (h - size) * 0.5, (w + size) * 0.5, (h + size) * 0.5)
    )
    return xp.asarray(image, dtype=np.float32).transpose(2, 0, 1)


def gram_matrix(y):
    b, ch, h, w = y.data.shape
    features = F.reshape(y, (b, ch, w * h))
    gram = F.batch_matmul(features, features, transb=True) / \
        np.float32(ch * w * h)
    return gram


def total_variation(x):
    xp = cuda.get_array_module(x.data)
    b, ch, h, w = x.data.shape
    wh = Variable(
        xp.asarray(
            [
                [[[1], [-1]], [[0], [0]], [[0], [0]]],
                [[[0], [0]], [[1], [-1]], [[0], [0]]],
                [[[0], [0]], [[0], [0]], [[1], [-1]]],
            ],
            dtype=np.float32,
        )
    )
    ww = Variable(
        xp.asarray(
            [
                [[[1, -1]], [[0, 0]], [[0, 0]]],
                [[[0, 0]], [[1, -1]], [[0, 0]]],
                [[[0, 0]], [[0, 0]], [[1, -1]]],
            ],
            dtype=np.float32,
        )
    )
    return F.sum(F.convolution_2d(x, W=wh) ** 2) + F.sum(F.convolution_2d(x, W=ww) ** 2)


parser = argparse.ArgumentParser(description='Real-time style transfer')
group = parser.add_mutually_exclusive_group()
parser.add_argument(
    '--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)'
)
parser.add_argument(
    '--dataset',
    '-d',
    default='./coco/train2014',
    type=str,
    help='dataset directory path (according to the paper, use MSCOCO 80k images)',
)
parser.add_argument(
    '--style_image', '-s', type=str, required=True, help='style image path'
)
parser.add_argument(
    '--batchsize', '-b', type=int, default=1, help='batch size (default value is 1)'
)
group.add_argument(
    '--initmodel',
    '-i',
    default=None,
    type=str,
    help='initialize the model from given file',
)
parser.add_argument(
    '--resume',
    '-r',
    default=None,
    type=str,
    help='resume the optimization from snapshot',
)
parser.add_argument(
    '--output',
    '-o',
    default=None,
    type=str,
    help='output model file path without extension',
)
parser.add_argument(
    '--lambda_tv',
    '-ltv',
    default=1e-6,
    type=float,
    help='weight of total variation regularization according to the paper to be set between 10e-4 and 10e-6.',
)
parser.add_argument('--lambda_feat', '-lf', default=1.0, type=float)
parser.add_argument('--lambda_style', '-ls', default=5.0, type=float)
parser.add_argument(
    '--lambda_noise',
    '-ln',
    default=1000.0,
    type=float,
    help='Training weight of the popping induced by noise',
)
parser.add_argument(
    '--noise', '-n', default=30, type=int, help='range of noise for popping reduction'
)
parser.add_argument(
    '--noisecount',
    '-nc',
    default=1000,
    type=int,
    help='number of pixels to modify with noise',
)
parser.add_argument('--epoch', '-e', default=2, type=int)
parser.add_argument('--lr', '-l', default=1e-3, type=float)
parser.add_argument('--checkpoint', '-c', default=0, type=int)
parser.add_argument('--checkpoint_number', '-cn', default=2, type=int)
parser.add_argument('--image_size', '-is', default=256, type=int)
group.add_argument(
    '--auto_resume', '-a', default=False, action=argparse.BooleanOptionalAction
)
parser.add_argument(
    '--resume_from_oldest', '-rfo', default=False, action=argparse.BooleanOptionalAction
)
args = parser.parse_args()

dataset = args.dataset
batchsize = args.batchsize
image_size = args.image_size
n_epoch = args.epoch
lambda_tv = args.lambda_tv
lambda_f = args.lambda_feat
lambda_s = args.lambda_style
lambda_noise = args.lambda_noise
noise_range = args.noise
noise_count = args.noisecount
style_prefix, _ = os.path.splitext(os.path.basename(args.style_image))
output = style_prefix if args.output == None else args.output
output = output_name_check(output)
checkpoint = args.checkpoint
checkpoint_number = args.checkpoint_number

if os.path.exists(f'{dataset}/fs.list'):
    # read from file with names to save time
    print('reading fs.list')
    with open(f'{dataset}/fs.list') as f:
        imagepaths = f.read().splitlines()
else:
    # one off, create file with image names
    print('reading dataset directory')
    fs = os.listdir(f'{dataset}/images')
    imagepaths = []
    for fn in fs:
        base, ext = os.path.splitext(fn)
        if ext == '.jpg' or ext == '.png':
            imagepath = os.path.join(f'{dataset}/images', fn)
            imagepaths.append(imagepath)
    print('saving in fs.list')
    with open(f'{dataset}/fs.list', 'w') as tfile:
        tfile.write('\n'.join(imagepaths))

n_data = len(imagepaths)
print('num traning images:', n_data)
n_iter = n_data // batchsize
print(n_iter, 'iterations,', n_epoch, 'epochs')

model = FastStyleNet()
vgg = VGG()

rel_model_dir_path = f'models/{output}/'

if not os.path.exists(rel_model_dir_path):
    os.mkdir(rel_model_dir_path)

query_models, query_states = check_available_models(
    rel_model_dir_path, output, checkpoint_number
)

if args.auto_resume:
    # gather initmodel, resume and last iteration and epoch from saved files
    args.initmodel, args.resume, start_it, start_ep = check_resume(
        query_models, query_states, args.resume_from_oldest
    )
else:
    # manual resume, if specified model and state files will be used, but starting from it and ep 0
    start_it = 0
    start_ep = 0
    if args.initmodel:    
        _, model_name = os.path.split(args.initmodel)
        base_epoch = resume_from_epoch(model_name, output)
        if base_epoch != None: # it means also has ended is != None
            start_ep += base_epoch
            n_epoch  += base_epoch

serializers.load_npz('vgg16.model', vgg)
if args.initmodel:
    print('load model from', args.initmodel)
    serializers.load_npz(args.initmodel, model)

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
    vgg.to_gpu()
xp = np if args.gpu < 0 else cuda.cupy

O = optimizers.Adam(alpha=args.lr)
O.setup(model)
if args.resume:
    print('load optimizer state from', args.resume)
    serializers.load_npz(args.resume, O)

style = vgg.preprocess(
    np.asarray(
        Image.open(args.style_image).convert(
            'RGB').resize((image_size, image_size)),
        dtype=np.float32,
    )
)
style = xp.asarray(style, dtype=xp.float32)
style_b = xp.zeros((batchsize,) + style.shape, dtype=xp.float32)
for i in range(batchsize):
    style_b[i] = style
feature_s = vgg(Variable(style_b))
gram_s = [gram_matrix(y) for y in feature_s]

# before starting mark old models as they are, old
mark_as_old(query_models, query_states)

for epoch in range(start_ep, n_epoch):
    print('epoch', epoch)

    if noise_count:
        noiseimg = xp.zeros((3, image_size, image_size), dtype=xp.float32)

        # prepare a noise image
        for ii in range(noise_count):
            xx = random.randrange(image_size)
            yy = random.randrange(image_size)

            noiseimg[0][yy][xx] += random.randrange(-noise_range, noise_range)
            noiseimg[1][yy][xx] += random.randrange(-noise_range, noise_range)
            noiseimg[2][yy][xx] += random.randrange(-noise_range, noise_range)

    for i in range(start_it, n_iter):
        model.zerograds()
        vgg.zerograds()

        indices = range(i * batchsize, (i + 1) * batchsize)
        x = xp.zeros((batchsize, 3, image_size, image_size), dtype=xp.float32)
        for j in range(batchsize):
            x[j] = load_image(imagepaths[i * batchsize + j], image_size)

        xc = Variable(x.copy())

        if noise_count:
            # add the noise image to the source image
            noisy_x = x.copy()
            noisy_x = noisy_x + noiseimg

            noisy_x = Variable(noisy_x)
            noisy_y = model(noisy_x)
            noisy_y -= 120

        x = Variable(x)

        y = model(x)

        xc -= 120
        y -= 120

        feature = vgg(xc)
        feature_hat = vgg(y)

        # compute for only the output of layer conv3_3
        L_feat = lambda_f * F.mean_squared_error(
            Variable(feature[2].data), feature_hat[2]
        )

        L_style = Variable(xp.zeros((), dtype=np.float32))
        for f, f_hat, g_s in zip(feature, feature_hat, gram_s):
            L_style += lambda_s * F.mean_squared_error(
                gram_matrix(f_hat), Variable(g_s.data)
            )

        L_tv = lambda_tv * total_variation(y)

        # the 'popping' noise is the difference in resulting stylizations
        # from two images that are very similar. Minimizing it results
        # in a much more stable stylization that can be applied to video.
        # Small changes in the input result in small changes in the output.
        if noise_count:
            L_pop = lambda_noise * F.mean_squared_error(y, noisy_y)
            L = L_feat + L_style + L_tv + L_pop
            to_print = f'Epoch {epoch},{i}/{n_iter}. Total loss: {L.data}. Loss distribution: feat {L_feat.data/L.data}, style {L_style.data/L.data}, tv {L_tv.data/L.data}, pop {L_pop.data/L.data}'
        else:
            L = L_feat + L_style + L_tv
            to_print = f'Epoch {epoch},{i}/{n_iter}. Total loss: {L.data}. Loss distribution: feat {L_feat.data/L.data}, style {L_style.data/L.data}, tv {L_tv.data/L.data}'

        L.backward()
        O.update()

        if checkpoint > 0 and i % checkpoint == 0:
            serializers.save_npz(
                f'{rel_model_dir_path}check_{output}_{epoch}_{i}.model', model
            )
            serializers.save_npz(
                f'{rel_model_dir_path}check_{output}_{epoch}_{i}.state', O
            )
            delete_oldest_model(rel_model_dir_path, output,
                                checkpoint_number)

        # i'm done saving if needed
        print(to_print)

    print('save "style.model"')
    serializers.save_npz(
        f'{rel_model_dir_path}finalep_{output}_{epoch}.model', model)
    serializers.save_npz(
        f'{rel_model_dir_path}finalep_{output}_{epoch}.state', O)

    # finished an epoch, restarting from it 0
    start_it = 0

serializers.save_npz(f'{rel_model_dir_path}final_{output}.model', model)
serializers.save_npz(f'{rel_model_dir_path}final_{output}.state', O)
