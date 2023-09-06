from __future__ import print_function, division
import numpy as np
import os
import argparse
import random
from PIL import Image

from chainer import cuda, Variable, optimizers, serializers
from net import *

import glob


def check_resume(rel_model_path, output, oldest):
    oldest_it = 0 if oldest else 1

    query_model = glob.glob(f'{rel_model_path}check_{output}*.model')
    if (query_model):
        assert len(
            query_model) == 2, 'dunno where to start, 1 or more than 2 checkpoints for the same model found'
        query_model.sort()
        info = query_model[oldest_it].split('/')[-1].split('.')[0].split('_')
        start_it_model = int(info[-1])
        start_epoch_model = int(info[-2])

    query_state = glob.glob(f'{rel_model_path}check_{output}*.state')
    if (query_state):
        assert len(
            query_state) == 2, 'dunno where to start, 1 or more than 2 checkpoints for the same state found'
        query_state.sort()
        info = query_state[oldest_it].split('/')[-1].split('.')[0].split('_')
        start_it_state = int(info[-1])
        start_epoch_state = int(info[-2])

    if not query_model or not query_state:
        print('unmatching state and model found, starting over')
        return None, None, 0, 0

    if (start_epoch_model == start_epoch_state and start_it_model == start_it_state):
        initmodel = query_model[oldest_it]
        resume = query_state[oldest_it]
        return initmodel, resume, start_it_model, start_epoch_model


def load_image(path, size):
    image = Image.open(path).convert('RGB')
    w, h = image.size
    if w < h:
        if w < size:
            image = image.resize((size, size*h//w))
            w, h = image.size
    else:
        if h < size:
            image = image.resize((size*w//h, size))
            w, h = image.size
    image = image.crop(((w-size)*0.5, (h-size)*0.5,
                       (w+size)*0.5, (h+size)*0.5))
    return xp.asarray(image, dtype=np.float32).transpose(2, 0, 1)


def gram_matrix(y):
    b, ch, h, w = y.data.shape
    features = F.reshape(y, (b, ch, w*h))
    gram = F.batch_matmul(features, features, transb=True)/np.float32(ch*w*h)
    return gram


def total_variation(x):
    xp = cuda.get_array_module(x.data)
    b, ch, h, w = x.data.shape
    wh = Variable(xp.asarray([[[[1], [-1]], [[0], [0]], [[0], [0]]], [[[0], [0]], [
                  [1], [-1]], [[0], [0]]], [[[0], [0]], [[0], [0]], [[1], [-1]]]], dtype=np.float32))
    ww = Variable(xp.asarray([[[[1, -1]], [[0, 0]], [[0, 0]]], [[[0, 0]], [
                  [1, -1]], [[0, 0]]], [[[0, 0]], [[0, 0]], [[1, -1]]]], dtype=np.float32))
    return F.sum(F.convolution_2d(x, W=wh) ** 2) + F.sum(F.convolution_2d(x, W=ww) ** 2)


parser = argparse.ArgumentParser(description='Real-time style transfer')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--dataset', '-d', default='./coco/train2014', type=str,
                    help='dataset directory path (according to the paper, use MSCOCO 80k images)')
parser.add_argument('--style_image', '-s', type=str, required=True,
                    help='style image path')
parser.add_argument('--batchsize', '-b', type=int, default=1,
                    help='batch size (default value is 1)')
parser.add_argument('--initmodel', '-i', default=None, type=str,
                    help='initialize the model from given file')
parser.add_argument('--resume', '-r', default=None, type=str,
                    help='resume the optimization from snapshot')
parser.add_argument('--output', '-o', default=None, type=str,
                    help='output model file path without extension')
parser.add_argument('--lambda_tv', '-ltv', default=1e-6, type=float,
                    help='weight of total variation regularization according to the paper to be set between 10e-4 and 10e-6.')
parser.add_argument('--lambda_feat', '-lf', default=1.0, type=float)
parser.add_argument('--lambda_style', '-ls', default=5.0, type=float)
parser.add_argument('--lambda_noise', '-ln', default=1000.0, type=float,
                    help='Training weight of the popping induced by noise')
parser.add_argument('--noise', '-n', default=30, type=int,
                    help='range of noise for popping reduction')
parser.add_argument('--noisecount', '-nc', default=1000, type=int,
                    help='number of pixels to modify with noise')
parser.add_argument('--epoch', '-e', default=2, type=int)
parser.add_argument('--lr', '-l', default=1e-3, type=float)
parser.add_argument('--checkpoint', '-c', default=0, type=int)
parser.add_argument('--image_size', '-is', default=256, type=int)
parser.add_argument('--auto_resume', '-a', default=False,
                    action=argparse.BooleanOptionalAction)
parser.add_argument('--resume_from_oldest', '-rfo',
                    default=False, action=argparse.BooleanOptionalAction)
args = parser.parse_args()

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
checkpoint = args.checkpoint
slack = checkpoint*2  # 2 save only the last 2 checkpoints

if os.path.exists(f'{args.dataset}/fs.list'):
    # read from file with names to save time
    print('reading fs.list')
    with open(f'{args.dataset}/fs.list') as f:
        imagepaths = f.read().splitlines()
else:
    # one off, create file with image names
    print('reading dataset directory')
    fs = os.listdir(args.dataset)
    imagepaths = []
    for fn in fs:
        base, ext = os.path.splitext(fn)
        if ext == '.jpg' or ext == '.png':
            imagepath = os.path.join(args.dataset, fn)
            imagepaths.append(imagepath)
    print('saving in fs.list')
    with open(f'{args.dataset}/fs.list', 'w') as tfile:
        tfile.write('\n'.join(imagepaths))

n_data = len(imagepaths)
print('num traning images:', n_data)
n_iter = n_data // batchsize
print(n_iter, 'iterations,', n_epoch, 'epochs')

model = FastStyleNet()
vgg = VGG()

if not os.path.exists(output):
    os.mkdir(output)

rel_model_dir_path = f'models/{output}/'

if args.auto_resume:
    # gather initmodel, resume and last iteration and epoch from saved files
    args.initmodel, args.resume, start_it, start_ep = check_resume(
        rel_model_dir_path, output, args.resume_from_oldest)
else:
    # manual resume, if specified model and state files will be used, but starting from it and ep 0
    start_it = 0
    start_ep = 0

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

style = vgg.preprocess(np.asarray(Image.open(args.style_image).convert(
    'RGB').resize((image_size, image_size)), dtype=np.float32))
style = xp.asarray(style, dtype=xp.float32)
style_b = xp.zeros((batchsize,) + style.shape, dtype=xp.float32)
for i in range(batchsize):
    style_b[i] = style
feature_s = vgg(Variable(style_b))
gram_s = [gram_matrix(y) for y in feature_s]


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

        indices = range(i * batchsize, (i+1) * batchsize)
        x = xp.zeros((batchsize, 3, image_size, image_size), dtype=xp.float32)
        for j in range(batchsize):
            x[j] = load_image(imagepaths[i*batchsize + j], image_size)

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
        L_feat = lambda_f * \
            F.mean_squared_error(Variable(feature[2].data), feature_hat[2])

        L_style = Variable(xp.zeros((), dtype=np.float32))
        for f, f_hat, g_s in zip(feature, feature_hat, gram_s):
            L_style += lambda_s * \
                F.mean_squared_error(gram_matrix(f_hat), Variable(g_s.data))

        L_tv = lambda_tv * total_variation(y)

        # the 'popping' noise is the difference in resulting stylizations
        # from two images that are very similar. Minimizing it results
        # in a much more stable stylization that can be applied to video.
        # Small changes in the input result in small changes in the output.
        if noise_count:
            L_pop = lambda_noise * F.mean_squared_error(y, noisy_y)
            L = L_feat + L_style + L_tv + L_pop
            print('Epoch {},{}/{}. Total loss: {}. Loss distribution: feat {}, style {}, tv {}, pop {}'
                  .format(epoch, i, n_iter, L.data,
                          L_feat.data/L.data, L_style.data/L.data,
                          L_tv.data/L.data, L_pop.data/L.data))
        else:
            L = L_feat + L_style + L_tv
            print('Epoch {},{}/{}. Total loss: {}. Loss distribution: feat {}, style {}, tv {}'
                  .format(epoch, i, n_iter, L.data,
                          L_feat.data/L.data, L_style.data/L.data,
                          L_tv.data/L.data))

        L.backward()
        O.update()

        if checkpoint > 0 and i % checkpoint == 0:
            if i >= start_it + slack:
                os.remove(
                    f'{rel_model_dir_path}check_{output}_{epoch}_{i-slack}.model')
                os.remove(
                    f'{rel_model_dir_path}check_{output}_{epoch}_{i-slack}.state')
            serializers.save_npz(
                f'{rel_model_dir_path}check_{output}_{epoch}_{i}.model', model)
            serializers.save_npz(
                f'{rel_model_dir_path}check_{output}_{epoch}_{i}.state', O)

    print('save "style.model"')
    serializers.save_npz(f'{rel_model_dir_path}{output}_{epoch}.model', model)
    serializers.save_npz(f'{rel_model_dir_path}{output}_{epoch}.state', O)

    # finished an epoch, restarting from it 0
    start_it = 0

serializers.save_npz(f'{rel_model_dir_path}{output}.model', model)
serializers.save_npz(f'{rel_model_dir_path}{output}.state', O)