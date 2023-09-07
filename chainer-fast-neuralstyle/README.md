# Chainer implementation of "Perceptual Losses for Real-Time Style Transfer and Super-Resolution"
Fast artistic style transfer by using feed forward network.

**checkout [resize-conv](https://github.com/yusuketomoto/chainer-fast-neuralstyle/tree/resize-conv) which provides better result.**

<img src="https://raw.githubusercontent.com/yusuketomoto/chainer-fast-neuralstyle/master/sample_images/tubingen.jpg" height="200px">

<img src="https://raw.githubusercontent.com/yusuketomoto/chainer-fast-neuralstyle/master/sample_images/style_1.png" height="200px">
<img src="https://raw.githubusercontent.com/yusuketomoto/chainer-fast-neuralstyle/master/sample_images/output_1.jpg" height="200px">

<img src="https://raw.githubusercontent.com/yusuketomoto/chainer-fast-neuralstyle/master/sample_images/style_2.png" height="200px">
<img src="https://raw.githubusercontent.com/yusuketomoto/chainer-fast-neuralstyle/master/sample_images/output_2.jpg" height="200px">

- input image size: 1024x768
- process time(CPU): 17.78sec (Core i7-5930K)
- process time(GPU): 0.994sec (GPU TitanX)


## Requirement
- [Chainer](https://github.com/pfnet/chainer)
```
$ pip install chainer
```

## Prerequisite
Download VGG16 model and convert it into smaller file so that we use only the convolutional layers which are 10% of the entire model.
```
sh setup_model.sh
```

## Train
Need to train one image transformation network model per one style target.
According to the paper, the models are trained on the [Microsoft COCO dataset](http://mscoco.org/dataset/#download).
```
python train.py -s <style_image_path> -d <training_dataset_path> -g <use_gpu ? gpu_id : -1>
```

Arguments:
| Parameter | Shortcut | Default | Type | Description |
|---|---|---|---|---|
| --gpu | -g | -1 | int | GPU selector. Anything > 0 selects that GPU if available. -1 means CPU is going to be used. |
| --dataset | -d | ./coco/train2014 | str | Path to the dataset folder. |
| --style_image | -s | REQUIRED | str | Path to the image from which the style will be extracted. |
| --batchsize | -b | 1 | int | Size of the batches of images used to train the model. |
| --output | -o | None | str | Output model name path without extension. |
| --auto_resume | -a | // | // | If -a is present the program will automatically try to start the training from the existing model and state file, associated to the output variable. Training will restart from the last completed iteration.  |
| <div style="width:160px">--resume_from_oldest</div> | -rso | // | // | If present start from the oldest saved checkpoint. |
| --initmodel | -i | None | str | Initialize model manually with given file. |
| --resume | -r | None | str | Initialize optimizer manyally with given file.  |
| --lr | -r | 1e-3 | float | Learning rate for the Adam optimizer. |
| --lambda_tv | -ltv | 1e-6 | float | Weight of total variation regularization (to be set between 10e-4 and 10e-6). |
| --lambda_feat | -lf | 1.0 | float | Feature loss weight. |
| --lambda_style | -ls | 5.0 | float | Style loss weight. |
| --lambda_noise | -ls | 1000.0 | float | Training weight of the popping induced by noise |
| --noise | -n | 30 | int | Range of noise for popping reduction. |
| --noisecount | -nc | 1000 | int | Number of pixels to modify with noise. |
| --epoch | -e | 2 | int | Number of epochs to train for. |
| --checkpoint | -c | 0 | int | If > 0, each time the current iteration % checkpoint == 0, the model and state will be saved as checkpoints. |
| --checkpoint_number | -cn | 2 | int | Number of checkpoints to keep saved during execution. |
| --image_size | -is | 256 | int | Size to which the images are resized to. |

## Generate
```
python generate.py <input_image_path> -m <model_path> -o <output_image_path> -g <use_gpu ? gpu_id : -1>
```

This repo has pretrained models as an example.

- example:
```
python generate.py sample_images/tubingen.jpg -m models/composition.model -o sample_images/output.jpg
```
or
```
python generate.py sample_images/tubingen.jpg -m models/seurat.model -o sample_images/output.jpg
```

#### Transfer only style but not color (**--keep_colors option**)
`python generate.py <input_image_path> -m <model_path> -o <output_image_path> -g <use_gpu ? gpu_id : -1> --keep_colors`

<img src="https://raw.githubusercontent.com/yusuketomoto/chainer-fast-neuralstyle/master/sample_images/output_1.jpg" height="200px">
<img src="https://raw.githubusercontent.com/yusuketomoto/chainer-fast-neuralstyle/master/sample_images/output_keep_colors_1.jpg" height="200px">

<img src="https://raw.githubusercontent.com/yusuketomoto/chainer-fast-neuralstyle/master/sample_images/output_2.jpg" height="200px">
<img src="https://raw.githubusercontent.com/yusuketomoto/chainer-fast-neuralstyle/master/sample_images/output_keep_colors_2.jpg" height="200px">


## A collection of pre-trained models
Fashizzle Dizzle created pre-trained models collection repository, [chainer-fast-neuralstyle-models](https://github.com/gafr/chainer-fast-neuralstyle-models). You can find a variety of models.

## Difference from paper
- Convolution kernel size 4 instead of 3.
- Training with batchsize(n>=2) causes unstable result.

## No Backward Compatibility
##### Jul. 19, 2016
This version is not compatible with the previous versions. You can't use models trained by the previous implementation. Sorry for the inconvenience!

## License
MIT

## Reference
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://arxiv.org/abs/1603.08155)

Codes written in this repository based on following nice works, thanks to the author.

- [chainer-gogh](https://github.com/mattya/chainer-gogh.git) Chainer implementation of neural-style. I heavily referenced it.
- [chainer-cifar10](https://github.com/mitmul/chainer-cifar10) Residual block implementation is referred.
