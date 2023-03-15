# DEff-GAN

Official implementation of the paper [*"DEff-GAN: Diverse Attribute Transfer for Few-Shot Image Synthesis "* by  Rajiv Kumar and G. Sivakumar]
Arxiv: https://arxiv.org/pdf/2302.14533v1.pdf) Scitepress: https://www.scitepress.org/Papers/2023/117996/117996.pdf).
# Installation
- python 3.5 or above 
- pytorch 1.1.0 or above

```
pip install -r requirements.txt
```
# Installation using Conda
The code was tested on an environment that can be imported using the environment.yml file.
```
conda env create -f environment.yml
```
However, there are lots of extra packages that may consume more disk space.
# Unconditional Generation
To train a model with the default parameters from our paper run:
```
python train.py
```
Training one model should take about 20-25 minutes on an NVIDIA GeForce GTX 1080Ti.

### Modify Learning Rate Scaling and Number of Trained Stages
To affect sample diversity and image quality we recomment playing around with the learning rate scaling (default is `0.1`) and the number of trained stages (default is `6`).
This can be especially helpful if the images are more complex (use a higher learning rate scaling) or you want to train on images with higher resolution (use more stages).
For example, increasing the learning rate scaling will mean that lower stages are trained with a higher learning rate and can, therefore, learn a more faithful model of the original image.
For example, observe the difference in generated images of the Colusseum if the model is trained with a learning rate scale of `0.1` or `0.5`:

Training on more stages can help with images that exhibit a large global structure that should stay the same. 

### Results
The output is saved to `TrainedModels/`.

### Sample More Images
To sample more images from a trained model run:
This will use the model to generate `num_samples` images in the default as well as scaled resolutions.
The results will be saved in a folder `Evaluation` in the `model_dir`.

### Unconditional Generation (Arbitrary Sizes)
The default unconditional image generation is geared to also induce diversity at the edges of generated images.
When generating images of arbitrary sizes (especially larger) this often break the image layout.
Therefore, we also provide the option where we change the upsampling and noise addition slightly to improve results when we want to use a model to generate images of arbitrary sizes.
To train a model more suited for image generation of arbitrary sizes run:

# Additional Data
The folder `User-Studies` contains the raw images we used to conduct our user study.

# Acknowledgements
This code implementation borrows heavily from [implementation](https://github.com/tohinz/ConSinGAN) of the [ConSinGAN paper](https://openaccess.thecvf.com/content/WACV2021/papers/Hinz_Improved_Techniques_for_Training_Single-Image_GANs_WACV_2021_paper.pdf).

# Citation
If you found this code useful please consider citing:

```
@conference{visapp23,
author={Rajiv Kumar and G. Sivakumar},
title={DEff-GAN: Diverse Attribute Transfer for Few-Shot Image Synthesis},
booktitle={Proceedings of the 18th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications - Volume 5: VISAPP,},
year={2023},
pages={870-877},
publisher={SciTePress},
organization={INSTICC},
doi={10.5220/0011799600003417},
isbn={978-989-758-634-7},
}
```
# https://www.scitepress.org/PublishedPapers/2023/117996/
