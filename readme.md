# Handwritten Image Generation Network

## Abstract
With the advancement of deep learning and generative modeling techniques, remarkable progress has been made in the field of image-to-image translation. Particularly, models such as Pix2Pix, cGAN (conditional GAN), and DCGAN (Deep Convolution GAN) have demonstrated outstanding performance in image generation and transformation tasks. In this project, we applied these models to the task of Korean character image transformation. Specifically, we leveraged the initial consonant (초성), medial vowel (중성), and final consonant (종성) of Korean characters as conditioning factors to enhance the performance of our model. We utilized the 나눔 고딕 font, which closely resembles regular handwriting, as the input and employed the paired dataset from Pix2Pix for Korean character transformations. Moreover, we adopted the structure of DCGAN for both the Generator and Discriminator networks to enhance stability and efficiency.

## Architecture
<img width="307" alt="image" src="https://github.com/Tim3s/font_generation/assets/84570397/a26702cc-f26b-4a7a-9ca4-cd85700b2b60">

As shown above, we had composed our network with DCGAN & pix2pix baseline. We attatched embeddings for including rich representation about "음소". This embeddings lead the data to contain geometry information as condition.

## Requirement
Our network requires pandas, numpy, torch, matplotlib, and jamo package for train & test the model.

## How to use

### Preprocessing
