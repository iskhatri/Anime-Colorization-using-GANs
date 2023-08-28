# Anime Colorization using GANsAnime Image Colorization using Conditional GANs
This repository contains the implementation of conditional Generative Adversarial Networks (cGANs) for anime image colorization using the PyTorch framework.

## Overview
In this project, I have implemented a conditional GAN architecture to perform anime image colorization. The goal is to automatically generate colorized versions of black-and-white anime images. The project includes the following key components and techniques:

U-Net Generator: I have employed a U-Net generator architecture to capture high-frequency details and perform effective image-to-image translation.

PatchGAN Discriminator: A PatchGAN discriminator is used to distinguish between real color images and generated colorized images, enabling more fine-grained analysis of image patches.

Alternating Gradient Descent: The training process involves alternating gradient descent on the discriminator and generator. I have used the Adam optimizer to update the model parameters.

Loss Functions: I employ Mean Absolute Error (MAE) loss and adversarial loss functions to enhance both the colorization quality and the alignment of the generated image distribution with the real image distribution.

## Dataset
The model is trained on a dataset consisting of 7,800 paired anime images. Additionally, I have validated the model's performance on a separate validation set of 3,500 images.

Link:
https://www.kaggle.com/datasets/ktaebum/anime-sketch-colorization-pair


Getting Started

1. Clone this repository to your local machine.
   
2. Install the required dependencies
!pip install pytorch
!pip install pytorchsummary

3. Prepare your own dataset dividing data points into training and validation dataset.

4. Run the Anime_colorization.ipynb to train the cGAN model. Use the trained model to perform colorization on new images using the colorize.py script.
Results.After training, the model is capable of generating colorized versions of anime images. The generated images are assessed based on their visual quality and similarity to real color images.


Reference :

Dataset taken from Kaggle:
https://www.kaggle.com/datasets/ktaebum/anime-sketch-colorization-pair

