# Fashion_GAN

**Overview**

The Fashion GAN project implements a Generative Adversarial Network (GAN) to generate fashion items based on the Fashion MNIST dataset. This project explores the capabilities of GANs in generating high-quality images of clothing items.

**Dataset**

This project uses the Fashion MNIST dataset, which consists of 70,000 grayscale images of 10 different types of clothing items (e.g., T-shirts, dresses, sneakers, etc.). The dataset is divided into a training set of 60,000 images and a test set of 10,000 images.


**Model Architecture**

The GAN consists of two main components:

Generator: This model generates new images based on random noise.

Discriminator: This model distinguishes between real and generated images.

**Generator**

Takes random noise as input and generates an image.

Uses several layers of transposed convolution, batch normalization, and activation functions.

**Discriminator**

Takes an image as input and outputs a probability of whether the image is real or fake.

Uses convolutional layers followed by Leaky ReLU activation and dropout for regularization

**Training**

The model has been trained for 200 epochs in this implementation. While this initial training has yielded some promising results, further training for additional epochs is expected to improve the quality of the generated images significantly.

The training process involves the following steps:

Generate random noise and use the generator to produce fake images.

Train the discriminator with both real images (from the Fashion MNIST dataset) and fake images (from the generator).

Train the generator using the discriminator's feedback to improve the quality of generated images.

The process is repeated for a specified number of epochs.

The training curve of the GAN highlighting both the discriminator and the generator loss is shown below:

![GAN TRAINING CURVE](https://github.com/user-attachments/assets/d2a381c7-f70c-4637-b787-eaa8212fd98b)

We can see that the training of the GAN becomes stable as we train for more and more epochs.
