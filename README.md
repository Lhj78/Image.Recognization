This program builds, trains, and tests a Convolutional Neural Network (CNN) to classify handwritten digits (0–9) using the MNIST dataset. MNIST is a collection of 70,000 grayscale images of handwritten numbers, each image being 28×28 pixels.

First, the MNIST dataset is loaded. It provides 60,000 training images and 10,000 test images along with their labels. Since raw pixel values range from 0 to 255, the images are normalized by dividing by 255. This converts pixel values to a range between 0 and 1, which helps the model train faster and more accurately.

MNIST images are originally 28×28 in 2D form. A CNN expects a 3D format (height, width, channels), so each image is reshaped to 28×28×1, where “1” represents the single grayscale channel.

Next, a CNN model is built. The first Conv2D layer has 32 filters of size 3×3 and uses ReLU activation. This layer detects simple patterns such as edges. A MaxPooling layer follows, reducing the image size to make computation faster. A second Conv2D layer with 64 filters learns more detailed features, and another MaxPooling layer reduces the size again.

The output of the convolution layers is then flattened into a 1D vector. A Dense (fully connected) layer with 128 neurons learns complex patterns. A Dropout layer randomly disables 30% of the neurons during training to prevent overfitting. The final Dense layer has 10 neurons with softmax activation, producing a probability distribution over the 10 digit classes (0 to 9).

The model is compiled using the Adam optimizer (which adjusts learning rate automatically), sparse categorical cross-entropy loss (used when labels are integers), and accuracy as the metric.

The model is trained for 10 epochs using 90% of the training data, while 10% is used for validation. During training, it learns patterns of handwritten digits and improves its accuracy.

After training, the model is evaluated on the test dataset to measure how well it performs on unseen images. The test accuracy is printed as a percentage.

The model then predicts the labels for all test images. For the first five test images, the code displays each image along with the predicted digit and the actual true digit. The prediction is taken by selecting the class with the highest probability output from the softmax layer.

Overall, this program demonstrates the entire workflow of a CNN for digit recognition: loading data, preprocessing, building the model, training it, evaluating accuracy, and visually checking predictions.
