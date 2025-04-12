# Image-classification-model-using-pytorch
FashionMNIST Classification with PyTorch This project uses a simple neural network to classify FashionMNIST images. It includes data loading, model training, evaluation, saving/loading, and visualization—perfect for beginners learning deep learning with PyTorch.
# 🧠 FashionMNIST Classification with PyTorch

This project demonstrates how to build a basic image classification model using PyTorch and the FashionMNIST dataset. It includes all the steps from data loading to training, evaluation, model saving, and visualization.

## 📦 Dataset

The [FashionMNIST dataset](https://github.com/zalandoresearch/fashion-mnist) consists of 28x28 grayscale images of 10 fashion categories:

- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

## 🚀 Features

- Uses `torchvision.datasets` to load FashionMNIST
- Custom neural network built with `torch.nn`
- Training and evaluation with accuracy and loss tracking
- Saves and loads model using `torch.save()` and `torch.load()`
- Visualizes predictions and dataset samples with `matplotlib`

## 🧪 Requirements

- Python 3.7+
- PyTorch
- torchvision
- matplotlib
- numpy

Install requirements:

```bash
        pip install torch torchvision matplotlib numpy
##📈 Training the Model
-Run the training script:

```bash
    python main.py
The script trains the model for 10 epochs and prints out the training loss and test accuracy at each epoch.

##💾 Saving & Loading the Model
-The trained model is saved as model.pth and can be reloaded for predictions:

```python
model.load_state_dict(torch.load("model.pth"))
model.eval()
##📊 Visualization
-The script includes a section to visualize:
-A prediction on a single test image
-A 4x4 grid of training samples with labels

##📝 License
-This project is open-source and available under the MIT License.
