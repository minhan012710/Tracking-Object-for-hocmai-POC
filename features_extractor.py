import torch
import torchvision.models as models
import cv2
from torchvision import transforms
from PIL import Image
import numpy as np

# Load a pre-trained ResNet-18 model
resnet18 = models.resnet18(pretrained=True)

# Set the model to evaluation mode
resnet18.eval()

# Define a preprocessing transform to be applied to input images
preprocess = transforms.Compose(
    [
        transforms.Resize(200),
        transforms.CenterCrop(140),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# # Load an input image
# image = cv2.imread('1810_1.png')


def embedding(image):
    image = Image.fromarray(image)

    # Apply the preprocessing transform to the input image
    input_tensor = preprocess(image)

    # Add a batch dimension to the input tensor
    input_batch = input_tensor.unsqueeze(0)

    # Use the ResNet model to extract embeddings from the input image
    with torch.no_grad():
        embeddings = resnet18(input_batch)
    return np.array(embeddings)
