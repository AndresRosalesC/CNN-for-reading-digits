# Step 1: Load MNIST Train Dataset

import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
from torch.autograd import Variable

train_dataset = torchvision.datasets.MNIST(root='./mnist',
                            train=True,
                            transform=torchvision.transforms.ToTensor(),
                            download=True)

test_dataset = torchvision.datasets.MNIST(root='./mnist',
                           train=False,
                           transform=torchvision.transforms.ToTensor())

# Step 2: Make Dataset Iterable

batch_size = 100
n_iters = 3000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = int(num_epochs)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# image visulization (you can try to check what other images look like)
images, label = next(iter(test_loader))
# images, label = next(iter(train_loader))
images_example = torchvision.utils.make_grid(images)
images_example = images_example.numpy().transpose(1,2,0)
# mean = [0.1,0.2,0.5]
# std = [0.5,0.5,0.5]
images_example = images_example #* std + mean
plt.imshow(images_example )
plt.show()

# Step 3: Create Model Class

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()

        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()

        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Fully connected 1 (readout)
        self.fc1 = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)

        # Max pool 1
        out = self.maxpool1(out)

        # Convolution 2
        out = self.cnn2(out)
        out = self.relu2(out)

        # Max pool 2
        out = self.maxpool2(out)

        # Resize
        # Original size: (100, 32, 7, 7)
        # out.size(0): 100
        # New out size: (100, 32*7*7)
        out = out.view(out.size(0), -1)

        # Linear function (readout)
        out = self.fc1(out)

        return out
    
# Step 4: Instantiate Model Class
    
print(torch.cuda.is_available())
model = CNNModel()

if torch.cuda.is_available():
  model = model.cuda()

# Step 5: Instantiate Loss Class
  
criterion = nn.CrossEntropyLoss()

# Step 6: Instantiate Optimizer Class

learning_rate = 0.01

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(model)

print(model.parameters())

print(len(list(model.parameters())))

# Convolution 1: 16 Kernels
print(list(model.parameters())[0].size())

# Convolution 1 Bias: 16 Kernels
print(list(model.parameters())[1].size())

# Convolution 2: 32 Kernels with depth = 16
print(list(model.parameters())[2].size())

# Convolution 2 Bias: 32 Kernels with depth = 16
print(list(model.parameters())[3].size())

# Fully Connected Layer 1
print(list(model.parameters())[4].size())

# Fully Connected Layer Bias
print(list(model.parameters())[5].size())


# Step 7: Train Model

iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        # Load images
        images = images.requires_grad_()

        # Use GPU if it is available
        if torch.cuda.is_available():
          images = images.cuda()
          labels = labels.cuda()

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward pass to get output/logits
        outputs = model(images)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        iter += 1

        if iter % 500 == 0:
            # Calculate Accuracy
            correct = 0
            total = 0
            # Iterate through test dataset
            for images, labels in test_loader:


                # Load images
                images = images.requires_grad_()

                # Use GPU if it is available
                if torch.cuda.is_available():
                  images = images.cuda()
                  labels = labels.cuda()

                # Forward pass only to get logits/output
                outputs = model(images)

                # Get predictions from the maximum value
                _, predicted = torch.max(outputs.data, 1)


                # Total number of labels
                total += labels.size(0)

                # Total correct predictions
                correct += (predicted == labels).sum()

            accuracy = 100 * correct / total

            # image visulization: try to compare the classification results with the test image, you can try to distract other more images
            images_test = torchvision.utils.make_grid(images)

            # Send images to CPU if we used GPU before
            if torch.cuda.is_available():
              images_test = images_test.cpu()

            images_test = images_test.numpy().transpose(1,2,0)

            plt.imshow(images_test)
            plt.show()
            # Print output information
            print('Iteration: {}. Outputs: {}.  Loss: {}. Accuracy: {}. Total: {}. Correct: {}'.format(iter, predicted, loss.item(), accuracy, total, correct))

# Step 8: Test Model with your own handwriting
            
import numpy as np
import cv2

# Load an image file
image = cv2.imread('ownsample.jpg')

# Use an interpolation method
resized_image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
plt.imshow(resized_image)
plt.show()
# Convert the image to grayscale
gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
# Apply adaptive thresholding
_, gray_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# Invert the color
gray_image = 255 - gray_image


# Change model to evaluate mode
model.eval()

# Disable gradient calculation
with torch.no_grad():
    # Visualize the image
    test_image = torch.tensor(gray_image)
    test_image = test_image[None, None, :, :]
    visual_image = torchvision.utils.make_grid(test_image)
    visual_image = visual_image.numpy().transpose(1,2,0)
    plt.imshow(visual_image)
    plt.show()

    # Predict the number
    test_image_tensor = test_image.type(torch.FloatTensor)
    if torch.cuda.is_available():
      test_image_tensor = test_image_tensor.cuda()

    test_output = model(test_image_tensor)

    # Get predictions from the maximum value
    _, predicted = torch.max(test_output.data, 1)

    # Print the number
    print("The identified number is %d"%predicted)

    #out_data = model(data)



    