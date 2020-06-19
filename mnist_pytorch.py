import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Conv2d, Linear, Flatten
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

# Download the MNIST Data and create dataloader
transform = transforms.Compose([transforms.ToTensor()])
xy_train = datasets.MNIST('./', download=True, train=True, transform=transform)
xy_test = datasets.MNIST('./', download=True, train=False, transform=transform)

train_ds = DataLoader(xy_train, batch_size=32, shuffle=True)
test_ds = DataLoader(xy_test, batch_size=32, shuffle=True)


# Model Definition
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.flatten = Flatten()
        self.d1 = Linear(21632, 128)
        self.d2 = Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.flatten(x)
        x = F.relu(self.d1(x))
        x = self.d2(x)
        output = F.log_softmax(x, dim=1)
        return output


# Instantiate Model, Optimizer, Loss
model = MyModel()
optimizer = Adam(model.parameters())
loss_object = CrossEntropyLoss(reduction='sum')

for epoch in range(2):
    # Train
    model.train()
    train_loss = 0
    train_n = 0
    for image, labels in train_ds:
        predictions = model(image).squeeze()
        loss = loss_object(predictions, labels)
        train_loss += loss.item()
        train_n += labels.shape[0]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    train_loss /= train_n

    # Evaluate Test accuracy
    model.eval()
    test_loss = 0
    test_accuracy =0
    test_n = 0
    for batch_num, (test_images, test_labels) in enumerate(test_ds):
        test_predictions = model(test_images)
        t_loss = loss_object(test_predictions, test_labels)

        test_loss += t_loss.item()
        test_n += test_labels.shape[0]
        test_accuracy += (torch.argmax(test_predictions.data, 1) == test_labels).float().sum().item()

    test_accuracy /= test_n
    test_loss /= test_n

    # Calculate Loss / accuracy
    template = 'Epoch {}, Loss: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss,
                          test_loss,
                          test_accuracy * 100))