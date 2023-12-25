import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def train_nn():

    input_size = 784  # 28x28, size of mnist
    hidden_size = 500
    num_classes = 10
    num_epochs = 2
    batch_size = 100
    learning_rate = 0.001
    device = torch.device('cpu')

    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='./data',
                                              train=False,
                                              transform=transforms.ToTensor(),
                                              download=True)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)

    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False)

    class NeuralNet(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(NeuralNet, self).__init__()
            self.input_size = input_size
            self.l1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.l2 = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            out = self.l1(x)
            out = self.relu(out)
            out = self.l2(out)
            return out

    model = NeuralNet(input_size, hidden_size, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    n_total_steps = len(train_dataloader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_dataloader):
            # origin shape: [100, 1, 28, 28]
            # resized: [100, 784]
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step[{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in test_dataloader:
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network on the 10000 test images: {acc} %')


    with torch.no_grad():
        examples = iter(test_dataloader)
        example_data, example_targets = next(examples)
        for i in range(6):
            images = example_data[i][0].reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            plt.subplot(2, 3, i + 1)
            plt.imshow(example_data[i][0], cmap='gray')
            plt.title('Pred: {}'.format(predicted[0]))

        plt.show()




if __name__ == '__main__':
    train_nn()
