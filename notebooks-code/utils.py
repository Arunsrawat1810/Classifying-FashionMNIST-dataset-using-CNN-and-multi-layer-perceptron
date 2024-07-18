import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms


class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, activation_fn, dropout_rate=0.0, use_batch_norm=False):
        super(MLP, self).__init__()
        layers = []
        current_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(current_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(activation_fn())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))
            current_size = hidden_size
        
        layers.append(nn.Linear(current_size, output_size))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

def mlp_train(train_loader, test_loader, input_size, hidden_layers, output_size, activation_fn, dropout_rate=0.0, use_batch_norm=False, num_epochs=10, learning_rate=0.001):
    # Initialize the MLP model
    mlp_model = MLP(input_size, hidden_layers, output_size, activation_fn, dropout_rate, use_batch_norm)

    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mlp_model.to(device)

    # loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp_model.parameters(), lr=learning_rate)

    # Lists to store training history
    train_loss_history = []
    test_loss_history = []
    test_accuracy_history = []

    # Training loop
    for epoch in range(num_epochs):
        mlp_model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = mlp_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Computing average training loss
        train_loss = running_loss / len(train_loader)
        train_loss_history.append(train_loss)

        # Evaluate on test set
        mlp_model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.view(images.size(0), -1).to(device)
                labels = labels.to(device)
                outputs = mlp_model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Computing average test loss and accuracy
        test_loss = test_loss / len(test_loader)
        test_accuracy = correct / total
        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)

        # Printing epoch statistics
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    
    return mlp_model, train_loss_history, test_loss_history, test_accuracy_history


def mlp_apply(model, test_dataset, test_indexes):
    # Create a Subset of the test dataset from  the indexes we choose in mlp file
    test_subset = Subset(test_dataset, test_indexes)
    
    # Create a DataLoader for the test subset
    test_loader = DataLoader(test_subset, batch_size=10, shuffle=False)
    
    # Set model to evaluation mode
    model.eval()
    
    # Get the first batch of images and labels from test_loader
    images, labels = next(iter(test_loader))
    images, labels = images.to(next(model.parameters()).device), labels.to(next(model.parameters()).device)
    
    # Perform inference
    outputs = model(images.view(-1, 28 * 28))
    _, predicted = torch.max(outputs, 1)
    
    # Display images and predictions
    fig, axes = plt.subplots(1, 10, figsize=(15, 1.5))
    for i in range(10):
        axes[i].imshow(images[i].cpu().numpy().squeeze(), cmap='gray')
        axes[i].set_title(f'True: {labels[i].item()}\nPred: {predicted[i].item()}')
        axes[i].axis('off')
    plt.show()
