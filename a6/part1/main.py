import ray
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


ray.init()

print("Ray version :", ray.__version__)


#dataset
url = "https://gist.github.com/curran/a08a1080b88344b0c8a7/raw/iris.csv"
df = pd.read_csv(url)

X = df.drop('species', axis=1).values.astype(np.float32)
y = df['species'].values

# species labels into integers (0, 1, 2)
le = LabelEncoder()
y = le.fit_transform(y)

# Split into training 80% and testing 20%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

class IrisNet(nn.Module):
    def __init__(self, input_size=4, hidden1=5, hidden2=4, output_size=3):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out

model = IrisNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

@ray.remote
def train_model(model_state_dict, epochs, train_data, test_data):
    model = IrisNet()
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    X_train_tensor, y_train_tensor = train_data
    X_test_tensor, y_test_tensor = test_data

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True) # batch size of 16
    
    history = {"loss": [], "accuracy": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        # avg loss for the epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # accuracy on the test set
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == y_test_tensor).sum().item()
            accuracy = correct / len(y_test_tensor)
        
        history["loss"].append(epoch_loss)
        history["accuracy"].append(accuracy)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")
    return model.state_dict(), history

epochs = 50
trained_state_dict, history = ray.get(
    train_model.remote(model.state_dict(), epochs, 
                       (X_train_tensor, y_train_tensor), 
                       (X_test_tensor, y_test_tensor))
)

model.load_state_dict(trained_state_dict)

model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == y_test_tensor).sum().item()
    test_accuracy = correct / len(y_test_tensor)
    final_loss = criterion(outputs, y_test_tensor).item()

print(f"Final Test Accuracy: {test_accuracy:.4f}, Final Test Loss: {final_loss:.4f}")

with open("pytorch_results.txt", "w") as f:
    f.write(f"Final Test Loss: {final_loss:.4f}\n")
    f.write(f"Final Test Accuracy: {test_accuracy:.4f}\n")
    f.write("Epoch\tLoss\tAccuracy\n")
    for epoch in range(epochs):
        f.write(f"{epoch+1}\t{history['loss'][epoch]:.4f}\t{history['accuracy'][epoch]:.4f}\n")

ray.shutdown()
