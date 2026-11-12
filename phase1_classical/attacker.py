import torchvision
import torch
import requests
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from common import ApiMode

# Constants
API_URL = "http://localhost:8000/predict"
QUERY_BUDGET = 2000
API_MODE = ApiMode.CONFIDENCE_SCORE
trainSize = int(QUERY_BUDGET * 0.8)

# Surrogate model
class SurrogateModel(nn.Module):
    # Initialize layers
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(28*28, 32)
        self.layer2 = nn.Linear(32, 4)

    # Describing data flow
    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x

# Download input data
inputImages = torchvision.datasets.MNIST(root="./data", train=False, download=True).data
queryImages = inputImages[:QUERY_BUDGET]

# Open session and prepare lists for stolen goods
session = requests.Session()
stolenImages = []
stolenAnswers = []

# Sending requests to victim
for image in queryImages:
    imgFlatten = image.float().flatten().tolist()
    response = session.post(f"{API_URL}?mode={API_MODE.value}", json={"image": imgFlatten})
    prediction = response.json()["prediction"]
    stolenImages.append(imgFlatten)
    stolenAnswers.append(prediction)

# Transforming to tensor
target_dtype = torch.long if API_MODE == ApiMode.HARD_LABEL else torch.float32
stolenImages = torch.tensor(stolenImages, dtype=torch.float32)
stolenAnswers = torch.tensor(stolenAnswers, dtype=target_dtype)

# Normalise images
stolenImagesNorm = stolenImages.float() / 255.0

# Joining stolen images with predictions
stolenDataset = TensorDataset(stolenImagesNorm[:trainSize], stolenAnswers[:trainSize])
testDataset = TensorDataset(stolenImagesNorm[trainSize:], stolenAnswers[trainSize:])

# Making dataloader
stolenDataloader = DataLoader(stolenDataset, batch_size=32, shuffle=True)
testingDataLoader = DataLoader(testDataset, batch_size=32, shuffle=False)

# Model training
surrogateModel = SurrogateModel()
lossFunc = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(surrogateModel.parameters(), lr=0.001)
epochs = 10

print(f"------ Surrogate training ------")
for epoch in range(epochs):
    surrogateModel.train()
    running_loss = 0.0

    for batch_x, batch_y in stolenDataloader:
        # Deleting old gradients
        optimizer.zero_grad()

        # Giving data to model
        outputs = surrogateModel(batch_x)

        # Looking if model made misstakes
        loss = lossFunc(outputs, batch_y)

        # Loss function for weights editing
        loss.backward()

        # Editing weights 
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(stolenDataloader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    print(f"------ Surrogate testing ------")
    surrogateModel.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in testingDataLoader:
            outputs = surrogateModel(batch_x)

            # Selecting the most likely prediction
            predictions = torch.argmax(outputs, dim=1)

            # Victim prediction
            victimPrediction = torch.argmax(batch_y, dim=1) if batch_y.dim() > 1 else batch_y
            
            # Verification of results
            correct += (predictions == victimPrediction).sum().item()
            total += len(batch_y)

    fidelity = (correct / total) * 100
    print(f"Fidelity: {fidelity:.2f}%")