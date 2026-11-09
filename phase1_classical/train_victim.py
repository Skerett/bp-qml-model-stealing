import torch
import torch.nn as nn
import joblib
from pathlib import Path
import dataset as myDataset
import model as myModel

trainLoader, testLoader, pca = myDataset.getDataLoaders()
model = myModel.ClassicalModel()
lossFunc = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 5

print(f"------ Training ------")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for batch_x, batch_y in trainLoader:
        # Deleting old gradients
        optimizer.zero_grad()

        # Giving data to model
        outputs = model(batch_x)

        # Looking if model made misstakes
        loss = lossFunc(outputs, batch_y)

        # Loss function for weights editing
        loss.backward()

        # Editing weights 
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(trainLoader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    print(f"------ Testing ------")
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in testLoader:
            outputs = model(batch_x)

            # Selecting the most likely prediction
            predictions = torch.argmax(outputs, dim=1)
            
            # Verification of results
            correct += (predictions == batch_y).sum().item()
            total += len(batch_y)

    accuracy = (correct / total) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

# Path
save_dir = Path(__file__).resolve().parent.parent / "artifacts"
save_dir.mkdir(exist_ok=True)

# Saving model weights
torch.save(model.state_dict(), save_dir / "victim_model.pth")

# Saving PCA
joblib.dump(pca, save_dir / "pca.joblib")
