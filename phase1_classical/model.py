import torch
import torch.nn as nn

class ClassicalModel(nn.Module):
    # Initialize layers
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(8, 16)
        self.layer2 = nn.Linear(16, 4)

    # Describing data flow
    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x

if __name__ == "__main__":
    model = ClassicalModel()

    # Random testing sample
    dummy_input = torch.randn(1, 8)

    # Sending it to model
    output = model(dummy_input)

    print(f"Input:  {dummy_input.shape}")
    print(f"Output: {output.shape}")
    print(f"Logits: {output}")