import torch
import torch.nn as nn

# Define a simple neural network model with one fully connected layer
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Fully connected (linear) layer: input dim = 10, output dim = 2
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        # Forward pass simply applies the linear layer
        return self.fc(x)

# Instantiate the model
model = MyModel()

# Print parameter names, shapes, and the first few values
for name, param in model.named_parameters():
    print(f"Layer: {name} | Shape: {param.shape} | Values:\n{param[:10]}")
    # Note: param[:10] only makes sense for 1D tensors (like biases)

# Create a sample input tensor (1D tensor of 10 ones)
example_input = torch.as_tensor([1,1,1,1,1,1,1,1,1,1], dtype=torch.float32)

# Use TorchScript to trace the model using the example input
traced_script = torch.jit.trace(model, example_input)

# Save the traced model to disk for later C++ deployment
traced_script.save("../models/full_model_traced.pt")

# Run inference using the original model (not the traced one)
output = model(example_input)

# Print model output
print("output:", output)
