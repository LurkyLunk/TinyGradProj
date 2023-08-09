from tinygrad.tensor import Tensor
import tinygrad.nn.optim as optim

class SimpleNN:
    def __init__(self):
        # Initialize weights
        self.l1 = Tensor.uniform(5, 4)  # 5 inputs, 4 neurons in hidden layer 1
        self.l2 = Tensor.uniform(4, 3)  # 4 neurons in hidden layer 1, 3 neurons in hidden layer 2
        self.l3 = Tensor.uniform(3, 1)  # 3 neurons in hidden layer 2, 1 output neuron

    def forward(self, x):
        # Forward pass: Input -> Hidden (ReLU Activation) -> Output
        return x.dot(self.l1).relu().dot(self.l2)

# Create a simple model
model = SimpleNN()

# Define an optimizer
optimizer = optim.SGD([model.l1, model.l2], lr=0.001)

# Mock some data
input_data = Tensor([[1, 2, 3, 4, 5]])
true_output = Tensor([[1]])

# Forward pass
predicted_output = model.forward(input_data)

# Calculate loss (Mean Squared Error)
loss = ((predicted_output - true_output) ** 2).mean()

# Backpropagation
loss.backward()

# Update weights
optimizer.step()
