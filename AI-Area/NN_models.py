from torch import nn

# Defining the Model

class NueralNetwork(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.flatten = nn.Flatten()
        # layers of the NN
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    