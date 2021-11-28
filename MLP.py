from torch import nn

class MultiLayerPerceptron(nn.Module):

    def __init__(self, size_features, size_nodes):
        super(MultiLayerPerceptron, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(size_features, size_nodes),
            nn.Dropout(0.25),
            nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(size_nodes, size_nodes),
            nn.ReLU(),
            # nn.Tanh(),
            nn.Dropout(0.25),
            # nn.Linear(size_nodes, size_nodes),
            # nn.ReLU(),
            # nn.Dropout(0.25),
            nn.Linear(size_nodes, 3),
        )

    def forward(self, x):
        logits = self.layer(x)
        return logits



