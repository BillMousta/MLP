from torch import nn

class MultiLayerPerceptron(nn.Module):

    def __init__(self, size_features, size_nodes):
        super(MultiLayerPerceptron, self).__init__()
        # We initialize the nn.Flatten layer to convert each 2D 28x28 image
        # into a contiguous array of 784 pixel values
        # the minibatch dimension (at dim=0) is maintained).
        # self.flatten = nn.Flatten()
        self.layer = nn.Sequential(
            nn.Linear(size_features, size_nodes),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(size_nodes, size_nodes),
            nn.ReLU(),
            nn.Dropout(0.25),
            # nn.Linear(size_nodes, size_nodes),
            # nn.ReLU(),
            # nn.Dropout(0.25),
            nn.Linear(size_nodes, 3),


        )

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.layer(x)
        return logits



