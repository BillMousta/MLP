"""
Create a data object with data and label
"""
class Data:
    def __init__(self, X_data, y_data):
       self.dataset = X_data
       self.labels = y_data

    def __getitem__(self, index):
        return self.dataset[index], self.labels[index]

    def __len__(self):
        return len(self.dataset)
