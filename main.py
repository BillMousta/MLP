import torch
from torch import nn
from torch.utils.data import DataLoader
import MLP
import TakeData
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import Data
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from math import sqrt
import seaborn as sns
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
import Visualizations
import numpy as np
import time

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    correct = 0
    pred_training_labels = []
    train_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)

        for label in pred.argmax(1):
            num = label.numpy()
            pred_training_labels .append(num)

        loss = loss_fn(pred, y)
        train_loss += loss.item()

        # Backpropagation
        # sets the gradients to zero before we start backpropagation.
        # This is a necessary step as PyTorch accumulates the gradients from
        # the backward passes from the previous epochs.

        optimizer.zero_grad()
        # computes the gradients
        loss.backward()
        # updates the weights accordingly
        optimizer.step()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        train_loss /= num_batches

    return pred_training_labels, correct/size, train_loss


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    labels = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            for label in pred.argmax(1):
                labels.append(label.numpy())
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct, labels


"""
Turn data into train and test data
"""
def processing_data(name,visual):
    d = TakeData
    # LOAD DATA from 2000 up to 2021
    index_data = d.get_data(name, start='2000-01-01', end='2021-01-01', interval='1d')
    X = index_data.drop(['Label'], axis=1)

    # Dimension reduction with PCA
    # pca = PCA(n_components=2)
    # pca.fit(X)
    # X = pca.transform(X)

    y = index_data['Label']

    # for time series dont need to shuffle
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    # visualizations
    # visual.visualize_train_test_data(X_train,X_test,'30%')
    # visual.visualize_indicators(index_data, 1000)
    # visual.visualize_FT(index_data)
    # visual.visualize_class_distribution(y_train.to_numpy(), 'Class Distribution for Training Set')
    # visual.visualize_class_distribution(y_test.to_numpy(), 'Class Distribution for Testing Set')

    #  Mean removal
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Machine Learning algorithms for classification problems
    # K - NN and Nearest Centroid
    k_nearest_neighbors(X_train, y_train, X_test, y_test)
    nearest_centroid(X_train, y_train, X_test, y_test)

    # turn into an object with data an label
    train_data = Data.Data(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_data = Data.Data(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    batch_size = 32
    tr_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    te_dataloader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    return tr_dataloader, te_dataloader

"""
K nearest neighbor algorithm using k = 3, to find accuracy
for stock data test
"""
def k_nearest_neighbors(X_train, y_train, X_test, y_test):
    start_time = time.time()
    print("----- 3 Nearest Neighbors -----")
    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train, y_train)
    train_preds = knn_model.predict(X_train)
    mse = mean_squared_error(y_train, train_preds)
    rmse = sqrt(mse)
    print("For training data")
    print("The square root error is: {:.4f}".format(rmse))
    print("Accuracy: {:.2f}%".format(metrics.accuracy_score(y_train,train_preds)*100))
    test_preds = knn_model.predict(X_test)
    mse = mean_squared_error(y_test, test_preds)
    rmse = sqrt(mse)

    print("For testing data")
    print("The square root error for testing data is: {:.4f}".format(rmse))
    print("Accuracy: {:.2f}%".format(metrics.accuracy_score(y_test, test_preds) * 100))

    print("--- %s seconds ---" % (time.time() - start_time))
    cmap = sns.cubehelix_palette(as_cmap=True)
    # f, ax = plt.subplots()
    # plt.title('3-NN Visualization')
    # points = ax.scatter(X_test[:, 0], X_test[:, 1], c = test_preds, s = 50, cmap = cmap)
    # f.colorbar(points)
    # plt.show()

"""
Nearest neighbor algorithm, to find accuracy
for stock data test
"""
def nearest_centroid(X_train, y_train, X_test, y_test):
    start_time = time.time()
    print("----- Nearest Centroids -----")
    centroid_model = NearestCentroid(metric='euclidean', shrink_threshold=None)
    centroid_model.fit(X_train, y_train)
    train_preds = centroid_model.predict(X_train)
    mse = mean_squared_error(y_train, train_preds)
    rmse = sqrt(mse)
    print("For training data")
    print("The square root error is: {:.4f}".format(rmse))
    print("Accuracy: {:.2f}%".format(metrics.accuracy_score(y_train, train_preds) * 100))

    test_preds = centroid_model.predict(X_test)
    mse = mean_squared_error(y_test, test_preds)
    rmse = sqrt(mse)
    print("For testing data")
    print("The square root error for testing data is: {:.4f}".format(rmse))
    print("Accuracy: {:.2f}%".format(metrics.accuracy_score(y_test, test_preds) * 100))
    print("--- %s seconds ---" % (time.time() - start_time))


"""
Training our Neural Network for different values of 
learning rate, with other parameters fixed
"""
def testing_learning_rates(train_dataloader, test_dataloader, device, visual):
    learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    data_acc = []
    data_loss = []
    total_labels = []
    for lr in learning_rates:
        model = MLP.MultiLayerPerceptron(9, 256).to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(),lr=lr, weight_decay=1e-4, momentum=0.9)
        epochs = 50
        # the below lists help us to visualize the results
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []
        labels_train = []
        labels_test = []
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            labels_train, acc, train_loss = train(train_dataloader, model, loss_fn, optimizer, device)
            train_losses.append(train_loss)
            train_accuracies.append(acc)
            test_loss, accuracy, labels_test = test(test_dataloader, model, loss_fn, device)
            test_losses.append(test_loss)
            test_accuracies.append(accuracy)

        total_labels.append(labels_test)
        data_acc.append(test_accuracies)
        data_loss.append(test_losses)
    visual.visualize_many_results(data_acc, 'Accuracy vs. No of epochs for different learning rates', 'accuracy', learning_rates)
    visual.visualize_many_results(data_loss, 'Loss vs. No of epochs for different learning rates', 'Loss', learning_rates)

    # print label distribution for last epoch
    for i, label in enumerate(np.array(total_labels)):
        print("Learning rate = {} ".format(learning_rates[i]))
        print("Number of Sells: {}".format(label[label == 1].size))
        print("Number of Holds: {}".format(label[label == 0].size))
        print("Number of Buy: {}".format(label[label == 2].size))


"""
Run multi layer perceptron
"""
def mlp(train_dataloader, test_dataloader, stocks_names, num_features):
    visual = Visualizations
    torch.manual_seed(213)
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    start_time = time.time()
    model = MLP.MultiLayerPerceptron(num_features, 256).to(device)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)

    # testing_learning_rates(train_dataloader,test_dataloader,device,visual)

    epochs = 200
    # the below lists help us to visualize the results
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    labels_train = []
    labels_test = []
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        labels_train, acc, train_loss = train(train_dataloader, model, loss_fn, optimizer, device)
        train_losses.append(train_loss)
        train_accuracies.append(acc)
        test_loss, accuracy, labels_test = test(test_dataloader, model, loss_fn, device)
        test_losses.append(test_loss)
        test_accuracies.append(accuracy)

    print("--- %s seconds ---" % (time.time() - start_time))
    visual.visualize_results_of_MLP(test_losses, 'Loss vs. No of epochs', 'Loss')
    visual.visualize_results_of_MLP(test_accuracies, 'Accuracy vs. No of epochs', 'accuracy')

    visual.validation(train_accuracies, test_accuracies)

    visual.visualize_class_distribution(np.array(labels_train), 'Train')
    visual.visualize_class_distribution(np.array(labels_test), 'Predict')

    # Testing different stocks
    if stocks_names != '':
        d = TakeData
        for i in range(len(stocks_names)):
            print('For ' + stocks_names[i] + " we have:")
            data = d.get_data(stocks_names[i], start="2019-01-01", end='2021-01-01', interval='1d')
            X = data.drop(['Label'], axis=1)
            y = data['Label']
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            test_data_stock = Data.Data(torch.FloatTensor(X), torch.LongTensor(y))
            test_dataloader = DataLoader(dataset=test_data_stock, batch_size=1, shuffle=False)
            test(test_dataloader, model, loss_fn, device)

    return labels_test


"""
Crash example for sp500 for 2008(train) and 2020(test)
----need to change the value from TakeData.py in line 28 from True to False----
"""
def crash():
    d = TakeData
    sp500_train = d.get_data('SPY', start='2008-01-01', end='2010-01-01', interval='1d')
    sp500_test = d.get_data('SPY', start='2020-02-01', end='2020-05-01', interval='1d')

    y_train = sp500_train['Label']
    X_train = sp500_train.drop(['Label'], axis=1)
    y_test = sp500_test['Label']
    X_test = sp500_test.drop(['Label'], axis=1)

    # Mean removal
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Machine learning Algorithms
    # k_nearest_neighbors(X_train, y_train, X_test, y_test)
    # nearest_centroid(X_train, y_train, X_test, y_test)

    train_data = Data.Data(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_data = Data.Data(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    batch_size = 2
    tr_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    te_dataloader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    labels = mlp(tr_dataloader, te_dataloader, '', 2)
    sp500_test['Label'] = labels
    Visualizations.visualize_trend(sp500_test)
    return tr_dataloader, te_dataloader

def run():
    stocks_names = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'FB']
    # index_names = ['BTC-USD']
    # stocks_names = ['ETH-USD']
    index_names = ['SPY']
    visual = Visualizations
    torch.manual_seed(213)
    train_dataloader, test_dataloader = processing_data(index_names[0], visual)
    mlp(train_dataloader, test_dataloader, stocks_names, 9)

    # crash()

    print("Done!")


if __name__ == '__main__':
    run()
