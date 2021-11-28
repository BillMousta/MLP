import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
"""
Plot some of indicators from dataset for SP500
"""
def visualize_indicators(data, start):
    data = data.iloc[-start:, :]
    x_ = range(3, data.shape[0])
    x_ = list(data.index)
    sns.set_theme()
    plt.figure(figsize=(16, 12), dpi=150)
    plt.plot(data['MA 21'], label='MA 21', color='g', linestyle='--')
    plt.plot(data['Adj Close'], label='Closing Price', color='b', linewidth=2)
    plt.plot(data['upper_band'], label='Upper Band', color='c', linewidth=2)
    plt.plot(data['lower_band'], label='Lower Band', color='c', linewidth=2)
    plt.fill_between(x_, data['lower_band'], data['upper_band'], alpha=0.2, color='r')
    plt.title('Technical indicators for S&P 500')
    plt.ylabel('USD')
    plt.xlabel('Date')
    plt.legend()
    # plt.savefig('indicators.png')
    plt.show()


"""
Plot Fourier transform of signal data
"""
def visualize_FT(data):
    sns.set_theme()
    plt.figure(figsize=(12, 8), dpi=200)
    plt.plot(data['Adj Close']['2020':], label='Original')
    plt.plot(data['FT 10000 comp']['2020':], label='Filtered',linewidth=2, color='r')
    plt.title('S&P 500 close prices and Fourier transform')
    plt.ylabel('USD')
    plt.xlabel('Date')
    plt.legend()
    # plt.savefig('denoise.png')
    plt.show()


"""
Visualize splited data into train and test data
 """
def visualize_train_test_data(train_data, test_data, size):
    sns.set_theme()
    plt.figure(figsize=(14,10), dpi=200)
    plt.title('Test data for '+size+' of train data for S&P 500')
    plt.plot(train_data['Adj Close'], label='Train data')
    plt.plot(test_data['Adj Close'], label='Test data')
    max_value = test_data['Adj Close'].max()
    if train_data['Adj Close'].max() > max_value:
        max_value = train_data['Adj Close']
    plt.vlines(test_data.index[0], 0 , max_value, linestyles='--',colors='black', label='Train/Test data cut-off')
    plt.ylabel('USD')
    plt.xlabel('Date')
    plt.legend()
    # plt.savefig('test_train_data.png')
    plt.show()


def visualize_correlation(data):
    # fig, ax = plt.subplots()
    # plt.title('Correlation between S&P500 and Apple')
    # ax.scatter(stock1, stock2, alpha=0.5)
    # ax.set_xlabel('S&P500', fontsize=15)
    # ax.set_ylabel('Apple', fontsize=15)

    # plt.show()
    # sns.set_theme(style='whitegrid')
    sns.set_theme()
    plt.title('Correlation between S&P500 and Apple')
    sns.scatterplot(data=data, x='S&P500',y='APPLE', alpha=0.5)
    plt.show()

    # sns.set(xscale="log", yscale="log")
    # g.ax.xaxis.grid(True, "minor", linewidth=.25)
    # g.ax.yaxis.grid(True, "minor", linewidth=.25)
    # g.despine(left=True, bottom=True)


def visualize_trend(data):
    sns.set_theme()
    fig = plt.figure(figsize=(15, 5))
    plt.title('Up trend for S&P 500')
    plt.plot(data['Adj Close'], color='r', lw=2.)
    plt.plot(data['Adj Close'], '^', markersize=10, color='m', label='buying signal', markevery=(data['Label']==2))
    plt.plot(data['Adj Close'], 'v', markersize=10, color='k', label='selling\shorting signal', markevery=(data['Label']==1))
    plt.ylabel('USD')
    plt.xlabel('Date')
    plt.legend(loc='upper left')
    # plt.savefig('Buy selling signals')
    plt.show()

"""
Shows the distribution of class 1 -> Sell, Hold -> 0, Buy -> 2
"""
def visualize_class_distribution(data, title):
    # data = data.to_numpy()
    sns.set_theme()
    dict = {'Buy': data[data == 2].size, 'Sell': data[data == 1].size, 'Hold': data[data == 0].size}
    fig = plt.figure(figsize=(12,8))

    plt.title(title)
    plt.bar('Sell', height=dict['Sell'], color='r', label='Sell')
    plt.bar('Hold', height=dict['Hold'], color='b', label='Hold')
    plt.bar('Buy', height=dict['Buy'], color='g', label='Buy')
    plt.xlabel('Movement')
    plt.ylabel('No of Labels')
    plt.legend()
    # plt.savefig(title)
    plt.show()


def visualize_many_results(data, title, ylabel, lg):
    sns.set_theme()
    fig = plt.figure(figsize=(12,8), dpi=200)
    for i,item in enumerate(data):
        plt.plot(item, label='lr = ' + str(lg[i]))
    plt.xlabel('No. of epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

"""
Visualization of results for all epochs
"""
def visualize_results_of_MLP(data, title, ylabel):
    # for i, res in enumerate(data):
    #     plt.plot(res, '-x', label=legend[i])
    sns.set_theme()
    plt.plot(data, '-x')
    # plt.plot(data)
    plt.xlabel('No. of epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def validation(train, test):
    sns.set_theme()
    mean_train_score = np.array(train)
    mean_test_score = np.array(test)

    plt.plot(mean_train_score, label="Training Score", color='b')
    plt.plot(mean_test_score,label="Cross Validation Score", color='g')

    # Creating the plot
    plt.title("Validation Curve")
    plt.xlabel("No. of epochs")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()