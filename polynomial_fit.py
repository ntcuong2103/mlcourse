from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def buildPolynomialModel(order=4):
    inputs = Input(shape=(order + 1, ))

    outputs = Dense(1, use_bias=False)(inputs)

    model = Model(inputs, outputs)

    model.compile(loss='mse', optimizer=SGD(0.01))

    model.summary()

    return model

def buildDataset(order = 4):
    x = np.random.uniform(0, 1, (50,1))
    inputs = []
    for i in range(0, order + 1):
        inputs.append(x ** i)

    inputs = np.concatenate(inputs, axis=-1)
    targets = np.sin(1 + x ** 2) + np.random.normal(0, 0.03, (50, 1))

    print(inputs.shape)
    print(targets.shape)
    return inputs, targets

def getOutput(weights):
    x_test = np.linspace(0, 1, 300)
    x_test = np.expand_dims(x_test, axis=1)

    x_feature = []
    for i in range(len(weights)):
        x_feature.append(x_test ** i)
    x_feature = np.concatenate(x_feature, axis=-1)

    y_test = np.sum(x_feature * np.expand_dims(weights, axis=0), axis=1)
    return y_test

def plotOutput(inputs, targets):
    x_test = np.linspace(0, 1, 300)
    x_test = np.expand_dims(x_test, axis=1)

    y_function = np.sin(1 + x_test ** 2)

    import matplotlib.pyplot as plt

    plt.ion()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(inputs[:, 1], targets[:, 0], 'o')
    line, = ax.plot(x_test, y_function)
    line, = ax.plot(x_test, y_function)

    ax.set_ylim(0.8, 1.05)

    return fig, line

def main():
    order = 2
    inputs, targets = buildDataset(order)
    # plotDataset(inputs, targets)

    model = buildPolynomialModel(order)

    fig, line = plotOutput(inputs, targets)
    fig.canvas.flush_events()

    num_epochs = 200
    # training models
    for epoch in range(num_epochs):
        model.fit(inputs, targets, batch_size=1, verbose=2)
        weights = model.get_layer('dense_1').get_weights()
        weights = np.squeeze(weights)
        line.set_ydata(getOutput(weights))
        fig.canvas.draw()
        fig.canvas.flush_events()

if __name__ == '__main__':
    main()