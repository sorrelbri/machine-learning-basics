import numpy as np

def tanH(x: float):
    # tanH activation function: f(x) = 2 / (1 + e^(-2x)) -1
    return 2 / (1 + np.exp(-2 * x)) -1

def deriv_tanH(x: float):
    # Derivative of tanH: f'(x) = f(x) * (1 - f(x))
    fx = tanH(x)
    return 1 - fx * fx

def mse_loss(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length.
    return ((y_true - y_pred) ** 2).mean()

class TwoNeuron:
    def __init__(self, weight1: float, weight2: float, bias: float):
        self.w1 = weight1
        self.w2 = weight2
        self.b = bias

    def feedforward(self, x):
        # print(f'x{x}')
        # print(f'x[0]{x[0]}')
        out = tanH(self.w1 * x[0] + self.w2 * x[1] + self.b)
        # print(f'out ${self, out}')
        return out

    def feed_no_activation(self, x):
        out = self.w1 * x[0] + self.w2 * x[1] + self.b
        return out

    def train(self, learn_rate, d_L_d_ypred, d_ypred_d_neuron, x):
        d_neuron = deriv_tanH(self.feed_no_activation(x))
        d_neuron_d_w1 = x[0] * d_neuron
        d_neuron_d_w2 = x[1] * d_neuron
        d_neuron_d_b = d_neuron
        self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_neuron * d_neuron_d_w1
        self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_neuron * d_neuron_d_w2
        self.b -= learn_rate * d_L_d_ypred * d_ypred_d_neuron * d_neuron_d_b

class NewNetwork:
    def __init__(self):
        self.h1 = TwoNeuron(np.random.normal(), np.random.normal(), np.random.normal())
        self.h2 = TwoNeuron(np.random.normal(), np.random.normal(), np.random.normal())
        self.o1 = TwoNeuron(np.random.normal(), np.random.normal(), np.random.normal())

    def feedforward(self, x):
        h1 = self.h1.feedforward(x)
        h2 = self.h2.feedforward(x)
        o1 = self.o1.feedforward(np.array([h1, h2]))
        return o1

    def train(self, data, all_y_trues):
        learn_rate = 0.1
        epochs = 2000
        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                d_L_d_ypred = -2 * (y_true - self.feedforward(x))
                d_ypred_d_h1 = self.o1.w1 * deriv_tanH(self.o1.feed_no_activation(x))
                d_ypred_d_h2 = self.o1.w2 * deriv_tanH(self.o1.feed_no_activation(x))
                
                self.h1.train(learn_rate, d_L_d_ypred, d_ypred_d_h1, x)
                self.h2.train(learn_rate, d_L_d_ypred, d_ypred_d_h2, x)
                self.o1.train(learn_rate, d_L_d_ypred, 1, np.array([self.h1.feedforward(x), self.h2.feedforward(x)]))
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))

    def test(self, data):
        for x in data:
            guess = self.feedforward(x)
            print(f'Guess for {x}: {guess}')

# Define dataset
data = np.array([
    [6, 0],
    [0, 0],
    [-4, 1],
    [2, 12],
    [-1, -7],
    [1, -12],
])
all_y_trues = np.array([ # return index of first positive or -1
    0,
    -1,
    1,
    0,
    -1,
    0,
])

test = np.array([
    [1, -8],
    [-1, 4],
    [-3, -5],
    [5, -1],
    [10, 2],
    [14, 3],
    [2, 5],
])

new_net = NewNetwork()
new_net.train(data, all_y_trues)

new_net.test(test)