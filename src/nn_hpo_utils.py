import torch.nn as nn


class NNBlockWithNormalization(nn.Module):

    def __init__(self, in_features, out_features):
        super(NNBlockWithNormalization, self).__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.norm = nn.BatchNorm1d(num_features=out_features)

    def forward(self, x):
        output = self.linear(x)
        output = self.norm(output)
        return output


class TunableFeedForwardNN(nn.Module):

    def __init__(self, in_features, num_layers, num_neurons, activation, drop_out=0, batch_normalization=False):
        assert len(num_neurons) == num_layers
        super(TunableFeedForwardNN, self).__init__()

        if batch_normalization:
            self.input_layer = NNBlockWithNormalization(in_features=in_features, out_features=num_neurons[0])
        else:
            self.input_layer = nn.Linear(in_features=in_features, out_features=num_neurons[0])
        self.hidden_layers = []
        for layer_n in range(num_layers - 1):
            if batch_normalization:
                h_layer = NNBlockWithNormalization(in_features=num_neurons[layer_n],
                                                   out_features=num_neurons[layer_n + 1])
            else:
                h_layer = nn.Linear(in_features=num_neurons[layer_n], out_features=num_neurons[layer_n + 1])
            self.hidden_layers.append(h_layer)
        self.output_layer = nn.Linear(in_features=num_neurons[-1], out_features=1)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "selu":
            self.activation = nn.SELU()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError("Unsupported activation")
        self.sigmoid = nn.Sigmoid()
        self.drop_out = nn.Dropout(p=drop_out)

    def forward(self, x):
        output = self.input_layer(x)
        output = self.activation(output)
        for layer in self.hidden_layers:
            output = layer(output)
            output = self.activation(output)
            output = self.drop_out(output)
        output = self.output_layer(output)
        output = self.sigmoid(output)
        return output

