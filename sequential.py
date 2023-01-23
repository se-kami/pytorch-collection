class SeqNN(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 num_hidden_layers=10,
                 hidden_layer_size=128)
    super.__init__()
    self.deep_nn = nn.Sequential()
    # all keys need to be distinct
    for i in range(num_hidden_layers):
        self.deep_nn.add_module(f'ff{i}', nn.Linear(input_size, hidden_layer_size))
        self.deep_nn.add_module(f'activation_{i}', nn.ReLU())
        input_size = hidden_layer_size
    self.deep_nn.add_module(f'classifer', nn.Linear(hidden_layer_size, output_size))

    def forward(self, x):
        return self.deep_nn(x)
