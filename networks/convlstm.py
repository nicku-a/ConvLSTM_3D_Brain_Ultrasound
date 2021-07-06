import torch
from torch import Tensor
from torch import nn

class CLSTM_cell(nn.Module):

    def __init__(self, shape, input_channels, filter_size, num_features, timesteps):
        super(CLSTM_cell, self).__init__()

        self.shape = shape  # H, W
        self.input_channels = input_channels
        self.filter_size = filter_size
        self.num_features = num_features
        self.seq_len = timesteps
        self.padding = (filter_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channels + self.num_features, 4 * self.num_features, self.filter_size, 1, self.padding),
            nn.GroupNorm(4 * self.num_features // 32, 4 * self.num_features))

    def forward(self, inputs=None, hidden_state=None):
        if hidden_state is None:
            hx = torch.zeros(inputs.size(1), self.num_features, self.shape[0], self.shape[1]).cuda()
            cx = torch.zeros(inputs.size(1), self.num_features, self.shape[0], self.shape[1]).cuda()
        else:
            hx, cx = hidden_state
        output_inner = []
        for index in range(self.seq_len):
            if inputs is None:
                x = torch.zeros(hx.size(0), self.input_channels, self.shape[0],  self.shape[1]).cuda()
            else:
                x = inputs[index, ...]

            combined = torch.cat((x, hx), 1)
            gates = self.conv(combined)  # gates: S, num_features*4, H, W
            ingate, forgetgate, cellgate, outgate = torch.split(gates, self.num_features, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * torch.tanh(cy)
            output_inner.append(hy)
            hx = hy
            cx = cy

        return torch.stack(output_inner), (hy, cy)


class ProcessOut(nn.Module):
    def __init__(self, n_f, ts, inputShape):

        super(ProcessOut, self).__init__()
        self.tanh = nn.Tanh()
        self.flat = nn.Flatten()
        self.linear = nn.Linear(n_f*ts*inputShape[0]*inputShape[1], 11)

    def forward(self, x):

        x = self.tanh(x)
        x = self.flat(x)
        y_pred = self.linear(x)
        return y_pred


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class RNN(nn.Module):
    def __init__(self, NUMconvLSTMlayers, NUMconvLSTMfeatures):
        super(RNN, self).__init__()
        self.input_shape = (23,23)
        self.input_channels = 1
        self.filter_size = 3
        self.num_layers = NUMconvLSTMlayers
        self.num_features = NUMconvLSTMfeatures
        self.timesteps = 9

        # self.convIn = nn.Conv3d(self.timesteps, self.timesteps, kernel_size=[1,3,3], stride=1, padding=0)
        
        setattr(self, 'convlstm' + str(1), CLSTM_cell(self.input_shape, self.input_channels, self.filter_size, self.num_features, self.timesteps))
        for layer in range(1, self.num_layers):
          setattr(self, 'convlstm' + str(layer+1), CLSTM_cell(self.input_shape, self.num_features, self.filter_size, self.num_features, self.timesteps))

        self.reshape_output = ProcessOut(self.num_features, self.timesteps, self.input_shape)
        
        self.lambda_1 = Lambda(lambda x: x.view(x.size(0), -1))

    def forward(self, x):
        
        # x = self.convIn(x)
        x = x.transpose(0, 1)  # to S,B,1,23,23

        x, (hy, cy) = getattr(self, 'convlstm' + str(1))(x)
        for i in range(1, self.num_layers):
          x, (hy, cy) = getattr(self, 'convlstm' + str(i+1))(x, (hy, cy))

        x = x.transpose(0, 1)  # to B,S,1,23,23
        reshaped_out = self.reshape_output(x)
        return self.lambda_1(reshaped_out)