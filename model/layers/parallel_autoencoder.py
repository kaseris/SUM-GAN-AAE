import torch
import torch.nn as nn


class ParallelEncoder(nn.Module):
    '''Parallel autoencoder'''
    def __init__(self, input_dim, num_hidden, num_layers, bidirectional=False,
                 dropout=0.2):
        super(ParallelEncoder, self).__init__()
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=num_hidden,
                            num_layers=num_layers,
                            dropout=dropout,
                            bidirectional=bidirectional)

    def forward(self, x):
        """
        @args:
            x [sequence_len, 1, 500]
                The original linearly compressed features

        @returns:
            e [sequence_len, 1, hidden_size]
                The encoded output
            h [num_layers=2, 1, hidden_size]
                The hidden state
            c [num_layers=2, 1, hidden_size]
                The cell state
        """
        x, (h, c) = self.lstm(x)
        return x, (h, c)


class ParallelDecoder(nn.Module):
    """Parallel Decoder"""
    def __init__(self, output_dim, num_hidden, num_layers, bidirectional=False,
                 dropout=0.2):
        super(ParallelDecoder, self).__init__()
        self.output_dim = output_dim
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_size=num_hidden,
                            hidden_size=num_hidden,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            dropout=dropout)
        self.leaky_relu = nn.LeakyReLU()

        self.output = nn.Linear(num_hidden, output_dim)

    def forward(self, x, seq_len, init_hidden):
        """
        @args:
            x [1, 1, hidden_size]
                The final hidden state of the last layer of the encoder

        @returns:
            o [seq_len, 1, 500]
                The reconstructed sequence
        """
        h, c = init_hidden

        decoded = []
        for _ in range(seq_len):
            out, (h, c) = self.lstm(x, (h, c))
            out = self.output(out)
            out = self.leaky_relu(out)
            decoded.append(out)
        decoded.reverse()
        return torch.stack(decoded).squeeze(1)


class ParallelAutoencoder(nn.Module):
    """Parallel Autoencoder abstraction"""
    def __init__(self, input_dim, num_hidden, num_layers, bidirectional=False, dropout=0.2):
        super(ParallelAutoencoder, self).__init__()
        self.enc = ParallelEncoder(input_dim=input_dim,
                                   num_hidden=num_hidden,
                                   num_layers=num_layers,
                                   bidirectional=bidirectional,
                                   dropout=dropout)

        self.dec = ParallelDecoder(output_dim=input_dim,
                                   num_hidden=num_hidden,
                                   num_layers=num_layers,
                                   bidirectional=bidirectional,
                                   dropout=dropout)

    def forward(self, x):
        """
        @args:
            x [seq_len, 1, feature_dim]
                The input features.

        @returns:
            reconstructed_features [seq_len, 1, feature_dim]
                The reconstructed features
            final_hidden [1, 1, num_hidden]
                The final hidden state of the encoder
        """
        encoded_output, (h, c) = self.enc(x)
        final_hidden = encoded_output[-1, :, :].unsqueeze(0)
        reconstructed_features = self.dec(final_hidden, encoded_output.size(0), (h, c))
        return reconstructed_features, final_hidden

if __name__ == "__main__":
    a = [[[1.0, 2.0]], [[3., 4.]], [[5., 6.]]]
    a = torch.tensor(torch.randn(2000, 1, 500)) # [3, 1, 2]
    enc = ParallelEncoder(a.size(2), 500, 2)
    result, (h, c) = enc(a)
    print("="*100)
    print("Values of h:")
    print(h)
    print("="*100)
    print("Values of c:")
    print(c)
    print("="*100)
    print("Values of LSTM output:")
    print(result)
    # result [3, 1, 10]
    # h [num_layers, 1, 10]
    # c [num_layers, 1, 10]
    # print(final_h)
    print("=" * 100)
    print("Size of LSTM Output: {}".format(result[-1, :, :].unsqueeze(0).size())) # [1, 1, 10]
    dec = ParallelDecoder(500, 500, 2)
    decoded_feats = dec(result[-1, :, :].unsqueeze(0), result.size(0), (h, c))
    print("=" * 100)
    print("Decoded features:")
    print(decoded_feats.size())

    criterion = torch.nn.MSELoss()
    loss = criterion(a, decoded_feats)
    print("=" * 100)
    print("Loss: {}".format(loss.data))
    print("="*100)
    print("Testing the parallel autoencoder")
    ae = ParallelAutoencoder(a.size(2), 500, 2)
    reconstructed, _ = ae(a)
    loss = criterion(a, reconstructed)
    print("="*100)
    print("Loss between original and reconstructed: {}".format(loss.data))