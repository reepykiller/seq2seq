import torch
import torch.nn as nn
import torch.optim as optim

# Define the Encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

# Define the Attention mechanism
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

    def forward(self, hidden, encoder_outputs):
        attn_energies = torch.bmm(encoder_outputs.transpose(0, 1), hidden.transpose(0, 1).transpose(1, 2)).squeeze(2)
        return torch.nn.functional.softmax(attn_energies, dim=1).unsqueeze(1)

# Define the Decoder with Attention
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=10):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = Attn('dot', hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = torch.nn.functional.dropout(embedded, self.dropout_p)

        attn_weights = self.attn(hidden, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        rnn_input = torch.cat((embedded, context), 2)

        output, hidden = self.gru(rnn_input, hidden)
        output = self.out(output[0])

        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

# Parameters for the model
input_size = 10
output_size = 10
hidden_size = 16
max_length = 10

# Initialize the Encoder and Decoder
encoder = EncoderRNN(input_size, hidden_size)
decoder = AttnDecoderRNN(hidden_size, output_size, max_length=max_length)

# Example input and output sequences (batch size = 1)
input_seq = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long).view(-1, 1)
target_seq = torch.tensor([9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=torch.long).view(-1, 1)

# Initial hidden state for the encoder
encoder_hidden = encoder.initHidden()

# Encoder outputs
encoder_outputs = torch.zeros(max_length, encoder.hidden_size)
for i in range(input_seq.size(0)):
    encoder_output, encoder_hidden = encoder(input_seq[i], encoder_hidden)
    encoder_outputs[i] += encoder_output[0, 0]

# Decoder with Attention
decoder_input = torch. = decoder(decoder_input, decoder_hidden, encoder_outputs)
    topv, topi = decoder_output.topk(1)
    decoder_input = topi.squeeze().detach()  # detach from history as input

    if decoder_input.item() == 1:  # EOS_token
        break
