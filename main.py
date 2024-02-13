import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the Encoder
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
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
        # hidden shape: (1, batch_size, hidden_size)
        # encoder_outputs shape: (seq_len, batch_size, hidden_size)
        # Transpose encoder_outputs to (batch_size, seq_len, hidden_size)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        # Transpose hidden to (batch_size, hidden_size, 1)
        hidden = hidden.transpose(0, 1).transpose(1, 2)
        # Perform batch matrix multiplication
        attn_energies = torch.bmm(encoder_outputs, hidden).squeeze(2)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

# Define the Decoder with Attention
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=None):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length or 10  # Provide a default max_length if not specified

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = Attn('dot', hidden_size)
        self.gru = nn.GRU(self.hidden_size * 2, self.hidden_size)
        self.out = nn.Linear(self.hidden_size * 2, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = F.dropout(embedded, self.dropout_p)

        attn_weights = self.attn(hidden, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        rnn_input = torch.cat((embedded, context), 2)

        output, hidden = self.gru(rnn_input, hidden)
        output = self.out(torch.cat((output[0], context[0]), 1))

        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

# Parameters for the model
input_size = 256  # Size of the input vocabulary
output_size = 256  # Size of the output vocabulary
hidden_size = 256  # Size of the hidden layers

# Initialize the Encoder and Decoder
encoder = EncoderRNN(input_size, hidden_size)
decoder = AttnDecoderRNN(hidden_size, output_size)

# Example input and output sequences (batch size = 1)
# These should be properly preprocessed and converted to indices from a vocabulary
input_seq = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long).view(-1, 1)
target_seq = torch.tensor([9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=torch.long).view(-1, 1)

# Initial hidden state for the encoder
encoder_hidden = encoder.initHidden()

# Encoder outputs
encoder_outputs = torch.zeros(input_seq.size(0), 1, encoder.hidden_size)  # Adjusted to 3D tensor
for i in range(input_seq.size(0)):
    encoder_output, encoder_hidden = encoder(input_seq[i], encoder_hidden)
    encoder_outputs[i] = encoder_output[0, 0]

# Decoder with Attention
decoder_input = torch.tensor([[0]], dtype=torch.long)  # SOS_token
decoder_hidden = encoder_hidden  # Use the last hidden state from the encoder to start the decoder

decoded_words = []
decoder_attentions = torch.zeros(target_seq.size(0), input_seq.size(0))

for di in range(target_seq.size(0)):
    decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
    decoder_attentions[di] = decoder_attention.data
    topv, topi = decoder_output.data.topk(1)
    if topi.item() == 1:  # Assuming EOS_token is 1
        decoded_words.append('<EOS>')
        break
    else:
        decoded_words.append(topi.item())

    decoder_input = topi.squeeze().detach()  # detach from history as input

# The decoded_words list now contains the predicted output sequence

print("Next number in the sequence:", decoded_words)