import torch
import torch.nn as nn

from .baseRNN import BaseRNN

class EncoderRNN(BaseRNN):
    r"""
    Applies a multi-layer RNN to an input sequence.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        n_layers (int, optional): number of recurrent layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional encodr (defulat False)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        variable_lengths (bool, optional): if use variable length RNN (default: False)

    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is a list of token IDs.
        - **input_lengths** (list of int, optional): list that contains the lengths of sequences
            in the mini-batch, it must be provided when using variable length RNN (default: `None`)
    Outputs: output, hidden
        - **output** (batch, seq_len, hidden_size): tensor containing the encoded features of the input sequence
        - **hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the features in the hidden state `h`

    Examples::

         >>> encoder = EncoderRNN(input_vocab, max_seq_length, hidden_size)
         >>> output, hidden = encoder(input)

    """

    def __init__(self, vocab_size, max_len, hidden_size,
            input_dropout_p=0, dropout_p=0,
            n_layers=1, bidirectional=False, rnn_cell='gru', variable_lengths=False):
        super(EncoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.variable_lengths = variable_lengths
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = self.rnn_cell(hidden_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)

    def init_vectors(self, vectors):
        self.embedding.weight.data = vectors

    def scale_vectors(self, max_val):
        self.embedding.weight.data = (
            self.embedding.weight.data / torch.max(torch.abs(self.embedding.weight.data)) * max_val)

    def normalize_vectors(self, target_norm):
        self.embedding.weight.data = (
          self.embedding.weight.data / torch.norm(self.embedding.weight.data) * target_norm)

    def vectors_stats(self):
        print("max: ", torch.max(self.embedding.weight.data))
        print("min: ", torch.min(self.embedding.weight.data))
        print("norm:", torch.norm(self.embedding.weight.data))

    def forward(self, input_var, input_lengths=None):
        """
        Applies a multi-layer RNN to an input sequence.

        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch

        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        if self.variable_lengths:
            lens, indices = torch.sort(torch.LongTensor(input_lengths).cuda(), 0, True)
            embedded = nn.utils.rnn.pack_padded_sequence(embedded[indices], lens.tolist(), batch_first=True)
        output, hidden = self.rnn(embedded)
        if self.variable_lengths:
            _, _indices = torch.sort(indices, 0)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            output = output[_indices]
        return output, hidden
