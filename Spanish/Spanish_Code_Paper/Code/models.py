import torch.nn as nn
import torch.nn.functional as F
import torch

class Dense(nn.Module):
    def __init__(self, model_name, label_size=6):
        super().__init__()
        #Input Dim = Batch * 12 * Seq_Len * 768
        
        if model_name.find('BASE'):
            num_layers = 12
            feature_dim = 768
        elif model_name.find('LARGE'):
            num_layers = 24
            feature_dim = 1024
           
        hidden_dim = 256

        #Averaging over 12 layers 
        self.aggr = nn.Conv1d(in_channels=num_layers, out_channels=1, kernel_size=1, bias=False)
        
        #Input Dim = Batch * Seq_Len * 768
        self.cnn  = nn.Conv1d(in_channels=feature_dim, out_channels=hidden_dim, kernel_size=1)
        self.cnn2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1)
        self.dropout = nn.Dropout(0.2)#not used yet

        self.linear = nn.Linear(in_features = hidden_dim, out_features = label_size)


    def forward(self, x, lengths, device):
        """
        padded_x: (B,T) padded LongTensor
        """

        batch_size, n_layers, seq_len, n_features = x.size(0), x.size(1), x.size(2), x.size(3)
        
        #Take average of 12 layers
        x = torch.flatten(x, start_dim=2)
        x = self.aggr(x)
        x = torch.reshape(x, (batch_size, seq_len, n_features))

        #Pass through CNN
        x = x.transpose(1,2) #now dimension is batch * n_features * seq_len
        x = F.relu(self.cnn(x))
        x = F.relu(self.cnn2(x))
        x = x.transpose(1,2) #now dimension is batch * seq_len * n_features

        #Do global average over time sequence
        global_avg = torch.tensor([]).to(device)
        for i in range(batch_size):
            mean_vector = torch.mean(x[i,:lengths[i],:], dim = 0)
            mean_vector = mean_vector.reshape(1,-1)
            global_avg = torch.cat((global_avg, mean_vector))

        logits = self.linear(global_avg)

        return logits


class ICASSP3CNN(nn.Module):
    def __init__(self, vocab_size, dims = 12, embed_size=128, hidden_size=512, num_lstm_layers = 2, 
                 bidirectional = False, label_size=6):
        super().__init__()
        self.n_layers = num_lstm_layers 
        self.hidden = hidden_size
        self.bidirectional = bidirectional
        
        self.aggr = nn.Conv1d(in_channels=dims, out_channels=1, kernel_size=1)
        
        self.embed = nn.Linear(in_features = vocab_size, out_features = embed_size)

        self.cnn  = nn.Conv1d(embed_size, embed_size, kernel_size=3, padding=1)
        self.cnn2 = nn.Conv1d(embed_size, embed_size, kernel_size=5, padding=2)
        self.cnn3 = nn.Conv1d(embed_size, embed_size, kernel_size=7, padding=3)

        self.batchnorm = nn.BatchNorm1d(3 * embed_size)

        self.lstm = nn.LSTM(input_size = 3 * embed_size, 
                            hidden_size = hidden_size, 
                            num_layers = num_lstm_layers, 
                            bidirectional = bidirectional)

        self.linear = nn.Linear(in_features = 2 * hidden_size if bidirectional else hidden_size, 
                                out_features = label_size)


    def forward(self, x, lengths):
        """
        padded_x: (B,T) padded LongTensor
        """
        n, d, b, t = x.size(0), x.size(1), x.size(2), x.size(3)
        x = torch.flatten(x, start_dim=2)
        input = self.aggr(x)
        input = torch.reshape(input, (n, b, t))
        input = self.embed(input)

        batch_size = input.size(0)
        input = input.transpose(1,2)    # (B,T,H) -> (B,H,T)

        cnn_output = torch.cat([self.cnn(input), self.cnn2(input), self.cnn3(input)], dim=1)

        input = F.relu(self.batchnorm(cnn_output))

        input = input.transpose(1,2)

        pack_tensor = nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=True, enforce_sorted=False)
        _, (hn, cn) = self.lstm(pack_tensor)

        if self.bidirectional:
            h_n = hn.view(self.n_layers, 2, batch_size, self.hidden)
            h_n = torch.cat([ h_n[-1, 0,:], h_n[-1,1,:] ], dim = 1)
        else:
            h_n = hn[-1]

        logits = self.linear(h_n)

        return logits
