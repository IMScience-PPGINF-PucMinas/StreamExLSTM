import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import mLSTM, sLSTM

class xLSTM(nn.Module):
    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")

    def __init__(self, input_size, output_size, num_segments, hidden_dim=512, num_layers=2, dropout=0.5, lstm_type='mLSTM'):
        super(xLSTM, self).__init__()
        self.slstm = sLSTM(input_size, hidden_dim, dropout=dropout)
        self.mlstm = mLSTM(input_size, hidden_dim, num_layers=num_layers, dropout=dropout)

        self.conv = nn.Conv1d(input_size, hidden_dim, kernel_size=1)
        
        self.attn_linear = nn.Linear(input_size, input_size)
        self.attn_softmax = nn.Softmax(dim=-1)
        
        self.fc = nn.Linear(input_size, output_size)
        self.fc_output = nn.Linear(output_size, 1)
        self.num_segments = num_segments

    def forward(self, x):
        x_slstm = self.mlstm(x)
        x_mlstm = self.mlstm(x_slstm)

        x_combined = (x_slstm + x_mlstm) / 2
        
        output = self.fc(x_combined)
        output = self.fc_output(output)
        output = output.view(output.size(0), -1)
        
        attn_weights = torch.zeros(output.size(0), output.size(1), output.size(1))
        
        return output, attn_weights

if __name__ == '__main__':
    pass