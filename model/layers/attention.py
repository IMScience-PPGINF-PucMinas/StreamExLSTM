# attention.py
import torch
import torch.nn as nn

class sLSTM(nn.Module):
    def __init__(self, input_size=1024, hidden_dim=512, conv_channels=128, dropout=0.5, num_groups=16):
        super(sLSTM, self).__init__()
        self.conv = nn.Conv1d(input_size, conv_channels, kernel_size=3, padding=1)
        self.ln = nn.LayerNorm(conv_channels)
        self.lstm = nn.LSTM(conv_channels, hidden_dim, num_layers=1, batch_first=True, dropout=dropout)
        self.gn = nn.GroupNorm(num_groups, hidden_dim)
        self.i_gate = nn.Linear(hidden_dim, hidden_dim)
        self.f_gate = nn.Linear(hidden_dim, hidden_dim)
        self.o_gate = nn.Linear(hidden_dim, hidden_dim)
        self.c_gate = nn.Linear(hidden_dim, hidden_dim)
        self.right_linear = nn.Linear(hidden_dim, hidden_dim)
        self.left_linear = nn.Linear(hidden_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, input_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_size, input_size)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # [batch_size, channels, seq_len]
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        
        # [batch_size, seq_len, conv_channels]
        x = x.permute(0, 2, 1)
        
        # [batch_size, seq_len, channels]
        x = self.ln(x)
        
        # [batch_size, seq_len, input_size]
        out, _ = self.lstm(x)
        
        # [batch_size, hidden_dim, seq_len]
        out = out.permute(0, 2, 1)
        out = self.gn(out)
        
        # [batch_size, seq_len, hidden_dim]
        out = out.permute(0, 2, 1)
        
        i_gate = torch.sigmoid(self.i_gate(out))
        f_gate = torch.sigmoid(self.f_gate(out))
        o_gate = torch.sigmoid(self.o_gate(out))
        c_gate = torch.tanh(self.c_gate(out))
        
        out = f_gate * out + i_gate * c_gate
        out = self.right_linear(out) + self.left_linear(out)
        
        # [batch_size, seq_len, input_size]
        out = self.proj(out)
        out = self.dropout(out)
        
        # [batch_size, seq_len, input_size]
        out = self.fc(out)
        return out


class mLSTM(nn.Module):
    def __init__(self, input_size=1024, hidden_dim=512, num_layers=2, dropout=0.5):
        super(mLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.wq = nn.Linear(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.i_gate = nn.Linear(hidden_dim, hidden_dim)
        self.f_gate = nn.Linear(hidden_dim, hidden_dim)
        self.o_gate = nn.Linear(hidden_dim, hidden_dim)
        self.c_gate = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, input_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        wq = self.wq(out)
        out = out + wq
        i_gate = torch.sigmoid(self.i_gate(out))
        f_gate = torch.sigmoid(self.f_gate(out))
        o_gate = torch.sigmoid(self.o_gate(out))
        c_gate = torch.tanh(self.c_gate(out))
        out = f_gate * out + i_gate * c_gate
        out = self.drop(out)
        out = self.fc(out)
        return out

if __name__ == '__main__':
    pass