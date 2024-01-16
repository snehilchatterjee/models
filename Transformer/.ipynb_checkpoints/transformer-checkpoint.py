import torch
from torch import nn

class InputEmbeddings(nn.Module):
    def __init__(self,d_model: int,vocab_size: int):
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embedding=nn.Embedding(vocab_size,d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self,d_model: int,seq_len: int,dropout: float):
        super().__init__()
        self.d_model=d_model
        self.seq_len=seq_len
        self.dropout=nn.Dropout(p=dropout)

        pe=torch.zeros(seq_len,d_model)

        pos=torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)

        div_term=torch.exp(torch.arange(0,512,2).float()*(-math.log(10000.0)/512))
        
        pe[:,0::2]=torch.sin(pos*div_term)
        pe[:,1::2]=torch.cos(pos*div_term)

        pe=pe.unsqueeze(0)

        self.register_buffer('pe',pe)

    def forward(self,x):
        x=x+(self.pe[:,:x.shape[1],:]).requires_grad_(False)
        print(self.pe[:,:x.shape[1],:].shape)
        return self.dropout(x)