import torch
from torch import nn

class InputEmbeddings(nn.Module):
    def __init__(self,d_model: int,vocab_size: int):
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size


        #vocab_size: The size of the vocabulary used
        self.embedding=nn.Embedding(vocab_size,d_model)

    def forward(self, x):
        #x: the id/position of the word in the vocabulary
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self,d_model: int,seq_len: int,dropout: float):
        super().__init__()
        self.d_model=d_model
        self.seq_len=seq_len
        self.dropout=nn.Dropout(p=dropout)

        pe=torch.zeros(seq_len,d_model)

        # Makes a variable with shape (seq_len,1)
        pos=torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)

        # Makes a variable with shape d_model/2 [d_model/2 because even position or odd poisitons are basically of total positions]
        div_term=torch.exp(torch.arange(0,self.d_model,2).float()*(-math.log(10000.0)/self.d_model))
        
        pe[:,0::2]=torch.sin(pos*div_term)
        pe[:,1::2]=torch.cos(pos*div_term)

        pe=pe.unsqueeze(0)

        self.register_buffer('pe',pe)

    def forward(self,x):
        x=x+(self.pe[:,:x.shape[1],:]).requires_grad_(False)
        print(self.pe[:,:x.shape[1],:].shape)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self,eps: float=1e-6):
        super().__init__()
        self.eps=eps
        self.alpha=nn.Parameter(torch.ones(1))
        self.bias=nn.Parameter(torch.zeros(1))

    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        std=x.std(dim=-1,keepdim=True)
        return (self.alpha*(x-mean)/(std+self.eps))+self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self,d_model: int, d_ff: int,droput:float):
        super().__init__()
        self.linear_1=nn.Linear(d_model,d_ff)
        self.dropout=nn.dropout(dropout)
        self.linear_2=nn.Linear(d_ff,d_model)

    def forward(self,x):
        return self.linear_2(self.dropout(self.linear_1(x)))


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self,d_model: int,h: int,dropout: float):
        super().__init__()
        self.d_model=d_model
        self.h=h
        self.dropout=nn.Dropout(p=self.dropout)
        assert d_model%h==0, f"{d_model} is not divisible by {h}"

        self.d_k=d_model//h
        self.w_q=nn.Linear(d_model,d_model) #Wq
        self.w_k=nn.Linear(d_model,d_model) #Wk
        self.w_v=nn.Linear(d_model,d_model) #Wv

        self.w_o=nn.Linear(d_model,d_model) #Wo

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k=query.shape[-1]

        attention_scores=(query@key.transpose(-2,-1))/math.sqrt(d_k)
        if(mask is not None):
            attention_scores.masked_fill_(mask==0,-1e9)
        attention_scores=attention_scores.softmax(dim=-1)
        if(dropout is not None):
            attention_scores=dropout(attention_scores)

        return (attention_scores@value), attention_scores

    def forward(self,q,k,v,mask):
        query=self.w_q(q) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        key=self.w_k(k) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        value=self.w_v(v) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)

        # (Batch, seq_len, d_model) --> (Batch, seq_len, h, d_model) --> (Batch, h , seq_len, d_k)

        query=query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key=query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        value=query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)

        x, self.attention_scores=MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (Batch, h, seq_len, d_k) --> (Batch, seq_len, h, d_k) --> (Batch, seq_len, d_model)
        x=x.transpose(1,2).contiguous().view(x.shape[0],-1,self.h*self.d_k)

        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.norm=LayerNormalization()

    def forward(self,x,sublayer):
        return x+self.dropout(sublayer(self.norm(x)))

