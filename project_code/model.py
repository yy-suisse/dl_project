import torch
from torch import nn
from torch.nn import functional as F
from util import config
import math

c = config()

class PositionalEncoding(nn.Module):

    def __init__(self, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, c.d_model, 2) * (-math.log(10000.0) / c.d_model))
        pe = torch.zeros(max_len, 1, c.d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term) # pe: [seq_lens * 1 * d_model] for each sample

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.head_size = head_size ###
        self.key = nn.Linear(c.d_model,head_size,bias=False)
        self.query = nn.Linear(c.d_model,head_size,bias=False)
        self.value = nn.Linear(c.d_model,head_size,bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(c.sequence_l, c.sequence_l)))


    def forward(self,x): # x:[batch, l_seq, d_model]
        k = self.key(x) # k:[batch, l_seq, head_size]
        q = self.query(x) # q:[batch, l_seq, head_size]
        v = self.value(x) # v:[batch, l_seq, head_size]
        qkt = q@k.transpose(2,1)/self.head_size**0.5 #[batch*l_seq*l_seq]
        qkt = qkt.masked_fill(self.tril == 0, float('-inf'))
        qkt = F.softmax(qkt, dim = -1)
        z = qkt@v # z:[batch * l_seq*l_seq]@[batch, l_seq, head_size] = [batch, l_seq, head_size]
        return z

class MultiHeadAttention(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.self_attention = nn.ModuleList([Head(head_size) for _ in range(c.number_head)])
        self.w0 = nn.Linear(head_size*c.number_head,c.d_model)

    def forward(self,x):
        head_outputs = [head(x) for head in self.self_attention]
        output = torch.cat(head_outputs, dim=-1) # [batch, l_seq, head_size*number_head]
        output = self.w0(output) # output:[batch, l_seq, d_model], so that it can be added with residual
        return output
    

# The Multi-Heads Self-Attention mechanism is followed by two fully connected layers of
# the Feed Forward block. The first (hidden) layer contains 4 times as many neurons as the input
# sequence with the ReLU activation function. The dimension of the second layer is
# equal to the dimension of the input sequence, and neurons do not use the activation function.
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff=nn.Sequential( nn.Linear(c.d_model,4*c.d_model),
                              nn.ReLU(),
                              nn.Linear(4*c.d_model,c.d_model))
    def forward(self,x):
        x = self.ff(x)
        return x
    

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = c.d_model // c.number_head

        self.self_attention = MultiHeadAttention(head_size)

        self.norm1  = nn.LayerNorm(c.d_model)

        self.ffn = FeedForward()

        self.norm2  = nn.LayerNorm(c.d_model)###

    def forward(self,x):
        x = x + self.self_attention(self.norm1(x))
        out = x + self.ffn(self.norm2(x))

        return out

# Model
class Model(nn.Module):
    def __init__(self,stoi):
        super().__init__()
        self.stoi = stoi
        self.tok_emb = nn.Embedding(len(stoi),c.d_model)
        self.pos_emb = PositionalEncoding()
        self.dropout1 = nn.Dropout(c.dropout)

        self.blocks = nn.Sequential(*[Block() for _ in range(c.num_layer)])
        self.norm_final = nn.LayerNorm(c.d_model)
        self.predict = nn.Linear(c.d_model,len(stoi))
        self.loss_compute = nn.CrossEntropyLoss()

    def forward(self, x, use='train',y = None ):
        emb_x = self.tok_emb(x)
        emb_x = self.pos_emb(emb_x) # x,y = emb = [batch size * sequence_l * d_model]
        emb_x = self.dropout1(emb_x)

        emb_x = self.blocks(emb_x)
        x = self.norm_final(emb_x)
        logit = self.predict(x) #[batch size * sequence_l * number_of_char]
        # y:[batch size * l_sequence * 1]

        if use == 'train':
            logit = logit.view(c.batch_size*c.sequence_l,len(self.stoi))
            y = y.view(c.batch_size*c.sequence_l)
            loss = self.loss_compute(logit,y)
        elif use == 'generate':
            loss = None

        return logit, loss # loss for training, logit for generate
    

    @torch.no_grad()
    def generate(self, output_length, seed_idx, criteria):
        out = seed_idx
        for _ in range(output_length):
            logit,_ = self(seed_idx, use = 'generate')
            prob = F.softmax(logit[-1,-1,:], dim = -1)
            
            if criteria == 'high_prob': 
                next_idx = prob.argmax()
                out  = torch.cat([out , next_idx.unsqueeze(0)], dim=-1)
            else: 
                # given a probability, get 1 sample
                next_idx = torch.multinomial(prob, 1)
                out  = torch.cat([out , next_idx], dim=-1)
            
            seed_idx = out[-c.sequence_l:]
        return out



def calculate_perplexity(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_words = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets, reduction='sum')

            total_loss += loss.item()
            total_words += targets.size(0)

    perplexity = 2 ** (total_loss / total_words)
    return perplexity

