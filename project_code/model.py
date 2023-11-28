import torch
from torch import nn
from torch.nn import functional as F
from util import config
import math

c = config()



class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.head_size = head_size ###
        self.key = nn.Linear(c.d_model,head_size,bias=False)
        self.query = nn.Linear(c.d_model,head_size,bias=False)
        self.value = nn.Linear(c.d_model,head_size,bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(c.sequence_l, c.sequence_l))) #tril : lower trangular = 1 --> causal


    def forward(self,x): 
        B,L,D = x.shape # x:[batch, l_seq, d_model]
        k = self.key(x) # k:[batch, l_seq, head_size]
        q = self.query(x) # q:[batch, l_seq, head_size]
        v = self.value(x) # v:[batch, l_seq, head_size]

        qkt = q@k.transpose(2,1)/(self.head_size**0.5) #[batch*l_seq*l_seq]

        qkt = qkt.masked_fill(self.tril[:L,:L] == 0, float('-inf'))
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
        self.ff=nn.Sequential(nn.Linear(c.d_model,4*c.d_model),
                              nn.ReLU(),
                              nn.Linear(4*c.d_model,c.d_model),
                              nn.Dropout(c.dropout))
    def forward(self,x):
        x = self.ff(x)
        return x
    

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = c.d_model // c.number_head

        self.self_attention = MultiHeadAttention(head_size)

        self.norm1  = nn.LayerNorm(c.d_model) # normalize for every sample

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

        self.apply(self._init_weights)
        self.stoi = stoi
        self.tok_emb = nn.Embedding(len(stoi),c.d_model)
        self.position_embedding_table = nn.Embedding(c.sequence_l,c.d_model)
        self.dropout1 = nn.Dropout(c.dropout)

        self.blocks = nn.Sequential(*[Block() for _ in range(c.num_layer)])
        self.norm_final = nn.LayerNorm(c.d_model)
        self.predict = nn.Linear(c.d_model,len(stoi))
        self.loss_compute = nn.CrossEntropyLoss()

    def forward(self, x, use='train',y = None ):
        _,L = x.shape
        emb_x = self.tok_emb(x)
        device = emb_x.device
        emb_pos = self.position_embedding_table(torch.arange(L,device = device))
        emb_x = emb_pos + emb_x
        
        emb_x = self.dropout1(emb_x)

        emb_x = self.blocks(emb_x)
        x = self.norm_final(emb_x)
        logit = self.predict(x) #[batch size * sequence_l * number_of_char]
        # y:[batch size * l_sequence * 1]

        if use == 'train':
            logit = logit.view(c.batch_size*c.sequence_l,len(self.stoi))
            y = y.view(c.batch_size*c.sequence_l)
            loss = self.loss_compute(logit,y)
            perplexity = torch.exp(loss)
        elif use == 'generate':
            loss = None
            perplexity = None

        return logit, loss, perplexity # loss for training, logit for generate
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    

    @torch.no_grad()
    def generate(self, output_length, seed_idx, criteria):
        out = seed_idx

        for i in range(output_length):
            print(i)
            logit,_,_ = self(seed_idx, use = 'generate')  #[batch size * sequence_l * number_of_char]
            prob = F.softmax(logit[:,-1,:], dim = -1) #[batch size * number_of_char]
            
            if criteria == 'high_prob': 
                _, next_idx = prob.topk(1)
                out  = torch.cat([out , next_idx], dim=-1)
            else: 
                # given a probability, get 1 sample
                next_idx = torch.multinomial(prob, 1)
                out  = torch.cat([out , next_idx], dim=-1)
            
            seed_idx = out[:,-c.sequence_l:]
        return out.squeeze(0)
    
   



