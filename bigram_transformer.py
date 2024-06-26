import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    # change the mode of the model to be evaluation state.
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)

            # this is will evaluate the model again at random samples data.
            # will evaluate the current weight of the params, then use it to predict on the training data.
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        # the we average over 200 data/ samples data, and not every one of samples data.
        out[split] = losses.mean()

    # change back to training mode.
    model.train()
    return out

class Head(nn.Module):
    "one head self attention!"
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # there is no tril in the paramater of the module, so in the torch convention, we call it buffer and not paramater
        #  and we have to assign it to the module.
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B, T, C = x.shape

        # do the self attention mechanism
        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)

        # calculating the wei (attention scores/ affinities)
        wei = (q @ k.transpose(-2,-1)) * (C **-0.5) # (B,T,T) ==> remember to do the normalization

        # masking the upper triangular part of the matrix
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # this will do softmax along with the last dimension, or along the row dim
        # sum to one one the row.
        wei = F.softmax(wei, dim=-1) # (B,T,T)

        # now we calculate the value and aggregate it with the attention scores!
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B,T,T)@(B,T,C) ==> (B,T,C)
        return out
    
class MultiHeadAttention(nn.Module):
    "multi-heads of self attention running in parallel!"""
    def __init__(self, num_heads, head_size):
        super().__init__()  
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        # x is (B,T,C)
        # this is to say that the output of the multi-head attention is then concatenated along the channel dim!
        return torch.cat([h(x) for h in self.heads], dim=-1)

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # layer to embed token
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # layer to add positional embeddings
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # adding self attention head ==> just for now, we keep the head size to be the same as the embedding size (C)/ size 32
        # self.sa_head = Head(n_embd)

        # Now we try using multi-head attention
        # the idea behind multi-head attention is to have multiple heads of self attention running in parallel
        # and the idea is instead of having one attention that have large heads (32 head-size / C), 
        # we can have multiple heads of smaller size (8 head-size) and then concatenate the output of each head
        # so in the end, we will have the same output size as the single head attention.
        self.sa_heads = MultiHeadAttention(num_heads=4, head_size=n_embd // 4) # 4 heads of size 8-dimensional attention. 

        # adding language model head, to predict the next token (logits)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # positional embeddings
        pos_embd = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,C)
        # idx and targets are both (B,T) tensor of integers
        tok_embd = self.token_embedding_table(idx) # ==> (B,T,C)
        x = tok_embd + pos_embd # ==> (T,C) + (B,T,C) ==> (B,T,C)
        # apply the self attention head
        x = self.sa_heads(x)

        # this will do tok_embd@lm_head
        # Note: assuming n_embd = C, will give us (B,T,vocab_size)
        logits = self.lm_head(x) # (B,T,C)@(n_embd,vocab_size) => (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop the idx to the latest block size token
            idx_crop = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_crop)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


# RUN HERE

model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))