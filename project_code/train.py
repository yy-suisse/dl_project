import torch
import util
import custum_dataset
from torch.utils.data import DataLoader
from model import Model


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path = "data/data_hm.txt"
split = 0.8
data_train,stoi,itos = util.load_data_with_split(path,split,type='train')

c = util.config()
dataset = custum_dataset.CustomDataset(c.sequence_l,device,stoi,itos,data_train,repeat = False)
data_loader = DataLoader(dataset, c.batch_size, shuffle=True)

model = Model(stoi=dataset.stoi)
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=c.learning_rate)
model.train(True)

for epoch in range(10):
    inputs, targets = next(iter(data_loader))
    logit,loss = m(inputs,'train', targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss)
    print(torch.exp(loss))

model.eval()
# seed = "His cat jumped onto the table, "
seed = "Y-Z1234567890123456"
empty = "z"*c.sequence_l
seed_idx = util.encode(seed,stoi)
if len(seed)<c.sequence_l: 
    input_idx = util.encode(empty,stoi)
    input_idx[-len(seed):] = seed_idx
else:
    input_idx = seed_idx[:util.sequence_l]

input_idx = torch.tensor(input_idx,dtype=torch.long).to(device)
out = m.generate(16,input_idx,'high_prob')
print(util.decode(out,itos))
