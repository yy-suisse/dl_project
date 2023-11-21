from torch.utils.data import Dataset, DataLoader
import torch


class CustomDataset(Dataset):
    def __init__(self,sequence_l,device,stoi,itos,data,repeat = False):

        self.data = data # data
        self.sequence_l = sequence_l
        self.device = device
        self.stoi = stoi # lookup table
        self.itos = itos # lookup table
        self.repeat = repeat # test generate repeat boolean

    def __len__(self):
        return len(self.data) - self.sequence_l

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        if self.repeat == False:
            chunk = self.data[idx:idx + self.sequence_l + 1]
            # encode every character to an integer
            idx_chunk = [self.stoi[c] for c in chunk]
            x = torch.tensor(idx_chunk[:-1], dtype=torch.long)
            # return the chunk and the shifted version as tensors
            y = torch.tensor(idx_chunk[1:], dtype=torch.long)
            x,y = x.to(self.device),y.to(self.device)
        else: 
            chunk = self.data[idx:idx + self.sequence_l]
            # encode every character to an integer
            idx_chunk = [self.stoi[c] for c in chunk]
            x = torch.tensor(idx_chunk, dtype=torch.long)
            # return the chunk and the shifted version as tensors
            y = torch.tensor(idx_chunk, dtype=torch.long)
            x,y = x.to(self.device),y.to(self.device)

        return x,y

