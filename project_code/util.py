import os

def load_data_with_split(path,split,type='train'):
    """load data, split them into train and test, and compute lookup disctionary"""
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    if type == 'train':
        data = text[0:int(split * len(text))]
    elif type == 'test':
        data = text[int(split * len(text)):]
    else:
        assert("false type")

    chars = sorted(list(set(text)))
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(stoi) }

    return data,stoi,itos

def encode(s,stoi):
    """take a string, output a list of integers"""
    return [stoi[c] for c in s]

def decode(l,itos):
    """take a list of integers, output a string"""
    l = l.tolist()
    return ''.join([itos[i] for i in l])

def save_generated_text(file_name,out):
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # Join the directory and file name
    file_path = os.path.join(output_dir, file_name)
    with open(file_path, 'w') as file:
    # Write the string to the file
        file.write(out)

class config:
    sequence_l = 128
    batch_size = 128
    d_model = 768 # d_model， embedding dim
    num_layer = 12 # number of blocks stacked
    number_head = 8  # multihead attention
    d_ff = 2048 # feedforward dimension
    dropout = 0.2

    learning_rate = 5e-4
    
    gamma = 0.9
    split = 0.5

