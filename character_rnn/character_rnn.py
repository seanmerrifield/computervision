import numpy as np
import torch
import helper
from model import CharRNN

## Load in Data

# open text file and read in data as `text`
with open('data/anna.txt', 'r') as f:
    text = f.read()

chars = tuple(set(text))
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}
encoded = np.array([char2int[ch] for ch in text])

# define and print the net
net = CharRNN(chars, n_hidden=512, n_layers=2)
print(net)


n_seqs, n_steps = 128, 100

# you may change cuda to True if you plan on using a GPU!
# also, if you do, please INCREASE the epochs to 25
helper.train(net, encoded, epochs=1, n_seqs=n_seqs, n_steps=n_steps, lr=0.001, cuda=False, print_every=10)

print(helper.sample(net, 2000, prime='Anna', top_k=5, cuda=False))


# change the name, for saving multiple files
model_name = 'rnn_1_epoch.net'

checkpoint = {'n_hidden': net.n_hidden,
              'n_layers': net.n_layers,
              'state_dict': net.state_dict(),
              'tokens': net.chars}

#Save trained model
with open(model_name, 'wb') as f:
    torch.save(checkpoint, f)