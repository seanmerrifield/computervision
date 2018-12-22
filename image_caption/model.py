import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.n_layers = num_layers

        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers)

        # embedding layer that turns words into a vector of a specified size
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)

        # Add dropout layer with dropout probabililty of 0.5
        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(hidden_size, vocab_size)

        # Initialize hidden state
        self.hidden = self.init_hidden()

    def forward(self, features, captions):
        # Convert captions to embedded word vector
        embeds = self.embed(captions)

        # Removes last caption from list
        # This is needed to maintain the same output size as the input
        n_captions = list(embeds.size())[1]
        embeds = embeds.narrow(1, 0, n_captions - 1)

        #Concat image feature with captions
        x = torch.cat((features.unsqueeze(1), embeds), 1)

        x, self.hidden = self.lstm(x)

        x = self.dropout(x)

        # Stack up LSTM outputs using view
        #         x = x.view(x.size()[0]*x.size()[1], self.hidden_size)

        x = self.fc(x)

        # get the scores for the most likely tag for a word
        tag_scores = F.log_softmax(x, dim=1)

        return tag_scores

    def init_hidden(self):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x n_seqs x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        return (torch.zeros(self.n_layers, self.embed_size, self.hidden_size),
                torch.zeros(self.n_layers, self.embed_size, self.hidden_size))

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass