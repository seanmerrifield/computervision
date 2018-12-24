import unittest
from image_caption.model import EncoderCNN, DecoderRNN
import cv2
import torch
import numpy as np

class TestRCNN(unittest.TestCase):
    embed_size = 256

    decoder = DecoderRNN(embed_size=256,
                         hidden_size=256,
                         vocab_size=100,
                         num_layers=2)


    def test_init(self):

        self.assertEquals(type(self.decoder), DecoderRNN)

    def test_forward(self):

        features = torch.ones(256, dtype=torch.float)
        features = features.unsqueeze(0)


        captions = torch.tensor([1, 2, 3, 4, 5, 6, 7 ,8 ])
        captions = captions.unsqueeze(0)

        output = self.decoder(features, captions)
        self.assertEqual(type(output), torch.Tensor, "Decoder output needs to be a PyTorch Tensor.")

    def test_sample(self):
        features = torch.rand(256, dtype=torch.float)
        features = features.unsqueeze(0)
        features = features.unsqueeze(1)

        self.decoder.sample(features, max_len=20)
