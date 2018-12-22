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

        features = torch.from_numpy(np.ones(256))
        features = features.unsqueeze(0)


        caption = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        captions = torch.from_numpy(caption)
        captions = captions.unsqueeze(0)

        self.decoder(features, captions)

