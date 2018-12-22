import unittest
from image_caption.model import EncoderCNN, DecoderRNN
import cv2
import torch
import numpy as np

class TestRCNN(unittest.TestCase):
    embed_size = 256

    encoder = EncoderCNN(embed_size=256)

    decoder = DecoderRNN(embed_size=256,
                         hidden_size=256,
                         vocab_size=100,
                         num_layers=2)


    def test_init(self):

        self.assertEquals(type(self.decoder), DecoderRNN)

    def test_forward(self):

        img = cv2.imread('image.png')

        features = torch.from_numpy(img)

        caption = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        captions = torch.from_numpy(caption)


        self.decoder(features, captions)

