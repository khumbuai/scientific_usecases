import unittest
import numpy as np
import torch
import tensorflow as tf
from DfpNet import TurbNetG, build_graph

path2weights = "../data/model_mata10_exp70.dms"
device = 'cpu'

class Example(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.num_samples = 10
        expo = 7  # parameter necessary for determining the number of channels after the first convolution layer
        print("load models")
        cls.netG = TurbNetG(channelExponent=expo).eval()
        cls.netG.load_state_dict(torch.load(path2weights, map_location=device))
        cls.model = build_graph()

    def test_outputs(self):
        print(f"check if outputs match for {self.num_samples} samples.")
        samples = np.random.random((self.num_samples,1,3,128,128)).astype('float32')
        max_diff = 0
        for x in samples:
            y_tf = self.model(model_inputs=tf.constant(x))[0].numpy()
            with torch.no_grad():
                y_torch = self.netG(torch.from_numpy(x)).numpy()
            delta = np.abs(y_tf-y_torch).max()
            if max_diff < delta:
                max_diff = delta
            self.assertTrue(delta<1e-6)
        print("max distance between predictions is",delta)



if __name__ == '__main__':
    unittest.main()
