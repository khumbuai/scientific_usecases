################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# CNN setup and data normalization
#
################

import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf


path_to_pb_file = "../data/flow_model.pb"

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def blockUNet(in_c, out_c, name, transposed=False, bn=True, relu=True, size=4, pad=1, dropout=0.):
    block = nn.Sequential()
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, kernel_size=size, stride=2, padding=pad, bias=True))
    else:
        block.add_module('%s_upsam' % name, nn.Upsample(scale_factor=2))
        block.add_module('%s_tconv' % name,
                         nn.Conv2d(in_c, out_c, kernel_size=(size - 1), stride=1, padding=pad, bias=True))
    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout > 0.:
        block.add_module('%s_dropout' % name, nn.Dropout2d(dropout, inplace=True))
    return block


# generator model
class TurbNetG(nn.Module):
    def __init__(self, channelExponent=6, dropout=0.):
        super(TurbNetG, self).__init__()
        channels = int(2 ** channelExponent + 0.5)

        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_conv', nn.Conv2d(3, channels, 4, 2, 1, bias=True))

        self.layer2 = blockUNet(channels, channels * 2, 'layer2', transposed=False, bn=True, relu=False,
                                dropout=dropout)
        self.layer2x = blockUNet(channels * 2, channels * 2, 'layer2x', transposed=False, bn=True, relu=False,
                                 dropout=dropout)
        self.layer3 = blockUNet(channels * 2, channels * 4, 'layer3', transposed=False, bn=True, relu=False,
                                dropout=dropout)
        self.layer4 = blockUNet(channels * 4, channels * 8, 'layer4', transposed=False, bn=True, relu=False,
                                dropout=dropout, size=2, pad=0)
        self.layer5 = blockUNet(channels * 8, channels * 8, 'layer5', transposed=False, bn=True, relu=False,
                                dropout=dropout, size=2, pad=0)
        self.layer6 = blockUNet(channels * 8, channels * 8, 'layer6', transposed=False, bn=False, relu=False,
                                dropout=dropout, size=2, pad=0)

        self.dlayer6 = blockUNet(channels * 8, channels * 8, 'dlayer6', transposed=True, bn=True, relu=True,
                                 dropout=dropout, size=2, pad=0)
        self.dlayer5 = blockUNet(channels * 16, channels * 8, 'dlayer5', transposed=True, bn=True, relu=True,
                                 dropout=dropout, size=2, pad=0)
        self.dlayer4 = blockUNet(channels * 16, channels * 4, 'dlayer4', transposed=True, bn=True, relu=True,
                                 dropout=dropout)
        self.dlayer3 = blockUNet(channels * 8, channels * 2, 'dlayer3', transposed=True, bn=True, relu=True,
                                 dropout=dropout)
        self.dlayer2x = blockUNet(channels * 4, channels * 2, 'dlayer2x', transposed=True, bn=True, relu=True,
                                  dropout=dropout)
        self.dlayer2 = blockUNet(channels * 4, channels, 'dlayer2', transposed=True, bn=True, relu=True,
                                 dropout=dropout)

        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_relu', nn.ReLU(inplace=True))
        self.dlayer1.add_module('dlayer1_tconv', nn.ConvTranspose2d(channels * 2, 3, 4, 2, 1, bias=True))

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out2x = self.layer2x(out2)
        out3 = self.layer3(out2x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        dout6 = self.dlayer6(out6)
        dout6_out5 = torch.cat([dout6, out5], 1)
        dout5 = self.dlayer5(dout6_out5)
        dout5_out4 = torch.cat([dout5, out4], 1)
        dout4 = self.dlayer4(dout5_out4)
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout3 = self.dlayer3(dout4_out3)
        dout3_out2x = torch.cat([dout3, out2x], 1)
        dout2x = self.dlayer2x(dout3_out2x)
        dout2x_out2 = torch.cat([dout2x, out2], 1)
        dout2 = self.dlayer2(dout2x_out2)
        dout2_out1 = torch.cat([dout2, out1], 1)
        dout1 = self.dlayer1(dout2_out1)
        return dout1


# discriminator (only for adversarial training, currently unused)
class TurbNetD(nn.Module):
    def __init__(self, in_channels1, in_channels2, ch=64):
        super(TurbNetD, self).__init__()

        self.c0 = nn.Conv2d(in_channels1 + in_channels2, ch, 4, stride=2, padding=2)
        self.c1 = nn.Conv2d(ch, ch * 2, 4, stride=2, padding=2)
        self.c2 = nn.Conv2d(ch * 2, ch * 4, 4, stride=2, padding=2)
        self.c3 = nn.Conv2d(ch * 4, ch * 8, 4, stride=2, padding=2)
        self.c4 = nn.Conv2d(ch * 8, 1, 4, stride=2, padding=2)

        self.bnc1 = nn.BatchNorm2d(ch * 2)
        self.bnc2 = nn.BatchNorm2d(ch * 4)
        self.bnc3 = nn.BatchNorm2d(ch * 8)

    def forward(self, x1, x2):
        h = self.c0(torch.cat((x1, x2), 1))
        h = self.bnc1(self.c1(F.leaky_relu(h, negative_slope=0.2)))
        h = self.bnc2(self.c2(F.leaky_relu(h, negative_slope=0.2)))
        h = self.bnc3(self.c3(F.leaky_relu(h, negative_slope=0.2)))
        h = self.c4(F.leaky_relu(h, negative_slope=0.2))
        h = F.sigmoid(h)
        return h


# https://www.tensorflow.org/guide/migrate
# https://www.tensorflow.org/api_docs/python/tf/graph_util/import_graph_def?version=stable
def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    print("-" * 50)
    print("Frozen model layers: ")
    layers = [op.name for op in import_graph.get_operations()]
    count = 0
    if print_graph == True:
        for layer in layers:
            print(layer)
            count+=1
    print('number of layers', len(layers))

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))


def build_graph():
    # Load frozen graph using TensorFlow 1.x functions
    with tf.io.gfile.GFile(path_to_pb_file , "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(f.read())

    # Wrap frozen graph to ConcreteFunctions
    frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                    inputs=["model_inputs:0"],
                                    outputs=["model_outputs:0"],
                                    print_graph=False)

    return frozen_func
