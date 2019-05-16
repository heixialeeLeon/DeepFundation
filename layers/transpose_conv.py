import torch
import torch.nn as nn
from torch.autograd import Variable

input = Variable(torch.randn(1,1,3,3))
conv_op = nn.Conv2d(1,1,kernel_size=3,stride=1,padding=0)
conv_output = conv_op(input)

"""
conv:
outoput = ((input-k+2p)/s) +1 
"""
print("conv output shape: {}".format(conv_output.shape))

"""
transpose conv:
output = (input-1)*s +k -2p
"""
trans_conv_op = nn.ConvTranspose2d(1,1,kernel_size=3,stride=1,padding=1)
trans_conv_output = trans_conv_op(input)
print("trans conv output shape, padding=1: {}".format(trans_conv_output.shape))

trans_conv_op = nn.ConvTranspose2d(1,1,kernel_size=3,stride=1,padding=0)
trans_conv_output = trans_conv_op(input)
print("trans conv output shape, padding=0: {}".format(trans_conv_output.shape))

trans_conv_op = nn.ConvTranspose2d(1,1,kernel_size=3,stride=2,padding=1)
trans_conv_output = trans_conv_op(input)
print("trans conv output shape: {}".format(trans_conv_output.shape))