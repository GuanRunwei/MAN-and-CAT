import torchsummary
import torch
# from net_publish.MixAttention import MixAttentionBlock
from net_publish.MixAttentionTinyBlock import MixAttentionBlock


device = "cuda:0" if torch.cuda.is_available() else "cpu"


if __name__ == '__main__':
    ma_block = MixAttentionBlock(in_channels=64, input_shape=[56, 56]).to(device)
    print(torchsummary.summary(ma_block, input_size=(64, 56, 56)))
