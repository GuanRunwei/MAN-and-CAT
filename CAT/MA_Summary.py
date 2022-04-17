import torchsummary
import torch
from nets.attention import ma_block


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    ma_net = ma_block(channels=128, input_shape=[20, 20]).to(device)
    print(torchsummary.summary(ma_net, input_size=(128, 20, 20)))