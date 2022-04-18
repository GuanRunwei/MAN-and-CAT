#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
import torch
from torchsummary import summary

from nets.yolo import YoloBody

if __name__ == "__main__":
    # 需要使用device来指定网络在GPU还是CPU运行
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    m = YoloBody([[3, 4, 5], [1, 2, 3]], 80, phi=4, input_shape=[512, 512]).to(device)
    summary(m, input_size=(3, 512, 512))