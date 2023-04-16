
from nets.CAL_Net import CAL_Net
from utils.utils import net_flops
# from transunet.model import TransUNet
if __name__ == "__main__":
    input_shape     = [256, 256]
    num_classes     = 2
    backbone        = 'yolox'
    # model =TransUNet(256)
    model = CAL_Net([input_shape[0], input_shape[1], 3], num_classes, backbone)
    #--------------------------------------------#
    #   查看网络结构网络结构
    #--------------------------------------------#
    model.summary()
    #--------------------------------------------#
    #   计算网络的FLOPS
    #--------------------------------------------#
    net_flops(model, table=False)
    #--------------------------------------------#
    #   获得网络每个层的名称与序号
    #--------------------------------------------#

    # for i,layer in enumerate(model.layers):
    #     print(i,layer.name)
