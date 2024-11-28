from models.layers import *
from .net_util import shallow_cell
from .net_util import EN_CNN_Block
from .net_util import EN_SNN_TemATT_Block
from .net_util import EN_SNN_SpaATT_Block
from .net_util import SNNTem_Module
from .net_util import CNNSpatialAttention
from .net_util import DE_Block
from .net_util import TSST
from .net_util import TSST_EN_SNN_Block


class FusionTemATT(nn.Module):
    def __init__(self, imgchannel, eventchannel, outchannel):
        super(FusionTemATT, self).__init__()
        self.pool = APLayer(2)

        snn_layers = [8, 16, 32, 64]

        # SNN Encoder
        self.head_tsst = SNNTem_Module()
        self.snn_head  = SNNConvLayer(eventchannel, snn_layers[0], 3, 1, 1)
        self.snn_down1 = EN_SNN_TemATT_Block(snn_layers[0], snn_layers[1])
        self.snn_down2 = EN_SNN_TemATT_Block(snn_layers[1], snn_layers[2])
        self.snn_down3 = EN_SNN_TemATT_Block(snn_layers[2], snn_layers[3])

        self.headnormal = torch.nn.InstanceNorm2d(eventchannel)
        self.normal1 = torch.nn.InstanceNorm2d(snn_layers[0])
        self.normal2 = torch.nn.InstanceNorm2d(snn_layers[1])
        self.normal3 = torch.nn.InstanceNorm2d(snn_layers[2])
        self.normal4 = torch.nn.InstanceNorm2d(snn_layers[3])

        # CNN Encoder
        self.cnn_head = shallow_cell(imgchannel + eventchannel, 32)
        self.cnn_down1 = EN_CNN_Block(32, 64)
        self.cnn_down2 = EN_CNN_Block(64, 128)
        self.cnn_down3 = EN_CNN_Block(128, 256)

        self.conv1 = shallow_cell(32 + snn_layers[0], 32)
        self.conv2 = shallow_cell(64 + snn_layers[1], 64)
        self.conv3 = shallow_cell(128 + snn_layers[2], 128)
        self.conv4 = shallow_cell(256 + snn_layers[3], 256)

        self.up1 = DE_Block(256, 128)
        self.up2 = DE_Block(128, 64)
        self.up3 = DE_Block(64, 32)
        self.out = nn.Conv2d(32, outchannel, 3, 1, 1)

    def forward(self, spike, img):
        attspike, _ = self.head_tsst(spike)
        fusion_img = self.fusion_block(attspike, img, self.headnormal)

        s1 = self.snn_head(spike)  # 3 10 4 512 512
        s2 = self.snn_down1(s1)  # 3 10 4 256 256
        s3 = self.snn_down2(s2)  # 3 10 4 128 128
        s4 = self.snn_down3(s3)  # 3 10 4 64 64

        m1 = self.cnn_head(fusion_img)  # 3 32 512 512
        m1 = self.conv1(self.fusion_block(s1, m1, self.normal1))
        m2 = self.cnn_down1(m1)
        m2 = self.conv2(self.fusion_block(s2, m2, self.normal2))
        m3 = self.cnn_down2(m2)
        m3 = self.conv3(self.fusion_block(s3, m3, self.normal3))
        m4 = self.cnn_down3(m3)
        m4 = self.conv4(self.fusion_block(s4, m4, self.normal4))

        x = self.up1(m4, m3)
        x = self.up2(x, m2)
        x = self.up3(x, m1)
        x = self.out(x)

        outlayer = x + img

        return outlayer

    def fusion_block(self, spikefeature, imgfeature, normal):
        '''
        Args:
            spikefeature: spike feature map [N, t, C, x, y], N is batch_size, C is channel number, t is time stamp.
            imgfeature: image feature map [N, C, x, y]

        Returns:
            fusionfeature: image fusion feature map [N, C', x, y]
        '''
        tensor_spikefeature = normal(spikefeature.sum(dim=1))
        img_fusion_feature = torch.cat((tensor_spikefeature, imgfeature), dim=1)

        return img_fusion_feature


class FusionSpaATT(nn.Module):
    def __init__(self, imgchannel, eventchannel, outchannel):
        super(FusionSpaATT, self).__init__()
        self.pool = APLayer(2)

        snn_layers = [8, 16, 32, 64]

        # SNN Encoder
        self.head_tsst = CNNSpatialAttention(eventchannel)
        self.snn_head  = SNNConvLayer(eventchannel, snn_layers[0], 3, 1, 1)
        self.snn_down1 = EN_SNN_SpaATT_Block(snn_layers[0], snn_layers[1])
        self.snn_down2 = EN_SNN_SpaATT_Block(snn_layers[1], snn_layers[2])
        self.snn_down3 = EN_SNN_SpaATT_Block(snn_layers[2], snn_layers[3])

        self.headnormal = torch.nn.InstanceNorm2d(eventchannel)
        self.normal1 = torch.nn.InstanceNorm2d(snn_layers[0])
        self.normal2 = torch.nn.InstanceNorm2d(snn_layers[1])
        self.normal3 = torch.nn.InstanceNorm2d(snn_layers[2])
        self.normal4 = torch.nn.InstanceNorm2d(snn_layers[3])

        # CNN Encoder
        self.cnn_head = shallow_cell(imgchannel + eventchannel, 32)
        self.cnn_down1 = EN_CNN_Block(32, 64)
        self.cnn_down2 = EN_CNN_Block(64, 128)
        self.cnn_down3 = EN_CNN_Block(128, 256)

        self.conv1 = shallow_cell(32 + snn_layers[0], 32)
        self.conv2 = shallow_cell(64 + snn_layers[1], 64)
        self.conv3 = shallow_cell(128 + snn_layers[2], 128)
        self.conv4 = shallow_cell(256 + snn_layers[3], 256)

        self.up1 = DE_Block(256, 128)
        self.up2 = DE_Block(128, 64)
        self.up3 = DE_Block(64, 32)
        self.out = nn.Conv2d(32, outchannel, 3, 1, 1)

    def forward(self, spike, img):
        attspike   = self.head_tsst(spike.sum(dim=1))
        fusion_img = self.fusion_block(attspike, img, self.headnormal)

        s1 = self.snn_head(spike)  # 3 10 4 512 512
        maps2, s2 = self.snn_down1(s1)  # 3 10 4 256 256
        maps3, s3 = self.snn_down2(s2)  # 3 10 4 128 128
        maps4, s4 = self.snn_down3(s3)  # 3 10 4 64 64

        m1 = self.cnn_head(fusion_img)  # 3 32 512 512
        m1 = self.conv1(self.fusion_block(s1.sum(dim=1), m1, self.normal1))
        m2 = self.cnn_down1(m1)
        m2 = self.conv2(self.fusion_block(maps2, m2, self.normal2))
        m3 = self.cnn_down2(m2)
        m3 = self.conv3(self.fusion_block(maps3, m3, self.normal3))
        m4 = self.cnn_down3(m3)
        m4 = self.conv4(self.fusion_block(maps4, m4, self.normal4))

        x = self.up1(m4, m3)
        x = self.up2(x, m2)
        x = self.up3(x, m1)
        x = self.out(x)

        outlayer = x + img

        return outlayer

    def fusion_block(self, spikefeature, imgfeature, normal):
        '''
        Args:
            spikefeature: spike feature map [N, t, C, x, y], N is batch_size, C is channel number, t is time stamp.
            imgfeature: image feature map [N, C, x, y]

        Returns:
            fusionfeature: image fusion feature map [N, C', x, y]
        '''
        tensor_spikefeature = normal(spikefeature)
        img_fusion_feature = torch.cat((tensor_spikefeature, imgfeature), dim=1)

        return img_fusion_feature


class Fusion_MOSNN(nn.Module):
    def __init__(self, imgchannel, eventchannel, outchannel):
        super(Fusion_MOSNN, self).__init__()
        self.pool = APLayer(2)

        snn_layers = [8, 16, 32, 64]

        # SNN Encoder
        self.head_tsst = TSST(d_model=eventchannel, d_k=eventchannel, d_v=eventchannel, scale=32)
        self.snn_head = SNNConvLayer(eventchannel, snn_layers[0], 3, 1, 1)
        self.snn_down1 = TSST_EN_SNN_Block(snn_layers[0], snn_layers[1], scale=16)
        self.snn_down2 = TSST_EN_SNN_Block(snn_layers[1], snn_layers[2], scale=8)
        self.snn_down3 = TSST_EN_SNN_Block(snn_layers[2], snn_layers[3], scale=4)

        self.headnormal = torch.nn.InstanceNorm2d(eventchannel)
        self.normal1 = torch.nn.InstanceNorm2d(snn_layers[0])
        self.normal2 = torch.nn.InstanceNorm2d(snn_layers[1])
        self.normal3 = torch.nn.InstanceNorm2d(snn_layers[2])
        self.normal4 = torch.nn.InstanceNorm2d(snn_layers[3])

        # CNN Encoder
        self.cnn_head = shallow_cell(imgchannel + eventchannel, 32)
        self.cnn_down1 = EN_CNN_Block(32, 64)
        self.cnn_down2 = EN_CNN_Block(64, 128)
        self.cnn_down3 = EN_CNN_Block(128, 256)

        self.conv1 = shallow_cell(32 + snn_layers[0], 32)
        self.conv2 = shallow_cell(64 + snn_layers[1], 64)
        self.conv3 = shallow_cell(128 + snn_layers[2], 128)
        self.conv4 = shallow_cell(256 + snn_layers[3], 256)

        self.up1 = DE_Block(256, 128)
        self.up2 = DE_Block(128, 64)
        self.up3 = DE_Block(64, 32)
        self.out = nn.Conv2d(32, outchannel, 3, 1, 1)

    def forward(self, spike, img):
        self.head_tsst(spike)
        attspike =self.head_tsst.fusionattention(spike)
        fusion_img = self.fusion_block(attspike, img, self.headnormal)

        s1 = self.snn_head(spike)       #3 10 4 512 512
        s2, atts2 = self.snn_down1(s1)  #3 10 4 256 256
        s3, atts3 = self.snn_down2(s2)  #3 10 4 128 128
        s4, atts4 = self.snn_down3(s3)  #3 10 4 64 64

        m1 = self.cnn_head(fusion_img)         #3 32 512 512
        m1 = self.conv1(self.fusion_block(s1, m1, self.normal1))
        m2 = self.cnn_down1(m1)
        m2 = self.conv2(self.fusion_block(atts2, m2, self.normal2))
        m3 = self.cnn_down2(m2)
        m3 = self.conv3(self.fusion_block(atts3, m3, self.normal3))
        m4 = self.cnn_down3(m3)
        m4 = self.conv4(self.fusion_block(atts4, m4, self.normal4))

        x = self.up1(m4, m3)
        x = self.up2(x, m2)
        x = self.up3(x, m1)
        x = self.out(x)

        outlayer = x + img

        return outlayer

    def fusion_block(self, spikefeature, imgfeature, normal):
        '''
        Args:
            spikefeature: spike feature map [N, t, C, x, y], N is batch_size, C is channel number, t is time stamp.
            imgfeature: image feature map [N, C, x, y]

        Returns:
            fusionfeature: image fusion feature map [N, C', x, y]
        '''
        tensor_spikefeature = normal(spikefeature.sum(dim=1))
        img_fusion_feature = torch.cat((tensor_spikefeature, imgfeature), dim=1)

        return img_fusion_feature


if __name__ == '__main__':
    model = Fusion_MOSNN()
