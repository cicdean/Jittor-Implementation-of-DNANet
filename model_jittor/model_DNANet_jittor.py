import jittor as jt
from jittor import nn


class ChannelAttention_jittor(nn.Module):
    def __init__(self, channels):
        super(ChannelAttention_jittor, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // 16, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(channels // 16, channels, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def execute(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention_jittor(nn.Module):
    def __init__(self):
        super(SpatialAttention_jittor, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, stride=1,bias=False)
        self.sigmoid = nn.Sigmoid()

    def execute(self, x):
        avg_out = jt.mean(x, dim=1, keepdims=True)
        max_out = jt.max(x, dim=1, keepdims=True)
        out = jt.contrib.concat((avg_out, max_out), dim=1)
        out = self.conv1(out)
        return self.sigmoid(out)


class Res_CBAM_block_jittor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Res_CBAM_block_jittor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = None

        self.ca = ChannelAttention_jittor(out_channels)
        self.sa = SpatialAttention_jittor()

    def execute(self, x):
        res = x
        if self.shortcut is not None:
            res = self.shortcut(res)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        ca = self.ca(x)
        x = ca * x
        x = self.sa(x) * x
        return self.relu(x + res)


class DNANet_jittor(nn.Module):
    def __init__(self, classes, in_channels, block, num_blocks, nb_filter, deep_supervision=False):
        super(DNANet_jittor, self).__init__()
        self.relu = nn.ReLU(inplace=True)   #useless
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.up = nn.UpsamplingBilinear2d(scale_factor=(2, 2))
        #self.down = nn.UpsamplingBilinear2d(scale_factor=(0.5, 0.5))
        #self.up_4 = nn.UpsamplingBilinear2d(scale_factor=(4, 4))
        #self.up_8 = nn.UpsamplingBilinear2d(scale_factor=(8, 8))
        #self.up_16 = nn.UpsamplingBilinear2d(scale_factor=(16, 16))

        self.conv0_0 = self.make_layer(block, in_channels, nb_filter[0], 1)
        self.conv1_0 = self.make_layer(block, nb_filter[0], nb_filter[1], num_blocks[0])
        self.conv2_0 = self.make_layer(block, nb_filter[1], nb_filter[2], num_blocks[1])
        self.conv3_0 = self.make_layer(block, nb_filter[2], nb_filter[3], num_blocks[2])
        self.conv4_0 = self.make_layer(block, nb_filter[3], nb_filter[4], num_blocks[3])

        self.conv0_1 = self.make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0], 1)
        self.conv1_1 = self.make_layer(block, nb_filter[0] + nb_filter[2] + nb_filter[1], nb_filter[1], num_blocks[0])
        self.conv2_1 = self.make_layer(block, nb_filter[1] + nb_filter[3] + nb_filter[2], nb_filter[2], num_blocks[1])
        self.conv3_1 = self.make_layer(block, nb_filter[2] + nb_filter[4] + nb_filter[3], nb_filter[3], num_blocks[2])

        self.conv0_2 = self.make_layer(block, nb_filter[0] * 2 + nb_filter[1], nb_filter[0], 1)
        self.conv1_2 = self.make_layer(block, nb_filter[0] + nb_filter[2] + nb_filter[1] * 2, nb_filter[1],
                                       num_blocks[0])
        self.conv2_2 = self.make_layer(block, nb_filter[1] + nb_filter[3] + nb_filter[2] * 2, nb_filter[2],
                                       num_blocks[1])

        self.conv0_3 = self.make_layer(block, nb_filter[0] * 3 + nb_filter[1], nb_filter[0], 1)
        self.conv1_3 = self.make_layer(block, nb_filter[0] + nb_filter[2] + nb_filter[1] * 3, nb_filter[1],
                                       num_blocks[0])

        self.conv0_4 = self.make_layer(block, nb_filter[0] * 4 + nb_filter[1], nb_filter[0], 1)

        self.conv0_4_final = self.make_layer(block, nb_filter[0] * 5, nb_filter[0], 1)
        self.conv0_4_1x1 = nn.Conv2d(nb_filter[4], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_3_1x1 = nn.Conv2d(nb_filter[3], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_2_1x1 = nn.Conv2d(nb_filter[2], nb_filter[0], kernel_size=1, stride=1)
        self.conv0_1_1x1 = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1, stride=1)

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], classes, kernel_size=1, stride=1)
            self.final2 = nn.Conv2d(nb_filter[0], classes, kernel_size=1, stride=1)
            self.final3 = nn.Conv2d(nb_filter[0], classes, kernel_size=1, stride=1)
            self.final4 = nn.Conv2d(nb_filter[0], classes, kernel_size=1, stride=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], classes, kernel_size=1, stride=1)

    def make_layer(self, block, in_channels, out_channels, num_blocks):
        layers = nn.Sequential(block(in_channels, out_channels))
        for i in range(num_blocks - 1):
            layers.append(block(out_channels, out_channels))
        return layers

    def execute(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x0_1 = self.conv0_1(jt.contrib.concat((x0_0, nn.upsample(x1_0,(x1_0.shape[2]*2,x1_0.shape[3]*2),'bilinear',align_corners=True)), dim=1))
        x1_1 = self.conv1_1(jt.contrib.concat((x1_0, nn.upsample(x2_0,(x2_0.shape[2]*2,x2_0.shape[3]*2),'bilinear',align_corners=True), nn.upsample(x0_1,(x0_1.shape[2]//2,x0_1.shape[3]//2),'bilinear',align_corners=True)), dim=1))
        x2_1 = self.conv2_1(jt.contrib.concat((x2_0, nn.upsample(x3_0,(x3_0.shape[2]*2,x3_0.shape[3]*2),'bilinear',align_corners=True), nn.upsample(x1_1,(x1_1.shape[2]//2,x1_1.shape[3]//2),'bilinear',align_corners=True)), dim=1))
        x3_1 = self.conv3_1(jt.contrib.concat((x3_0, nn.upsample(x4_0,(x4_0.shape[2]*2,x4_0.shape[3]*2),'bilinear',align_corners=True), nn.upsample(x2_1,(x2_1.shape[2]//2,x2_1.shape[3]//2),'bilinear',align_corners=True)), dim=1))

        x0_2 = self.conv0_2(jt.contrib.concat((x0_0, x0_1, nn.upsample(x1_1,(x1_1.shape[2]*2,x1_1.shape[3]*2),'bilinear',align_corners=True)), dim=1))
        x1_2 = self.conv1_2(jt.contrib.concat((x1_0, x1_1, nn.upsample(x2_1,(x2_1.shape[2]*2,x2_1.shape[3]*2),'bilinear',align_corners=True), nn.upsample(x0_2,(x0_2.shape[2]//2,x0_2.shape[3]//2),'bilinear',align_corners=True)), dim=1))
        x2_2 = self.conv2_2(jt.contrib.concat((x2_0, x2_1, nn.upsample(x3_1,(x3_1.shape[2]*2,x3_1.shape[3]*2),'bilinear',align_corners=True), nn.upsample(x1_2,(x1_2.shape[2]//2,x1_2.shape[3]//2),'bilinear',align_corners=True)), dim=1))

        x0_3 = self.conv0_3(jt.contrib.concat((x0_0, x0_1, x0_2, nn.upsample(x1_2,(x1_2.shape[2]*2,x1_2.shape[3]*2),'bilinear',align_corners=True)), dim=1))
        x1_3 = self.conv1_3(jt.contrib.concat((x1_0, x1_1, x1_2, nn.upsample(x2_2,(x2_2.shape[2]*2,x2_2.shape[3]*2),'bilinear',align_corners=True), nn.upsample(x0_3,(x0_3.shape[2]//2,x0_3.shape[3]//2),'bilinear',align_corners=True)), dim=1))

        x0_4 = self.conv0_4(jt.contrib.concat((x0_0, x0_1, x0_2, x0_3, nn.upsample(x1_3,(x1_3.shape[2]*2,x1_3.shape[3]*2),'bilinear',align_corners=True)), dim=1))

        x4_0f=self.conv0_4_1x1(x4_0)
        x3_1f=self.conv0_3_1x1(x3_1)
        x2_2f=self.conv0_2_1x1(x2_2)
        x1_3f=self.conv0_1_1x1(x1_3)
        finalf = jt.contrib.concat((nn.upsample(x4_0f,(x4_0f.shape[2]*16,x4_0f.shape[3]*16),'bilinear',align_corners=True), nn.upsample(x3_1f,(x3_1f.shape[2]*8,x3_1f.shape[3]*8),'bilinear',align_corners=True),
                                    nn.upsample(x2_2f,(x2_2f.shape[2]*4,x2_2f.shape[3]*4),'bilinear',align_corners=True), nn.upsample(x1_3f,(x1_3f.shape[2]*2,x1_3f.shape[3]*2),'bilinear',align_corners=True), x0_4), dim=1)
        finalf = self.conv0_4_final(finalf)

        if self.deep_supervision:
            out1 = self.final1(x0_1)
            out2 = self.final2(x0_2)
            out3 = self.final3(x0_3)
            out4 = self.final4(finalf)
            return [out1, out2, out3, out4]
        else:
            return self.final(finalf)