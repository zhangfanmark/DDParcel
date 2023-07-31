
# Copyright 2019 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# IMPORTS
import torch.nn as nn
import torch
import models.sub_module as sm
from collections import OrderedDict

class FastSurferCNN(nn.Module):
    """
    Network Definition of Fully Competitive Network network
    * Spatial view aggregation (input 7 slices of which only middle one gets segmented)
    * Same Number of filters per layer (normally 64)
    * Dense Connections in blocks
    * Unpooling instead of transpose convolutions
    * Concatenationes are replaced with Maxout (competitive dense blocks)
    * Global skip connections are fused by Maxout (global competition)
    * Loss Function (weighted Cross-Entropy and dice loss)
    """
    def __init__(self, params):
        super(FastSurferCNN, self).__init__()

        # Parameters for the Descending Arm
        self.encode1 = sm.CompetitiveEncoderBlockInput(params)
        params['num_channels'] = params['num_filters']
        self.encode2 = sm.CompetitiveEncoderBlock(params)
        self.encode3 = sm.CompetitiveEncoderBlock(params)
        self.encode4 = sm.CompetitiveEncoderBlock(params)
        self.bottleneck = sm.CompetitiveDenseBlock(params)

        # Parameters for the Ascending Arm
        params['num_channels'] = params['num_filters']
        self.decode4 = sm.CompetitiveDecoderBlock(params)
        self.decode3 = sm.CompetitiveDecoderBlock(params)
        self.decode2 = sm.CompetitiveDecoderBlock(params)
        self.decode1 = sm.CompetitiveDecoderBlock(params)

        params['num_channels'] = params['num_filters']
        self.classifier = sm.ClassifierBlock(params)

        # Code for Network Initialization

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Computational graph
        :param tensor x: input image
        :return tensor: prediction logits
        """
        encoder_output1, skip_encoder_1, indices_1 = self.encode1.forward(x)
        encoder_output2, skip_encoder_2, indices_2 = self.encode2.forward(encoder_output1)
        encoder_output3, skip_encoder_3, indices_3 = self.encode3.forward(encoder_output2)
        encoder_output4, skip_encoder_4, indices_4 = self.encode4.forward(encoder_output3)

        bottleneck = self.bottleneck(encoder_output4)

        decoder_output4 = self.decode4.forward(bottleneck, skip_encoder_4, indices_4)
        decoder_output3 = self.decode3.forward(decoder_output4, skip_encoder_3, indices_3)
        decoder_output2 = self.decode2.forward(decoder_output3, skip_encoder_2, indices_2)
        decoder_output1 = self.decode1.forward(decoder_output2, skip_encoder_1, indices_1)

        logits = self.classifier.forward(decoder_output1)

        return logits

class FastSurferCNN_return_all(nn.Module):
    """
    Network Definition of Fully Competitive Network network
    * Spatial view aggregation (input 7 slices of which only middle one gets segmented)
    * Same Number of filters per layer (normally 64)
    * Dense Connections in blocks
    * Unpooling instead of transpose convolutions
    * Concatenationes are replaced with Maxout (competitive dense blocks)
    * Global skip connections are fused by Maxout (global competition)
    * Loss Function (weighted Cross-Entropy and dice loss)
    """
    def __init__(self, params):
        super(FastSurferCNN_return_all, self).__init__()

        # Parameters for the Descending Arm
        self.encode1 = sm.CompetitiveEncoderBlockInput(params)
        params['num_channels'] = params['num_filters']
        self.encode2 = sm.CompetitiveEncoderBlock(params)
        self.encode3 = sm.CompetitiveEncoderBlock(params)
        self.encode4 = sm.CompetitiveEncoderBlock(params)
        self.bottleneck = sm.CompetitiveDenseBlock(params)

        # Parameters for the Ascending Arm
        params['num_channels'] = params['num_filters']
        self.decode4 = sm.CompetitiveDecoderBlock(params)
        self.decode3 = sm.CompetitiveDecoderBlock(params)
        self.decode2 = sm.CompetitiveDecoderBlock(params)
        self.decode1 = sm.CompetitiveDecoderBlock(params)

        params['num_channels'] = params['num_filters']
        self.classifier = sm.ClassifierBlock(params)

        # Code for Network Initialization

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Computational graph
        :param tensor x: input image
        :return tensor: prediction logits
        """
        encoder_output1, skip_encoder_1, indices_1 = self.encode1.forward(x)
        encoder_output2, skip_encoder_2, indices_2 = self.encode2.forward(encoder_output1)
        encoder_output3, skip_encoder_3, indices_3 = self.encode3.forward(encoder_output2)
        encoder_output4, skip_encoder_4, indices_4 = self.encode4.forward(encoder_output3)

        bottleneck = self.bottleneck(encoder_output4)

        decoder_output4 = self.decode4.forward(bottleneck, skip_encoder_4, indices_4)
        decoder_output3 = self.decode3.forward(decoder_output4, skip_encoder_3, indices_3)
        decoder_output2 = self.decode2.forward(decoder_output3, skip_encoder_2, indices_2)
        decoder_output1 = self.decode1.forward(decoder_output2, skip_encoder_1, indices_1)

        logits = self.classifier.forward(decoder_output1)

        return logits, \
               encoder_output1, skip_encoder_1, indices_1, \
               encoder_output2, skip_encoder_2, indices_2, \
               encoder_output3, skip_encoder_3, indices_3, \
               encoder_output4, skip_encoder_4, indices_4, \
               bottleneck, \
               decoder_output4, decoder_output3, decoder_output2, decoder_output1

class FastSurferCNN_no_classifer(nn.Module):
    """
    Network Definition of Fully Competitive Network network
    * Spatial view aggregation (input 7 slices of which only middle one gets segmented)
    * Same Number of filters per layer (normally 64)
    * Dense Connections in blocks
    * Unpooling instead of transpose convolutions
    * Concatenationes are replaced with Maxout (competitive dense blocks)
    * Global skip connections are fused by Maxout (global competition)
    * Loss Function (weighted Cross-Entropy and dice loss)
    """
    def __init__(self, params):
        super(FastSurferCNN_no_classifer, self).__init__()

        # Parameters for the Descending Arm
        self.encode1 = sm.CompetitiveEncoderBlockInput(params)
        params['num_channels'] = params['num_filters']
        self.encode2 = sm.CompetitiveEncoderBlock(params)
        self.encode3 = sm.CompetitiveEncoderBlock(params)
        self.encode4 = sm.CompetitiveEncoderBlock(params)
        self.bottleneck = sm.CompetitiveDenseBlock(params)

        # Parameters for the Ascending Arm
        params['num_channels'] = params['num_filters']
        self.decode4 = sm.CompetitiveDecoderBlock(params)
        self.decode3 = sm.CompetitiveDecoderBlock(params)
        self.decode2 = sm.CompetitiveDecoderBlock(params)
        self.decode1 = sm.CompetitiveDecoderBlock(params)

        # Code for Network Initialization

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Computational graph
        :param tensor x: input image
        :return tensor: prediction logits
        """
        encoder_output1, skip_encoder_1, indices_1 = self.encode1.forward(x)
        encoder_output2, skip_encoder_2, indices_2 = self.encode2.forward(encoder_output1)
        encoder_output3, skip_encoder_3, indices_3 = self.encode3.forward(encoder_output2)
        encoder_output4, skip_encoder_4, indices_4 = self.encode4.forward(encoder_output3)

        bottleneck = self.bottleneck(encoder_output4)

        decoder_output4 = self.decode4.forward(bottleneck, skip_encoder_4, indices_4)
        decoder_output3 = self.decode3.forward(decoder_output4, skip_encoder_3, indices_3)
        decoder_output2 = self.decode2.forward(decoder_output3, skip_encoder_2, indices_2)
        decoder_output1 = self.decode1.forward(decoder_output2, skip_encoder_1, indices_1)

        return decoder_output1


class FastSurferCNN_Fuse_Last_Layer(nn.Module):
    """
    Network Definition of Fully Competitive Network network
    * Spatial view aggregation (input 7 slices of which only middle one gets segmented)
    * Same Number of filters per layer (normally 64)
    * Dense Connections in blocks
    * Unpooling instead of transpose convolutions
    * Concatenationes are replaced with Maxout (competitive dense blocks)
    * Global skip connections are fused by Maxout (global competition)
    * Loss Function (weighted Cross-Entropy and dice loss)
    """
    def __init__(self, params):
        super(FastSurferCNN_Fuse_Last_Layer, self).__init__()

        num_channels = params['num_channels']
        # Parameters for the Descending Arm
        self.FastSurferCNNs = nn.ModuleList()
        for idx in range(params['num_modality']):
            params['num_channels'] = num_channels
            self.FastSurferCNNs.append(FastSurferCNN_no_classifer(params))

        self.fusion_layer = nn.Conv2d(params['num_filters'] * params['num_modality'], params['num_filters'], params['kernel_c'], params['stride_conv'])  # To generate logits

        params['num_channels'] = params['num_filters']
        self.classifier = sm.ClassifierBlock(params)

    def forward(self, x):
        """
        Computational graph
        :param tensor x: input image
        :return tensor: prediction logits
        """
        decoder_outputs = []
        for idx, fscnn in enumerate(self.FastSurferCNNs):
            decoder_outputs.append(fscnn(x[:, (idx*7):((idx+1) * 7), :, :]))

        decoder_outputs_fused = self.fusion_layer(torch.cat(decoder_outputs, dim=1))

        logits = self.classifier.forward(decoder_outputs_fused)

        return logits

class FastSurferCNN_Fuse_Unet(nn.Module):
    """
    Network Definition of Fully Competitive Network network
    * Spatial view aggregation (input 7 slices of which only middle one gets segmented)
    * Same Number of filters per layer (normally 64)
    * Dense Connections in blocks
    * Unpooling instead of transpose convolutions
    * Concatenationes are replaced with Maxout (competitive dense blocks)
    * Global skip connections are fused by Maxout (global competition)
    * Loss Function (weighted Cross-Entropy and dice loss)
    """
    def __init__(self, params):
        super(FastSurferCNN_Fuse_Unet, self).__init__()

        num_channels = params['num_channels']

        # Load backbones
        self.FastSurferCNNs = nn.ModuleList()
        for idx in range(params['num_modality']):
            params['num_channels'] = num_channels
            backbone_model = FastSurferCNN_return_all(params)
            if params['backbone_model'] is not None:
                # backbone_model = nn.DataParallel(backbone_model)
                model_state = torch.load(params['backbone_model'][idx])
                new_state_dict = OrderedDict()
                for k, v in model_state["model_state_dict"].items():
                    if k[:7] == "module.":
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v
                backbone_model.load_state_dict(new_state_dict)
                for param in backbone_model.parameters():
                    param.requires_grad = False
            self.FastSurferCNNs.append(backbone_model)

        params['num_channels'] = num_channels
        # Parameters for the Descending Arm
        self.encode1 = sm.CompetitiveEncoderBlockInput(params)
        params['num_channels'] = params['num_filters']
        self.encode2 = sm.CompetitiveEncoderBlock(params)
        self.encode3 = sm.CompetitiveEncoderBlock(params)
        self.encode4 = sm.CompetitiveEncoderBlock(params)
        self.bottleneck = sm.CompetitiveDenseBlock(params)

        # Parameters for the Ascending Arm
        params['num_channels'] = params['num_filters']
        self.decode4 = sm.CompetitiveDecoderBlock(params)
        self.decode3 = sm.CompetitiveDecoderBlock(params)
        self.decode2 = sm.CompetitiveDecoderBlock(params)
        self.decode1 = sm.CompetitiveDecoderBlock(params)

        self.fusion1 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion2 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion3 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion4 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion5 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])

        params['num_channels'] = params['num_filters']
        self.classifier = sm.ClassifierBlock(params)

        # Code for Network Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Computational graph
        :param tensor x: input image
        :return tensor: prediction logits
        """

        # return [0] logits, \
        #        [1] encoder_output1,  [2] skip_encoder_1,  [3] indices_1, \
        #        [4] encoder_output2,  [5] skip_encoder_2,  [6] indices_2, \
        #        [7] encoder_output3,  [8] skip_encoder_3,  [9] indices_3, \
        #        [10] encoder_output4, [11] skip_encoder_4, [12] indices_4, \
        #        [13] bottleneck, \
        #        [14] decoder_output4, [15] decoder_output3, [16] decoder_output2, [17] decoder_output1

        returns = []
        for idx in range(len(self.FastSurferCNNs)):
            fscnn = self.FastSurferCNNs[idx]
            returns.append(fscnn(x[:, (idx*7):((idx+1) * 7), :, :]))

        idx = 0
        encoder_output1, skip_encoder_1, indices_1 = self.encode1.forward(x[:, (idx*7):((idx+1) * 7), :, :])

        encoder_output1 = torch.unsqueeze(encoder_output1, 4)
        # skip_encoder_1 = torch.unsqueeze(skip_encoder_1, 4)
        for idx in range(1, len(self.FastSurferCNNs)):
            encoder_output1 = torch.cat((encoder_output1, returns[idx][1].unsqueeze(4)), dim=4)
            # skip_encoder_1 = torch.cat((skip_encoder_1, returns[idx][2].unsqueeze(4)), dim=4)
        encoder_output1, _ = torch.max(encoder_output1, 4)
        # skip_encoder_1, _ = torch.max(skip_encoder_1, 4)

        encoder_output1 = self.fusion1(encoder_output1)

        encoder_output2, skip_encoder_2, indices_2 = self.encode2.forward(encoder_output1)

        encoder_output2 = torch.unsqueeze(encoder_output2, 4)
        # skip_encoder_2 = torch.unsqueeze(skip_encoder_2, 4)
        for idx in range(1, len(self.FastSurferCNNs)):
            encoder_output2 = torch.cat((encoder_output2, returns[idx][4].unsqueeze(4)), dim=4)
            # skip_encoder_2 = torch.cat((skip_encoder_2, returns[idx][5].unsqueeze(4)), dim=4)
        encoder_output2, _ = torch.max(encoder_output2, 4)
        # skip_encoder_2, _ = torch.max(skip_encoder_2, 4)

        encoder_output2 = self.fusion2(encoder_output2)

        encoder_output3, skip_encoder_3, indices_3 = self.encode3.forward(encoder_output2)

        encoder_output3 = torch.unsqueeze(encoder_output3, 4)
        # skip_encoder_3 = torch.unsqueeze(skip_encoder_3, 4)
        for idx in range(1, len(self.FastSurferCNNs)):
            encoder_output3 = torch.cat((encoder_output3, returns[idx][7].unsqueeze(4)), dim=4)
            # skip_encoder_3 = torch.cat((skip_encoder_3, returns[idx][8].unsqueeze(4)), dim=4)
        encoder_output3, _ = torch.max(encoder_output3, 4)
        # skip_encoder_3, _ = torch.max(skip_encoder_3, 4)

        encoder_output3 = self.fusion3(encoder_output3)

        encoder_output4, skip_encoder_4, indices_4 = self.encode4.forward(encoder_output3)

        encoder_output4 = torch.unsqueeze(encoder_output4, 4)
        # skip_encoder_4 = torch.unsqueeze(skip_encoder_4, 4)
        for idx in range(1, len(self.FastSurferCNNs)):
            encoder_output4 = torch.cat((encoder_output4, returns[idx][10].unsqueeze(4)), dim=4)
            # skip_encoder_4 = torch.cat((skip_encoder_4, returns[idx][11].unsqueeze(4)), dim=4)
        encoder_output4, _ = torch.max(encoder_output4, 4)
        # skip_encoder_4, _ = torch.max(skip_encoder_4, 4)

        encoder_output4 = self.fusion4(encoder_output4)

        bottleneck = self.bottleneck(encoder_output4)

        bottleneck = torch.unsqueeze(bottleneck, 4)
        for idx in range(1, len(self.FastSurferCNNs)):
            bottleneck = torch.cat((bottleneck, returns[idx][13].unsqueeze(4)), dim=4)
        bottleneck, _ = torch.max(bottleneck, 4)

        bottleneck = self.fusion5(bottleneck)

        decoder_output4 = self.decode4.forward(bottleneck, skip_encoder_4, indices_4)
        decoder_output3 = self.decode3.forward(decoder_output4, skip_encoder_3, indices_3)
        decoder_output2 = self.decode2.forward(decoder_output3, skip_encoder_2, indices_2)
        decoder_output1 = self.decode1.forward(decoder_output2, skip_encoder_1, indices_1)

        logits = self.classifier.forward(decoder_output1)

        return logits

class FastSurferCNN_Fuse_Unet_v1(nn.Module):
    """
    Network Definition of Fully Competitive Network network
    * Spatial view aggregation (input 7 slices of which only middle one gets segmented)
    * Same Number of filters per layer (normally 64)
    * Dense Connections in blocks
    * Unpooling instead of transpose convolutions
    * Concatenationes are replaced with Maxout (competitive dense blocks)
    * Global skip connections are fused by Maxout (global competition)
    * Loss Function (weighted Cross-Entropy and dice loss)
    """
    def __init__(self, params):
        super(FastSurferCNN_Fuse_Unet_v1, self).__init__()

        num_channels = params['num_channels']

        # Load backbones
        self.FastSurferCNNs = nn.ModuleList()
        for idx in range(params['num_modality']-1):
            params['num_channels'] = num_channels
            backbone_model = FastSurferCNN_return_all(params)
            if params['backbone_model'] is not None:
                # backbone_model = nn.DataParallel(backbone_model)
                print('Loading: %s' % params['backbone_model'][idx+1])
                model_state = torch.load(params['backbone_model'][idx+1])
                new_state_dict = OrderedDict()
                for k, v in model_state["model_state_dict"].items():
                    if k[:7] == "module.":
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v
                backbone_model.load_state_dict(new_state_dict)
                for param in backbone_model.parameters():
                    param.requires_grad = False
            self.FastSurferCNNs.append(backbone_model)

        params['num_channels'] = num_channels
        # Parameters for the Descending Arm
        self.encode1 = sm.CompetitiveEncoderBlockInput(params)
        params['num_channels'] = params['num_filters']
        self.encode2 = sm.CompetitiveEncoderBlock(params)
        self.encode3 = sm.CompetitiveEncoderBlock(params)
        self.encode4 = sm.CompetitiveEncoderBlock(params)
        self.bottleneck = sm.CompetitiveDenseBlock(params)

        # Parameters for the Ascending Arm
        params['num_channels'] = params['num_filters']
        self.decode4 = sm.CompetitiveDecoderBlock(params)
        self.decode3 = sm.CompetitiveDecoderBlock(params)
        self.decode2 = sm.CompetitiveDecoderBlock(params)
        self.decode1 = sm.CompetitiveDecoderBlock(params)

        self.fusion1 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion2 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion3 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion4 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion5 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])

        params['num_channels'] = params['num_filters']
        self.classifier = sm.ClassifierBlock(params)

        # Code for Network Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Computational graph
        :param tensor x: input image
        :return tensor: prediction logits
        """

        # return [0] logits, \
        #        [1] encoder_output1,  [2] skip_encoder_1,  [3] indices_1, \
        #        [4] encoder_output2,  [5] skip_encoder_2,  [6] indices_2, \
        #        [7] encoder_output3,  [8] skip_encoder_3,  [9] indices_3, \
        #        [10] encoder_output4, [11] skip_encoder_4, [12] indices_4, \
        #        [13] bottleneck, \
        #        [14] decoder_output4, [15] decoder_output3, [16] decoder_output2, [17] decoder_output1

        returns = []
        for idx in range(len(self.FastSurferCNNs)):
            fscnn = self.FastSurferCNNs[idx]
            returns.append(fscnn(x[:, ((idx+1)*7):((idx+2) * 7), :, :]))

        idx = 0
        encoder_output1, skip_encoder_1, indices_1 = self.encode1.forward(x[:, (idx*7):((idx+1) * 7), :, :])

        encoder_output1 = torch.unsqueeze(encoder_output1, 4)
        # skip_encoder_1 = torch.unsqueeze(skip_encoder_1, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output1 = torch.cat((encoder_output1, returns[idx][1].unsqueeze(4)), dim=4)
            # skip_encoder_1 = torch.cat((skip_encoder_1, returns[idx][2].unsqueeze(4)), dim=4)
        encoder_output1, _ = torch.max(encoder_output1, 4)
        # skip_encoder_1, _ = torch.max(skip_encoder_1, 4)

        encoder_output1 = self.fusion1(encoder_output1)

        encoder_output2, skip_encoder_2, indices_2 = self.encode2.forward(encoder_output1)

        encoder_output2 = torch.unsqueeze(encoder_output2, 4)
        # skip_encoder_2 = torch.unsqueeze(skip_encoder_2, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output2 = torch.cat((encoder_output2, returns[idx][4].unsqueeze(4)), dim=4)
            # skip_encoder_2 = torch.cat((skip_encoder_2, returns[idx][5].unsqueeze(4)), dim=4)
        encoder_output2, _ = torch.max(encoder_output2, 4)
        # skip_encoder_2, _ = torch.max(skip_encoder_2, 4)

        encoder_output2 = self.fusion2(encoder_output2)

        encoder_output3, skip_encoder_3, indices_3 = self.encode3.forward(encoder_output2)

        encoder_output3 = torch.unsqueeze(encoder_output3, 4)
        # skip_encoder_3 = torch.unsqueeze(skip_encoder_3, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output3 = torch.cat((encoder_output3, returns[idx][7].unsqueeze(4)), dim=4)
            # skip_encoder_3 = torch.cat((skip_encoder_3, returns[idx][8].unsqueeze(4)), dim=4)
        encoder_output3, _ = torch.max(encoder_output3, 4)
        # skip_encoder_3, _ = torch.max(skip_encoder_3, 4)

        encoder_output3 = self.fusion3(encoder_output3)

        encoder_output4, skip_encoder_4, indices_4 = self.encode4.forward(encoder_output3)

        encoder_output4 = torch.unsqueeze(encoder_output4, 4)
        # skip_encoder_4 = torch.unsqueeze(skip_encoder_4, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output4 = torch.cat((encoder_output4, returns[idx][10].unsqueeze(4)), dim=4)
            # skip_encoder_4 = torch.cat((skip_encoder_4, returns[idx][11].unsqueeze(4)), dim=4)
        encoder_output4, _ = torch.max(encoder_output4, 4)
        # skip_encoder_4, _ = torch.max(skip_encoder_4, 4)

        encoder_output4 = self.fusion4(encoder_output4)

        bottleneck = self.bottleneck(encoder_output4)

        bottleneck = torch.unsqueeze(bottleneck, 4)
        for idx in range(len(self.FastSurferCNNs)):
            bottleneck = torch.cat((bottleneck, returns[idx][13].unsqueeze(4)), dim=4)
        bottleneck, _ = torch.max(bottleneck, 4)

        bottleneck = self.fusion5(bottleneck)

        decoder_output4 = self.decode4.forward(bottleneck, skip_encoder_4, indices_4)
        decoder_output3 = self.decode3.forward(decoder_output4, skip_encoder_3, indices_3)
        decoder_output2 = self.decode2.forward(decoder_output3, skip_encoder_2, indices_2)
        decoder_output1 = self.decode1.forward(decoder_output2, skip_encoder_1, indices_1)

        logits = self.classifier.forward(decoder_output1)

        for ind_return in returns:
            logits += ind_return[0]

        return logits

class FastSurferCNN_Fuse_Unet_v2(nn.Module):
    """
    Network Definition of Fully Competitive Network network
    * Spatial view aggregation (input 7 slices of which only middle one gets segmented)
    * Same Number of filters per layer (normally 64)
    * Dense Connections in blocks
    * Unpooling instead of transpose convolutions
    * Concatenationes are replaced with Maxout (competitive dense blocks)
    * Global skip connections are fused by Maxout (global competition)
    * Loss Function (weighted Cross-Entropy and dice loss)
    """
    def __init__(self, params):
        super(FastSurferCNN_Fuse_Unet_v2, self).__init__()

        num_channels = params['num_channels']

        params['num_channels'] = params['num_modality'] * 7
        # Parameters for the Descending Arm
        self.encode1 = sm.CompetitiveEncoderBlockInput(params)
        params['num_channels'] = params['num_filters']
        self.encode2 = sm.CompetitiveEncoderBlock(params)
        self.encode3 = sm.CompetitiveEncoderBlock(params)
        self.encode4 = sm.CompetitiveEncoderBlock(params)
        self.bottleneck = sm.CompetitiveDenseBlock(params)

        # Parameters for the Ascending Arm
        params['num_channels'] = params['num_filters']
        self.decode4 = sm.CompetitiveDecoderBlock(params)
        self.decode3 = sm.CompetitiveDecoderBlock(params)
        self.decode2 = sm.CompetitiveDecoderBlock(params)
        self.decode1 = sm.CompetitiveDecoderBlock(params)

        self.fusion1 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion2 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion3 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion4 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion5 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])

        params['num_channels'] = params['num_filters']
        self.classifier = sm.ClassifierBlock(params)

        # Code for Network Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Load backbones
        self.FastSurferCNNs = nn.ModuleList()
        for idx in range(params['num_modality']):
            params['num_channels'] = num_channels
            backbone_model = FastSurferCNN_return_all(params)
            if params['backbone_model'] is not None:
                # backbone_model = nn.DataParallel(backbone_model)
                print('Loading: %s' % params['backbone_model'][idx])
                model_state = torch.load(params['backbone_model'][idx])
                new_state_dict = OrderedDict()
                for k, v in model_state["model_state_dict"].items():
                    if k[:7] == "module.":
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v
                backbone_model.load_state_dict(new_state_dict)
                for param in backbone_model.parameters():
                    param.requires_grad = False
            else:
                print('No Backbone!!!')
            self.FastSurferCNNs.append(backbone_model)

        print('Initialize Fuse Unet v2 done!')

    def forward(self, x):
        """
        Computational graph
        :param tensor x: input image
        :return tensor: prediction logits
        """

        # return [0] logits, \
        #        [1] encoder_output1,  [2] skip_encoder_1,  [3] indices_1, \
        #        [4] encoder_output2,  [5] skip_encoder_2,  [6] indices_2, \
        #        [7] encoder_output3,  [8] skip_encoder_3,  [9] indices_3, \
        #        [10] encoder_output4, [11] skip_encoder_4, [12] indices_4, \
        #        [13] bottleneck, \
        #        [14] decoder_output4, [15] decoder_output3, [16] decoder_output2, [17] decoder_output1

        returns = []
        for idx in range(len(self.FastSurferCNNs)):
            fscnn = self.FastSurferCNNs[idx]
            returns.append(fscnn(x[:, ((idx)*7):((idx+1) * 7), :, :]))

        encoder_output1, skip_encoder_1, indices_1 = self.encode1.forward(x[:, :, :, :])

        encoder_output1 = torch.unsqueeze(encoder_output1, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output1 = torch.cat((encoder_output1, returns[idx][1].unsqueeze(4)), dim=4)
        encoder_output1, _ = torch.max(encoder_output1, 4)
        encoder_output1 = self.fusion1(encoder_output1)

        encoder_output2, skip_encoder_2, indices_2 = self.encode2.forward(encoder_output1)

        encoder_output2 = torch.unsqueeze(encoder_output2, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output2 = torch.cat((encoder_output2, returns[idx][4].unsqueeze(4)), dim=4)
        encoder_output2, _ = torch.max(encoder_output2, 4)
        encoder_output2 = self.fusion2(encoder_output2)

        encoder_output3, skip_encoder_3, indices_3 = self.encode3.forward(encoder_output2)

        encoder_output3 = torch.unsqueeze(encoder_output3, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output3 = torch.cat((encoder_output3, returns[idx][7].unsqueeze(4)), dim=4)
        encoder_output3, _ = torch.max(encoder_output3, 4)
        encoder_output3 = self.fusion3(encoder_output3)

        encoder_output4, skip_encoder_4, indices_4 = self.encode4.forward(encoder_output3)

        encoder_output4 = torch.unsqueeze(encoder_output4, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output4 = torch.cat((encoder_output4, returns[idx][10].unsqueeze(4)), dim=4)
        encoder_output4, _ = torch.max(encoder_output4, 4)
        encoder_output4 = self.fusion4(encoder_output4)

        bottleneck = self.bottleneck(encoder_output4)

        bottleneck = torch.unsqueeze(bottleneck, 4)
        for idx in range(len(self.FastSurferCNNs)):
            bottleneck = torch.cat((bottleneck, returns[idx][13].unsqueeze(4)), dim=4)
        bottleneck, _ = torch.max(bottleneck, 4)
        bottleneck = self.fusion5(bottleneck)

        decoder_output4 = self.decode4.forward(bottleneck, skip_encoder_4, indices_4)
        decoder_output3 = self.decode3.forward(decoder_output4, skip_encoder_3, indices_3)
        decoder_output2 = self.decode2.forward(decoder_output3, skip_encoder_2, indices_2)
        decoder_output1 = self.decode1.forward(decoder_output2, skip_encoder_1, indices_1)

        logits_list = []
        logits = self.classifier.forward(decoder_output1)
        logits_list.append(logits)
        for ind_return in returns:
            logits += ind_return[0]
            logits_list.append(ind_return[0])

        return logits

class FastSurferCNN_Fuse_Unet_v2_extended(nn.Module):
    """
    Network Definition of Fully Competitive Network network
    * Spatial view aggregation (input 7 slices of which only middle one gets segmented)
    * Same Number of filters per layer (normally 64)
    * Dense Connections in blocks
    * Unpooling instead of transpose convolutions
    * Concatenationes are replaced with Maxout (competitive dense blocks)
    * Global skip connections are fused by Maxout (global competition)
    * Loss Function (weighted Cross-Entropy and dice loss)
    """
    def __init__(self, params):
        super(FastSurferCNN_Fuse_Unet_v2_extended, self).__init__()

        num_channels = params['num_channels']

        params['num_channels'] = params['num_modality'] * 7
        # Parameters for the Descending Arm
        self.encode1 = sm.CompetitiveEncoderBlockInput(params)
        params['num_channels'] = params['num_filters']
        self.encode2 = sm.CompetitiveEncoderBlock(params)
        self.encode3 = sm.CompetitiveEncoderBlock(params)
        self.encode4 = sm.CompetitiveEncoderBlock(params)
        self.bottleneck = sm.CompetitiveDenseBlock(params)

        # Parameters for the Ascending Arm
        params['num_channels'] = params['num_filters']
        self.decode4 = sm.CompetitiveDecoderBlock(params)
        self.decode3 = sm.CompetitiveDecoderBlock(params)
        self.decode2 = sm.CompetitiveDecoderBlock(params)
        self.decode1 = sm.CompetitiveDecoderBlock(params)

        self.fusion1 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion2 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion3 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion4 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion5 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])

        params['num_channels'] = params['num_filters']
        self.classifier = sm.ClassifierBlock(params)

        # Code for Network Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Load backbones
        self.FastSurferCNNs = nn.ModuleList()
        for idx in range(params['num_modality']):
            params['num_channels'] = num_channels
            backbone_model = FastSurferCNN_return_all(params)
            if params['backbone_model'] is not None:
                # backbone_model = nn.DataParallel(backbone_model)
                print('Loading: %s' % params['backbone_model'][idx])
                model_state = torch.load(params['backbone_model'][idx])
                new_state_dict = OrderedDict()
                for k, v in model_state["model_state_dict"].items():
                    if k[:7] == "module.":
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v
                backbone_model.load_state_dict(new_state_dict)
                for param in backbone_model.parameters():
                    param.requires_grad = False
            self.FastSurferCNNs.append(backbone_model)

    def forward(self, x):
        """
        Computational graph
        :param tensor x: input image
        :return tensor: prediction logits
        """

        # return [0] logits, \
        #        [1] encoder_output1,  [2] skip_encoder_1,  [3] indices_1, \
        #        [4] encoder_output2,  [5] skip_encoder_2,  [6] indices_2, \
        #        [7] encoder_output3,  [8] skip_encoder_3,  [9] indices_3, \
        #        [10] encoder_output4, [11] skip_encoder_4, [12] indices_4, \
        #        [13] bottleneck, \
        #        [14] decoder_output4, [15] decoder_output3, [16] decoder_output2, [17] decoder_output1

        returns = []
        for idx in range(len(self.FastSurferCNNs)):
            fscnn = self.FastSurferCNNs[idx]
            returns.append(fscnn(x[:, ((idx)*7):((idx+1) * 7), :, :]))

        encoder_output1, skip_encoder_1, indices_1 = self.encode1.forward(x[:, :, :, :])

        encoder_output1 = torch.unsqueeze(encoder_output1, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output1 = torch.cat((encoder_output1, returns[idx][1].unsqueeze(4)), dim=4)
        encoder_output1, _ = torch.max(encoder_output1, 4)
        encoder_output1 = self.fusion1(encoder_output1)

        encoder_output2, skip_encoder_2, indices_2 = self.encode2.forward(encoder_output1)

        encoder_output2 = torch.unsqueeze(encoder_output2, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output2 = torch.cat((encoder_output2, returns[idx][4].unsqueeze(4)), dim=4)
        encoder_output2, _ = torch.max(encoder_output2, 4)
        encoder_output2 = self.fusion2(encoder_output2)

        encoder_output3, skip_encoder_3, indices_3 = self.encode3.forward(encoder_output2)

        encoder_output3 = torch.unsqueeze(encoder_output3, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output3 = torch.cat((encoder_output3, returns[idx][7].unsqueeze(4)), dim=4)
        encoder_output3, _ = torch.max(encoder_output3, 4)
        encoder_output3 = self.fusion3(encoder_output3)

        encoder_output4, skip_encoder_4, indices_4 = self.encode4.forward(encoder_output3)

        encoder_output4 = torch.unsqueeze(encoder_output4, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output4 = torch.cat((encoder_output4, returns[idx][10].unsqueeze(4)), dim=4)
        encoder_output4, _ = torch.max(encoder_output4, 4)
        encoder_output4 = self.fusion4(encoder_output4)

        bottleneck = self.bottleneck(encoder_output4)

        bottleneck = torch.unsqueeze(bottleneck, 4)
        for idx in range(len(self.FastSurferCNNs)):
            bottleneck = torch.cat((bottleneck, returns[idx][13].unsqueeze(4)), dim=4)
        bottleneck, _ = torch.max(bottleneck, 4)
        bottleneck = self.fusion5(bottleneck)

        decoder_output4 = self.decode4.forward(bottleneck, skip_encoder_4, indices_4)
        decoder_output3 = self.decode3.forward(decoder_output4, skip_encoder_3, indices_3)
        decoder_output2 = self.decode2.forward(decoder_output3, skip_encoder_2, indices_2)
        decoder_output1 = self.decode1.forward(decoder_output2, skip_encoder_1, indices_1)

        logits_list = []
        logits = self.classifier.forward(decoder_output1)
        logits_list.append(logits)
        for ind_return in returns:
            logits += ind_return[0]
            logits_list.append(ind_return[0])

        return logits, logits_list

class FastSurferCNN_Fuse_Unet_v3(nn.Module):
    """
    Network Definition of Fully Competitive Network network
    * Spatial view aggregation (input 7 slices of which only middle one gets segmented)
    * Same Number of filters per layer (normally 64)
    * Dense Connections in blocks
    * Unpooling instead of transpose convolutions
    * Concatenationes are replaced with Maxout (competitive dense blocks)
    * Global skip connections are fused by Maxout (global competition)
    * Loss Function (weighted Cross-Entropy and dice loss)
    """
    def __init__(self, params):
        super(FastSurferCNN_Fuse_Unet_v3, self).__init__()

        num_channels = params['num_channels']

        params['num_channels'] = params['num_modality'] * 7
        # Parameters for the Descending Arm
        self.encode1 = sm.CompetitiveEncoderBlockInput(params)
        params['num_channels'] = params['num_filters']
        self.encode2 = sm.CompetitiveEncoderBlock(params)
        self.encode3 = sm.CompetitiveEncoderBlock(params)
        self.encode4 = sm.CompetitiveEncoderBlock(params)
        self.bottleneck = sm.CompetitiveDenseBlock(params)

        # Parameters for the Ascending Arm
        params['num_channels'] = params['num_filters']
        self.decode4 = sm.CompetitiveDecoderBlock(params)
        self.decode3 = sm.CompetitiveDecoderBlock(params)
        self.decode2 = sm.CompetitiveDecoderBlock(params)
        self.decode1 = sm.CompetitiveDecoderBlock(params)

        self.fusion1 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion2 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion3 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion4 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion5 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])

        self.fusion_layer = nn.Conv2d(params['num_filters'] * (params['num_modality'] + 1), params['num_filters'], params['kernel_c'], params['stride_conv'])  # To generate logits

        params['num_channels'] = params['num_filters']
        self.classifier = sm.ClassifierBlock(params)

        # Code for Network Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Load backbones
        self.FastSurferCNNs = nn.ModuleList()
        for idx in range(params['num_modality']):
            params['num_channels'] = num_channels
            backbone_model = FastSurferCNN_return_all(params)
            if params['backbone_model'] is not None:
                # backbone_model = nn.DataParallel(backbone_model)
                print('Loading: %s' % params['backbone_model'][idx])
                model_state = torch.load(params['backbone_model'][idx])
                new_state_dict = OrderedDict()
                for k, v in model_state["model_state_dict"].items():
                    if k[:7] == "module.":
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v
                backbone_model.load_state_dict(new_state_dict)
                for param in backbone_model.parameters():
                    param.requires_grad = False
            else:
                print('No Backbone!!!')
            self.FastSurferCNNs.append(backbone_model)

        print('Initialize Fuse Unet v3 done!')

    def forward(self, x):
        """
        Computational graph
        :param tensor x: input image
        :return tensor: prediction logits
        """

        # return [0] logits, \
        #        [1] encoder_output1,  [2] skip_encoder_1,  [3] indices_1, \
        #        [4] encoder_output2,  [5] skip_encoder_2,  [6] indices_2, \
        #        [7] encoder_output3,  [8] skip_encoder_3,  [9] indices_3, \
        #        [10] encoder_output4, [11] skip_encoder_4, [12] indices_4, \
        #        [13] bottleneck, \
        #        [14] decoder_output4, [15] decoder_output3, [16] decoder_output2, [17] decoder_output1

        returns = []
        for idx in range(len(self.FastSurferCNNs)):
            fscnn = self.FastSurferCNNs[idx]
            returns.append(fscnn(x[:, ((idx)*7):((idx+1) * 7), :, :]))

        encoder_output1, skip_encoder_1, indices_1 = self.encode1.forward(x[:, :, :, :])

        encoder_output1 = torch.unsqueeze(encoder_output1, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output1 = torch.cat((encoder_output1, returns[idx][1].unsqueeze(4)), dim=4)
        encoder_output1, _ = torch.max(encoder_output1, 4)
        encoder_output1 = self.fusion1(encoder_output1)

        encoder_output2, skip_encoder_2, indices_2 = self.encode2.forward(encoder_output1)

        encoder_output2 = torch.unsqueeze(encoder_output2, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output2 = torch.cat((encoder_output2, returns[idx][4].unsqueeze(4)), dim=4)
        encoder_output2, _ = torch.max(encoder_output2, 4)
        encoder_output2 = self.fusion2(encoder_output2)

        encoder_output3, skip_encoder_3, indices_3 = self.encode3.forward(encoder_output2)

        encoder_output3 = torch.unsqueeze(encoder_output3, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output3 = torch.cat((encoder_output3, returns[idx][7].unsqueeze(4)), dim=4)
        encoder_output3, _ = torch.max(encoder_output3, 4)
        encoder_output3 = self.fusion3(encoder_output3)

        encoder_output4, skip_encoder_4, indices_4 = self.encode4.forward(encoder_output3)

        encoder_output4 = torch.unsqueeze(encoder_output4, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output4 = torch.cat((encoder_output4, returns[idx][10].unsqueeze(4)), dim=4)
        encoder_output4, _ = torch.max(encoder_output4, 4)
        encoder_output4 = self.fusion4(encoder_output4)

        bottleneck = self.bottleneck(encoder_output4)

        bottleneck = torch.unsqueeze(bottleneck, 4)
        for idx in range(len(self.FastSurferCNNs)):
            bottleneck = torch.cat((bottleneck, returns[idx][13].unsqueeze(4)), dim=4)
        bottleneck, _ = torch.max(bottleneck, 4)
        bottleneck = self.fusion5(bottleneck)

        decoder_output4 = self.decode4.forward(bottleneck, skip_encoder_4, indices_4)
        decoder_output3 = self.decode3.forward(decoder_output4, skip_encoder_3, indices_3)
        decoder_output2 = self.decode2.forward(decoder_output3, skip_encoder_2, indices_2)
        decoder_output1 = self.decode1.forward(decoder_output2, skip_encoder_1, indices_1)

        decoder_outputs = []
        decoder_outputs.append(decoder_output1)
        for ind_return in returns:
            decoder_outputs.append(ind_return[17])

        decoder_outputs_fused = self.fusion_layer(torch.cat(decoder_outputs, dim=1))
        logits = self.classifier.forward(decoder_outputs_fused)

        logits_list = []
        logits_list.append(logits)
        for ind_return in returns:
            logits += ind_return[0]
            logits_list.append(ind_return[0])

        return logits

class FastSurferCNN_Fuse_Unet_v3_extended(nn.Module):
    """
    Network Definition of Fully Competitive Network network
    * Spatial view aggregation (input 7 slices of which only middle one gets segmented)
    * Same Number of filters per layer (normally 64)
    * Dense Connections in blocks
    * Unpooling instead of transpose convolutions
    * Concatenationes are replaced with Maxout (competitive dense blocks)
    * Global skip connections are fused by Maxout (global competition)
    * Loss Function (weighted Cross-Entropy and dice loss)
    """
    def __init__(self, params):
        super(FastSurferCNN_Fuse_Unet_v3_extended, self).__init__()

        num_channels = params['num_channels']

        params['num_channels'] = params['num_modality'] * 7
        # Parameters for the Descending Arm
        self.encode1 = sm.CompetitiveEncoderBlockInput(params)
        params['num_channels'] = params['num_filters']
        self.encode2 = sm.CompetitiveEncoderBlock(params)
        self.encode3 = sm.CompetitiveEncoderBlock(params)
        self.encode4 = sm.CompetitiveEncoderBlock(params)
        self.bottleneck = sm.CompetitiveDenseBlock(params)

        # Parameters for the Ascending Arm
        params['num_channels'] = params['num_filters']
        self.decode4 = sm.CompetitiveDecoderBlock(params)
        self.decode3 = sm.CompetitiveDecoderBlock(params)
        self.decode2 = sm.CompetitiveDecoderBlock(params)
        self.decode1 = sm.CompetitiveDecoderBlock(params)

        self.fusion1 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion2 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion3 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion4 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion5 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])

        self.fusion_layer = nn.Conv2d(params['num_filters'] * (params['num_modality'] + 1), params['num_filters'], params['kernel_c'], params['stride_conv'])  # To generate logits

        params['num_channels'] = params['num_filters']
        self.classifier = sm.ClassifierBlock(params)

        # Code for Network Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Load backbones
        self.FastSurferCNNs = nn.ModuleList()
        for idx in range(params['num_modality']):
            params['num_channels'] = num_channels
            backbone_model = FastSurferCNN_return_all(params)
            if params['backbone_model'] is not None:
                # backbone_model = nn.DataParallel(backbone_model)
                print('Loading: %s' % params['backbone_model'][idx])
                if params["use_cuda"]:
                    model_state = torch.load(params['backbone_model'][idx])
                else:
                    model_state = torch.load(params['backbone_model'][idx], map_location=torch.device('cpu'))
                new_state_dict = OrderedDict()
                for k, v in model_state["model_state_dict"].items():
                    if k[:7] == "module.":
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v
                backbone_model.load_state_dict(new_state_dict)
                for param in backbone_model.parameters():
                    param.requires_grad = False
            else:
                print('No Backbone!!!')
            self.FastSurferCNNs.append(backbone_model)

        print('Initialize Fuse Unet v3 done!')

    def forward(self, x):
        """
        Computational graph
        :param tensor x: input image
        :return tensor: prediction logits
        """

        # return [0] logits, \
        #        [1] encoder_output1,  [2] skip_encoder_1,  [3] indices_1, \
        #        [4] encoder_output2,  [5] skip_encoder_2,  [6] indices_2, \
        #        [7] encoder_output3,  [8] skip_encoder_3,  [9] indices_3, \
        #        [10] encoder_output4, [11] skip_encoder_4, [12] indices_4, \
        #        [13] bottleneck, \
        #        [14] decoder_output4, [15] decoder_output3, [16] decoder_output2, [17] decoder_output1

        returns = []
        for idx in range(len(self.FastSurferCNNs)):
            fscnn = self.FastSurferCNNs[idx]
            returns.append(fscnn(x[:, ((idx)*7):((idx+1) * 7), :, :]))

        encoder_output1, skip_encoder_1, indices_1 = self.encode1.forward(x[:, :, :, :])

        encoder_output1 = torch.unsqueeze(encoder_output1, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output1 = torch.cat((encoder_output1, returns[idx][1].unsqueeze(4)), dim=4)
        encoder_output1, _ = torch.max(encoder_output1, 4)
        encoder_output1 = self.fusion1(encoder_output1)

        encoder_output2, skip_encoder_2, indices_2 = self.encode2.forward(encoder_output1)

        encoder_output2 = torch.unsqueeze(encoder_output2, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output2 = torch.cat((encoder_output2, returns[idx][4].unsqueeze(4)), dim=4)
        encoder_output2, _ = torch.max(encoder_output2, 4)
        encoder_output2 = self.fusion2(encoder_output2)

        encoder_output3, skip_encoder_3, indices_3 = self.encode3.forward(encoder_output2)

        encoder_output3 = torch.unsqueeze(encoder_output3, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output3 = torch.cat((encoder_output3, returns[idx][7].unsqueeze(4)), dim=4)
        encoder_output3, _ = torch.max(encoder_output3, 4)
        encoder_output3 = self.fusion3(encoder_output3)

        encoder_output4, skip_encoder_4, indices_4 = self.encode4.forward(encoder_output3)

        encoder_output4 = torch.unsqueeze(encoder_output4, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output4 = torch.cat((encoder_output4, returns[idx][10].unsqueeze(4)), dim=4)
        encoder_output4, _ = torch.max(encoder_output4, 4)
        encoder_output4 = self.fusion4(encoder_output4)

        bottleneck = self.bottleneck(encoder_output4)

        bottleneck = torch.unsqueeze(bottleneck, 4)
        for idx in range(len(self.FastSurferCNNs)):
            bottleneck = torch.cat((bottleneck, returns[idx][13].unsqueeze(4)), dim=4)
        bottleneck, _ = torch.max(bottleneck, 4)
        bottleneck = self.fusion5(bottleneck)

        decoder_output4 = self.decode4.forward(bottleneck, skip_encoder_4, indices_4)
        decoder_output3 = self.decode3.forward(decoder_output4, skip_encoder_3, indices_3)
        decoder_output2 = self.decode2.forward(decoder_output3, skip_encoder_2, indices_2)
        decoder_output1 = self.decode1.forward(decoder_output2, skip_encoder_1, indices_1)

        decoder_outputs = []
        decoder_outputs.append(decoder_output1)
        for ind_return in returns:
            decoder_outputs.append(ind_return[17])

        decoder_outputs_fused = self.fusion_layer(torch.cat(decoder_outputs, dim=1))
        logits = self.classifier.forward(decoder_outputs_fused)

        logits_list = []
        logits_list.append(logits)
        for ind_return in returns:
            logits += ind_return[0]
            logits_list.append(ind_return[0])

        return logits, logits_list

class FastSurferCNN_Fuse_Unet_v4(nn.Module):
    """
    Network Definition of Fully Competitive Network network
    * Spatial view aggregation (input 7 slices of which only middle one gets segmented)
    * Same Number of filters per layer (normally 64)
    * Dense Connections in blocks
    * Unpooling instead of transpose convolutions
    * Concatenationes are replaced with Maxout (competitive dense blocks)
    * Global skip connections are fused by Maxout (global competition)
    * Loss Function (weighted Cross-Entropy and dice loss)
    """
    def __init__(self, params):
        super(FastSurferCNN_Fuse_Unet_v4, self).__init__()

        num_channels = params['num_channels']

        params['num_channels'] = params['num_modality'] * 7
        # Parameters for the Descending Arm
        self.encode1 = sm.CompetitiveEncoderBlockInput(params)
        params['num_channels'] = params['num_filters']
        self.encode2 = sm.CompetitiveEncoderBlock(params)
        self.encode3 = sm.CompetitiveEncoderBlock(params)
        self.encode4 = sm.CompetitiveEncoderBlock(params)
        self.bottleneck = sm.CompetitiveDenseBlock(params)

        # Parameters for the Ascending Arm
        params['num_channels'] = params['num_filters']
        self.decode4 = sm.CompetitiveDecoderBlock(params)
        self.decode3 = sm.CompetitiveDecoderBlock(params)
        self.decode2 = sm.CompetitiveDecoderBlock(params)
        self.decode1 = sm.CompetitiveDecoderBlock(params)

        self.fusion1 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion2 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion3 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion4 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion5 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])

        self.fusion_layer = nn.Conv2d(params['num_filters'] * (params['num_modality'] + 1), params['num_filters'], params['kernel_c'], params['stride_conv'])  # To generate logits

        params['num_channels'] = params['num_filters']
        self.classifier = sm.ClassifierBlock(params)

        # Code for Network Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Load backbones
        self.FastSurferCNNs = nn.ModuleList()
        for idx in range(params['num_modality']):
            params['num_channels'] = num_channels
            backbone_model = FastSurferCNN_return_all(params)
            if params['backbone_model'] is not None:
                # backbone_model = nn.DataParallel(backbone_model)
                print('Loading: %s' % params['backbone_model'][idx])
                model_state = torch.load(params['backbone_model'][idx])
                new_state_dict = OrderedDict()
                for k, v in model_state["model_state_dict"].items():
                    if k[:7] == "module.":
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v
                backbone_model.load_state_dict(new_state_dict)
                for param in backbone_model.parameters():
                    param.requires_grad = False
            else:
                print('No Backbone!!!')
            self.FastSurferCNNs.append(backbone_model)

        print('Initialize Fuse Unet v4 done!')

    def forward(self, x):
        """
        Computational graph
        :param tensor x: input image
        :return tensor: prediction logits
        """

        # return [0] logits, \
        #        [1] encoder_output1,  [2] skip_encoder_1,  [3] indices_1, \
        #        [4] encoder_output2,  [5] skip_encoder_2,  [6] indices_2, \
        #        [7] encoder_output3,  [8] skip_encoder_3,  [9] indices_3, \
        #        [10] encoder_output4, [11] skip_encoder_4, [12] indices_4, \
        #        [13] bottleneck, \
        #        [14] decoder_output4, [15] decoder_output3, [16] decoder_output2, [17] decoder_output1

        returns = []
        for idx in range(len(self.FastSurferCNNs)):
            fscnn = self.FastSurferCNNs[idx]
            returns.append(fscnn(x[:, ((idx)*7):((idx+1) * 7), :, :]))

        encoder_output1, skip_encoder_1, indices_1 = self.encode1.forward(x[:, :, :, :])

        encoder_output1 = torch.unsqueeze(encoder_output1, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output1 = torch.cat((encoder_output1, returns[idx][1].unsqueeze(4)), dim=4)
        encoder_output1, _ = torch.max(encoder_output1, 4)
        encoder_output1 = self.fusion1(encoder_output1)

        encoder_output2, skip_encoder_2, indices_2 = self.encode2.forward(encoder_output1)

        encoder_output2 = torch.unsqueeze(encoder_output2, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output2 = torch.cat((encoder_output2, returns[idx][4].unsqueeze(4)), dim=4)
        encoder_output2, _ = torch.max(encoder_output2, 4)
        encoder_output2 = self.fusion2(encoder_output2)

        encoder_output3, skip_encoder_3, indices_3 = self.encode3.forward(encoder_output2)

        encoder_output3 = torch.unsqueeze(encoder_output3, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output3 = torch.cat((encoder_output3, returns[idx][7].unsqueeze(4)), dim=4)
        encoder_output3, _ = torch.max(encoder_output3, 4)
        encoder_output3 = self.fusion3(encoder_output3)

        encoder_output4, skip_encoder_4, indices_4 = self.encode4.forward(encoder_output3)

        encoder_output4 = torch.unsqueeze(encoder_output4, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output4 = torch.cat((encoder_output4, returns[idx][10].unsqueeze(4)), dim=4)
        encoder_output4, _ = torch.max(encoder_output4, 4)
        encoder_output4 = self.fusion4(encoder_output4)

        bottleneck = self.bottleneck(encoder_output4)

        bottleneck = torch.unsqueeze(bottleneck, 4)
        for idx in range(len(self.FastSurferCNNs)):
            bottleneck = torch.cat((bottleneck, returns[idx][13].unsqueeze(4)), dim=4)
        bottleneck, _ = torch.max(bottleneck, 4)
        bottleneck = self.fusion5(bottleneck)

        decoder_output4 = self.decode4.forward(bottleneck, skip_encoder_4, indices_4)
        decoder_output3 = self.decode3.forward(decoder_output4, skip_encoder_3, indices_3)
        decoder_output2 = self.decode2.forward(decoder_output3, skip_encoder_2, indices_2)
        decoder_output1 = self.decode1.forward(decoder_output2, skip_encoder_1, indices_1)

        decoder_outputs = []
        decoder_outputs.append(decoder_output1)
        for ind_return in returns:
            decoder_outputs.append(ind_return[17])

        decoder_outputs_fused = self.fusion_layer(torch.cat(decoder_outputs, dim=1))
        logits = self.classifier.forward(decoder_outputs_fused)

        # logits_list = []
        # logits_list.append(logits)
        # for ind_return in returns:
        #     logits += ind_return[0]
        #     logits_list.append(ind_return[0])

        return logits

class FastSurferCNN_Fuse_Unet_v4_extended(nn.Module):
    """
    Network Definition of Fully Competitive Network network
    * Spatial view aggregation (input 7 slices of which only middle one gets segmented)
    * Same Number of filters per layer (normally 64)
    * Dense Connections in blocks
    * Unpooling instead of transpose convolutions
    * Concatenationes are replaced with Maxout (competitive dense blocks)
    * Global skip connections are fused by Maxout (global competition)
    * Loss Function (weighted Cross-Entropy and dice loss)
    """
    def __init__(self, params):
        super(FastSurferCNN_Fuse_Unet_v4_extended, self).__init__()

        num_channels = params['num_channels']

        params['num_channels'] = params['num_modality'] * 7
        # Parameters for the Descending Arm
        self.encode1 = sm.CompetitiveEncoderBlockInput(params)
        params['num_channels'] = params['num_filters']
        self.encode2 = sm.CompetitiveEncoderBlock(params)
        self.encode3 = sm.CompetitiveEncoderBlock(params)
        self.encode4 = sm.CompetitiveEncoderBlock(params)
        self.bottleneck = sm.CompetitiveDenseBlock(params)

        # Parameters for the Ascending Arm
        params['num_channels'] = params['num_filters']
        self.decode4 = sm.CompetitiveDecoderBlock(params)
        self.decode3 = sm.CompetitiveDecoderBlock(params)
        self.decode2 = sm.CompetitiveDecoderBlock(params)
        self.decode1 = sm.CompetitiveDecoderBlock(params)

        self.fusion1 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion2 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion3 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion4 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])
        self.fusion5 = nn.Conv2d(params['num_filters'], params['num_filters'], params['kernel_c'], params['stride_conv'])

        self.fusion_layer = nn.Conv2d(params['num_filters'] * (params['num_modality'] + 1), params['num_filters'], params['kernel_c'], params['stride_conv'])  # To generate logits

        params['num_channels'] = params['num_filters']
        self.classifier = sm.ClassifierBlock(params)

        # Code for Network Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Load backbones
        self.FastSurferCNNs = nn.ModuleList()
        for idx in range(params['num_modality']):
            params['num_channels'] = num_channels
            backbone_model = FastSurferCNN_return_all(params)
            if params['backbone_model'] is not None:
                # backbone_model = nn.DataParallel(backbone_model)
                print('Loading: %s' % params['backbone_model'][idx])
                model_state = torch.load(params['backbone_model'][idx])
                new_state_dict = OrderedDict()
                for k, v in model_state["model_state_dict"].items():
                    if k[:7] == "module.":
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v
                backbone_model.load_state_dict(new_state_dict)
                for param in backbone_model.parameters():
                    param.requires_grad = False
            else:
                print('No Backbone!!!')
            self.FastSurferCNNs.append(backbone_model)

        print('Initialize Fuse Unet v4 done!')

    def forward(self, x):
        """
        Computational graph
        :param tensor x: input image
        :return tensor: prediction logits
        """

        # return [0] logits, \
        #        [1] encoder_output1,  [2] skip_encoder_1,  [3] indices_1, \
        #        [4] encoder_output2,  [5] skip_encoder_2,  [6] indices_2, \
        #        [7] encoder_output3,  [8] skip_encoder_3,  [9] indices_3, \
        #        [10] encoder_output4, [11] skip_encoder_4, [12] indices_4, \
        #        [13] bottleneck, \
        #        [14] decoder_output4, [15] decoder_output3, [16] decoder_output2, [17] decoder_output1

        returns = []
        for idx in range(len(self.FastSurferCNNs)):
            fscnn = self.FastSurferCNNs[idx]
            returns.append(fscnn(x[:, ((idx)*7):((idx+1) * 7), :, :]))

        encoder_output1, skip_encoder_1, indices_1 = self.encode1.forward(x[:, :, :, :])

        encoder_output1 = torch.unsqueeze(encoder_output1, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output1 = torch.cat((encoder_output1, returns[idx][1].unsqueeze(4)), dim=4)
        encoder_output1, _ = torch.max(encoder_output1, 4)
        encoder_output1 = self.fusion1(encoder_output1)

        encoder_output2, skip_encoder_2, indices_2 = self.encode2.forward(encoder_output1)

        encoder_output2 = torch.unsqueeze(encoder_output2, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output2 = torch.cat((encoder_output2, returns[idx][4].unsqueeze(4)), dim=4)
        encoder_output2, _ = torch.max(encoder_output2, 4)
        encoder_output2 = self.fusion2(encoder_output2)

        encoder_output3, skip_encoder_3, indices_3 = self.encode3.forward(encoder_output2)

        encoder_output3 = torch.unsqueeze(encoder_output3, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output3 = torch.cat((encoder_output3, returns[idx][7].unsqueeze(4)), dim=4)
        encoder_output3, _ = torch.max(encoder_output3, 4)
        encoder_output3 = self.fusion3(encoder_output3)

        encoder_output4, skip_encoder_4, indices_4 = self.encode4.forward(encoder_output3)

        encoder_output4 = torch.unsqueeze(encoder_output4, 4)
        for idx in range(len(self.FastSurferCNNs)):
            encoder_output4 = torch.cat((encoder_output4, returns[idx][10].unsqueeze(4)), dim=4)
        encoder_output4, _ = torch.max(encoder_output4, 4)
        encoder_output4 = self.fusion4(encoder_output4)

        bottleneck = self.bottleneck(encoder_output4)

        bottleneck = torch.unsqueeze(bottleneck, 4)
        for idx in range(len(self.FastSurferCNNs)):
            bottleneck = torch.cat((bottleneck, returns[idx][13].unsqueeze(4)), dim=4)
        bottleneck, _ = torch.max(bottleneck, 4)
        bottleneck = self.fusion5(bottleneck)

        decoder_output4 = self.decode4.forward(bottleneck, skip_encoder_4, indices_4)
        decoder_output3 = self.decode3.forward(decoder_output4, skip_encoder_3, indices_3)
        decoder_output2 = self.decode2.forward(decoder_output3, skip_encoder_2, indices_2)
        decoder_output1 = self.decode1.forward(decoder_output2, skip_encoder_1, indices_1)

        decoder_outputs = []
        decoder_outputs.append(decoder_output1)
        for ind_return in returns:
            decoder_outputs.append(ind_return[17])

        decoder_outputs_fused = self.fusion_layer(torch.cat(decoder_outputs, dim=1))
        logits = self.classifier.forward(decoder_outputs_fused)

        logits_list = []
        logits_list.append(logits)
        logits_sum = logits
        for ind_return in returns:
            logits_list.append(ind_return[0])
            logits_sum += ind_return[0]
        logits_list.append(logits_sum)

        return logits, logits_list