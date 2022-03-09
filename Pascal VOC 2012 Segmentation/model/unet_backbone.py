import torch
import torchvision
from torch import nn
from torch.nn import functional

def get_backbone(name='resnet18',pretrained=True):
    
    ''' 
        define backbone and skip connection location 
        return:
                backbone -- a pretrained model as a backbone
                features_name -- name of layer where skip connection is 
                backbone_output -- my desired layer output as orgin of upsample block  
    '''
    
    if name == 'resnet18':
        backbone = torchvision.models.resnet18(pretrained=pretrained)
    elif name == 'resnet34':
        backbone = torchvision.models.resnet34(pretrained=pretrained)
    elif name == 'resnet50':
        backbone = torchvision.models.resnet50(pretrained=pretrained)
    elif name == 'resnet101':
        backbone = torchvision.models.resnet101(pretrained=pretrained)
    elif name == 'resnet152':
        backbone = torchvision.models.resnet152(pretrained=pretrained)
    else:
        raise NotImplemented('{} backbone model is not implemented so far.'.format(name))
        
    # specifying skip feature and output names
    features_name = [None,'relu','layer1','layer2','layer3']
    backbone_output = 'layer4'
    return backbone, features_name, backbone_output

class UpsampleBlock(nn.Module):
    
    ''' upsample block '''
    
    def __init__(self,ch_in, ch_out=None, skip_in=0, use_bn=True):
        ''' 
            ch_in -- in_channels
            ch_out -- out_channels
            skip_in -- skip connection tensor channels
            use_bn -- use batchnorm
        '''
        super(UpsampleBlock,self).__init__()
        
        ch_out = ch_in/2 if ch_out is None else ch_out

        # first convolution: either transposed conv, or conv following the skip connection
        
        # versions: kernel=4 padding=1, kernel=2 padding=0
        self.up = nn.ConvTranspose2d(in_channels=ch_in, out_channels=ch_out, kernel_size=(4, 4),
                                     stride=2, padding=1, output_padding=0, bias=(not use_bn))
        self.bn1 = nn.BatchNorm2d(ch_out) if use_bn else None

        self.relu = nn.ReLU(inplace=True)

        # second convolution
        conv2_in = ch_out + skip_in
        self.conv2 = nn.Conv2d(in_channels=conv2_in, out_channels=ch_out, kernel_size=(3, 3),
                               stride=1, padding=1, bias=(not use_bn))
        self.bn2 = nn.BatchNorm2d(ch_out) if use_bn else None
        
    def forward(self, x, skip_connection=None):

        x = self.up(x) 
        x = self.bn1(x) if self.bn1 is not None else x
        x = self.relu(x)

        if skip_connection is not None:
            x = torch.cat([x, skip_connection], dim=1)

        x = self.conv2(x)
        x = self.bn2(x) if self.bn2 is not None else x
        x = self.relu(x)

        return x
        
class Unet(nn.Module):

    """ U-Net (https://arxiv.org/pdf/1505.04597.pdf) implementation with pre-trained torchvision backbones."""

    def __init__(self,
                 backbone_name='resnet18',
                 pretrained=True,
                 classes=21,
                 decoder_filters=(256, 128, 64, 32, 16),
                 decoder_use_batchnorm=True):
        super(Unet, self).__init__()

        self.backbone_name = backbone_name

        self.backbone, self.shortcut_features, self.bb_out_name = get_backbone(backbone_name, pretrained=pretrained)
        # channels number in skip conncetion and backbone output
        shortcut_chs, bb_out_chs = self.infer_skip_channels()
        
        # build decoder part
        self.upsample_blocks = nn.ModuleList()
        # decoder out_channels
        decoder_filters = decoder_filters[:len(self.shortcut_features)]  # avoiding having more blocks than skip connections
        # decoder in_channels
        decoder_filters_in = [bb_out_chs] + list(decoder_filters[:-1])
        num_blocks = len(self.shortcut_features)

        for i, [filters_in, filters_out] in enumerate(zip(decoder_filters_in, decoder_filters)):
            print('upsample_blocks[{}] in: {}   out: {}'.format(i, filters_in, filters_out))
            self.upsample_blocks.append(UpsampleBlock(filters_in, filters_out,
                                                      skip_in=shortcut_chs[num_blocks-i-1],
                                                      use_bn=decoder_use_batchnorm))

        self.final_conv = nn.Conv2d(decoder_filters[-1], classes, kernel_size=(1, 1))

    def forward(self, inputs):

        """ Forward propagation in U-Net. """

        x, features = self.forward_backbone(inputs)

        for skip_name, upsample_block in zip(self.shortcut_features[::-1], self.upsample_blocks):
            skip_features = features[skip_name]
            x = upsample_block(x, skip_features)

        x = self.final_conv(x)
        return x

    def forward_backbone(self, x):

        """ 
            Forward propagation in backbone encoder network. 
            return:
                    x -- backbone output
                    features -- all skip connection tensors are stored in dict
        """

        features = {None: None} if None in self.shortcut_features else dict()
        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.shortcut_features:
                features[name] = x
            if name == self.bb_out_name:
                break
    
        return x, features

    def infer_skip_channels(self):

        """ Getting the number of channels at skip connections and at the output of the encoder. """

        x = torch.zeros(1, 3, 320, 480)
        channels = [0] 

        # forward run in backbone to count channels (dirty solution but works for *any* Module)
        for name, child in self.backbone.named_children():
            x = child(x)
            if name in self.shortcut_features:
                channels.append(x.shape[1])
            if name == self.bb_out_name:
                out_channels = x.shape[1]
                break
        return channels, out_channels

