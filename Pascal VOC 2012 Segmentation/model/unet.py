import torch
import torchvision
from torch import nn
from torch.nn import functional

class Unet(nn.Module):
    
    def __init__(self,in_channels,channels=32,classes=23):
        super(Unet, self).__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.classes = classes
        self.cblock1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels,
                      out_channels=self.channels,
                      kernel_size=3,
                      padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.channels,
                      out_channels=self.channels,
                      kernel_size=3,
                      padding='same'),
            nn.ReLU()
        )
        self.cblock2 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels,
                      out_channels=self.channels*2,
                      kernel_size=3,
                      padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.channels*2,
                      out_channels=self.channels*2,
                      kernel_size=3,
                      padding='same'),
            nn.ReLU()
        )
        self.cblock3 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels*2,
                      out_channels=self.channels*2*2,
                      kernel_size=3,
                      padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.channels*2*2,
                      out_channels=self.channels*2*2,
                      kernel_size=3,
                      padding='same'),
            nn.ReLU()
        )
        self.cblock4 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels*2*2,
                      out_channels=self.channels*2*2*2,
                      kernel_size=3,
                      padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.channels*2*2*2,
                      out_channels=self.channels*2*2*2,
                      kernel_size=3,
                      padding='same'),
            nn.ReLU()
        )
        self.cblock5 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels*2*2*2,
                      out_channels=self.channels*2*2*2*2,
                      kernel_size=3,
                      padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.channels*2*2*2*2,
                      out_channels=self.channels*2*2*2*2,
                      kernel_size=3,
                      padding='same'),
            nn.ReLU()
        )
        self.up1 = nn.ConvTranspose2d(in_channels=self.channels*2*2*2*2,
                                      out_channels=self.channels*2*2*2,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      output_padding=1)
        self.ublock6 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels*2*2*2*2,
                      out_channels=self.channels*2*2*2,
                      kernel_size=3,
                      padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.channels*2*2*2,
                      out_channels=self.channels*2*2*2,
                      kernel_size=3,
                      padding='same'),
            nn.ReLU()
        )
        self.up2 = nn.ConvTranspose2d(in_channels=self.channels*2*2*2,
                                      out_channels=self.channels*2*2,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      output_padding=1)
        self.ublock7 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels*2*2*2,
                      out_channels=self.channels*2*2,
                      kernel_size=3,
                      padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.channels*2*2,
                      out_channels=self.channels*2*2,
                      kernel_size=3,
                      padding='same'),
            nn.ReLU()
        )
        self.up3 = nn.ConvTranspose2d(in_channels=self.channels*2*2,
                                      out_channels=self.channels*2,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      output_padding=1)
        self.ublock8 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels*2*2,
                      out_channels=self.channels*2,
                      kernel_size=3,
                      padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.channels*2,
                      out_channels=self.channels*2,
                      kernel_size=3,
                      padding='same'),
            nn.ReLU()
        )
        self.up4 = nn.ConvTranspose2d(in_channels=self.channels*2,
                                      out_channels=self.channels,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      output_padding=1)
        self.ublock9 = nn.Sequential(
            nn.Conv2d(in_channels=self.channels*2,
                      out_channels=self.channels,
                      kernel_size=3,
                      padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.channels,
                      out_channels=self.channels,
                      kernel_size=3,
                      padding='same'),
            nn.ReLU()
        )
        self.conv1 = nn.Conv2d(in_channels=self.channels,
                               out_channels=self.channels,
                               kernel_size=3,
                               padding='same')
        self.conv2 = nn.Conv2d(in_channels=self.channels,
                               out_channels=self.classes,
                               kernel_size=1,
                               padding='same')
    def forward(self,inputs):
        # downsampling encoder
        cblock1 = self.cblock1(inputs)
        skip_connection1 = cblock1
        cblock1 = functional.max_pool2d(cblock1,kernel_size=2)
        
        cblock2 = self.cblock2(cblock1)
        skip_connection2 = cblock2
        cblock2 = functional.max_pool2d(cblock2,kernel_size=2)
        
        cblock3 = self.cblock3(cblock2)
        skip_connection3 = cblock3
        cblock3 = functional.max_pool2d(cblock3,kernel_size=2)
        
        cblock4 = self.cblock4(cblock3)
        cblock4 = functional.dropout(cblock4,0.3)
        skip_connection4 = cblock4
        cblock4 = functional.max_pool2d(cblock4,kernel_size=2)
        
        cblock5 = self.cblock5(cblock4)
        cblock5 = functional.dropout(cblock5,0.3)
        skip_connection5 = cblock5
        
        #upsampling decoder
        ublock6 = self.up1(cblock5)
        merge = torch.cat((ublock6,skip_connection4),dim=1)
        ublock6 = self.ublock6(merge)
        
        ublock7 = self.up2(ublock6)
        merge = torch.cat((ublock7,skip_connection3),dim=1)
        ublock7 = self.ublock7(merge)
        
        ublock8 = self.up3(ublock7)
        merge = torch.cat((ublock8,skip_connection2),dim=1)
        ublock8 = self.ublock8(merge)
        
        ublock9 = self.up4(ublock8)
        merge = torch.cat((ublock9,skip_connection1),dim=1)
        ublock9 = self.ublock9(merge)
        
        conv = self.conv1(ublock9)
        conv = functional.relu(conv)
        conv = self.conv2(conv)
        return conv