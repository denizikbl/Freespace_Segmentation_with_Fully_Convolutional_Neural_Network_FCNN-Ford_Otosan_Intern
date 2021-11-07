import torch 
import torch.nn as nn

def double_conv(in_channels, out_channels, mid_channels = None):
    if not mid_channels:
        mid_channels = out_channels
        return nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3,padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),#activation function
            nn.Conv2d(mid_channels, out_channels, kernel_size=3,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
)
class FoInternNet(nn.Module):
    def __init__(self,input_size,n_classes):
        super(FoInternNet, self).__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        
        self.down1 = double_conv(3, 64)
        self.down2 = double_conv(64, 128)
        self.down3 = double_conv(128, 256)
        self.down4 = double_conv(256, 512)
        self.down5 = double_conv(512, 1024)
        
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear', align_corners=True)
        
        self.up4 = double_conv(512 + 1024, 512)
        self.up3 = double_conv(256 + 512, 256)
        self.up2 = double_conv(128 + 256, 128)
        self.up1 = double_conv(128 + 64, 64)
        self.conv_last = nn.Conv2d(64, n_classes, 1)
         
             
    def forward(self, x):
        #print(x.shape)
        conv1 = self.down1(x)
        x = self.maxpool(conv1)
        
        conv2 = self.down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.down3(x)
        x = self.maxpool(conv3) 
        
        conv4 = self.down4(x)
        x = self.maxpool(conv4)
        
        x = self.down5(x)
        
        x = self.upsample(x)    
        x = torch.cat([x, conv4], dim=1)#Combines the given tensor array at the given size 
        #dim: the size in which the tensors are combined
     
        x = self.up4(x)
        x = self.upsample(x)  
        x = torch.cat([x, conv3], dim=1)     
     
        x = self.up3(x)
        x = self.upsample(x)  
        x = torch.cat([x, conv2], dim=1)  

        x = self.up2(x)
        x = self.upsample(x)    
        x = torch.cat([x, conv1], dim=1)   
        #print("cat")
        #print(x.shape)

        x = self.up1(x)
        #print(x.shape)
        
        x = self.conv_last(x)
        #print(x.shape)

        
        x = nn.Softmax(dim=1)(x)
        #print(x.shape)

        return x
    
if __name__ == '__main__':
    model = FoInternNet(input_size=(224, 224), n_classes=2)
    
    
    