import torch
import torch.nn as nn
import numpy as np
import nibabel as nib 

def PositionalEncoding(size, value):
    d=1
    for i in range(len(size)):
        d=d*size[i]
    j=int(d/2)
    PE=np.zeros(d)
    for i in range(j):
        PE[2*i] = np.sin(value/(10000**(2*i/d)))
        PE[2*i + 1] = np.cos(value / (10000**(2*i/d)))
    if d%2 !=0:

        PE[2*j] = np.sin(value / (10000 ** (2 * j / d)))

    PE=np.reshape(PE,size)

    return PE

def Rango(edad):
    edadF=0
    rangos=np.arange(0,48,12)
    for i  in range(len(rangos)-1):
        if edad> rangos[i] and edad<=rangos[i+1]:
            edad=edadF
            break
        else:
            edadF=edadF+12
    return edad

def Rango1(edad):
    edadF=int(edad/4)
    return edadF

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        # number of input channels is a number of filters in the previous layer
        # number of output channels is a number of filters in the current layer
        # "same" convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class AttentionBlock(nn.Module):
    """Attention block with learnable parameters"""

    def __init__(self, F_g, F_l, n_coefficients):
        """
        :param F_g: number of feature maps (channels) in previous layer
        :param F_l: number of feature maps in corresponding encoder layer, transferred via skip connection
        :param n_coefficients: number of learnable multi-dimensional attention coefficients
        """
        super(AttentionBlock, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        """
        :param gate: gating signal from previous layer
        :param skip_connection: activation from corresponding encoder layer
        :return: output activations
        """
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = skip_connection * psi
        return out


class AttentionUNet(nn.Module):

    def __init__(self, img_ch=1, output_ch=8):
        super(AttentionUNet, self).__init__()

        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.n_channels = 1
        self.n_classes = 8

        self.Conv1 = ConvBlock(img_ch, 64)
        self.Conv2 = ConvBlock(64, 128)
        self.Conv3 = ConvBlock(128, 256)
        self.Conv4 = ConvBlock(256, 512)
        self.Conv5 = ConvBlock(512, 1024)

        self.Up5 = UpConv(1024, 512)
        self.Att5 = AttentionBlock(F_g=512, F_l=512, n_coefficients=256)
        self.UpConv5 = ConvBlock(1024, 512)

        self.Up4 = UpConv(512, 256)
        self.Att4 = AttentionBlock(F_g=256, F_l=256, n_coefficients=128)
        self.UpConv4 = ConvBlock(512, 256)

        self.Up3 = UpConv(256, 128)
        self.Att3 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64)
        self.UpConv3 = ConvBlock(256, 128)

        self.Up2 = UpConv(128, 64)
        self.Att2 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32)
        self.UpConv2 = ConvBlock(128, 64)

        self.Conv = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x, edad):
        """
        e : encoder layers
        d : decoder layers
        s : skip-connections from encoder layers to decoder layers
        """
        e1 = self.Conv1(x)

        e2 = self.MaxPool(e1)
        e2 = self.Conv2(e2)

        e3 = self.MaxPool(e2)
        e3 = self.Conv3(e3)

        e4 = self.MaxPool(e3)
        e4 = self.Conv4(e4)

        e5 = self.MaxPool(e4)
        e5 = self.Conv5(e5)

        dim=(1, e5.size(1),e5.size(2),e5.size(3))
        y1=PositionalEncoding(dim,Rango(22))
        y2=PositionalEncoding(dim,Rango(30))
        #print(edad)
        y1=torch.tensor(y1).cuda()
        y2=torch.tensor(y2).cuda()

        if edad[0].item()<24:
            en1=y1
        else:
            en1=y2
        # if edad[1].item()<24:
        #     en2=y1
        # else:
        #     en2=y2

        # encoding=torch.cat((en1,en2),0)

        # for i in range(e5.size(0)-2):
        #     if edad[i+2].item()<24:
        #         e=y1
        #     else:
        #         e=y2
        #     encoding=torch.cat((encoding,e),0)
        e5=torch.add(e5,en1).type(torch.cuda.FloatTensor)

        d5 = self.Up5(e5)

        s4 = self.Att5(gate=d5, skip_connection=e4)
        d5 = torch.cat((s4, d5), dim=1) # concatenate attention-weighted skip connection with previous layer output
        d5 = self.UpConv5(d5)

        d4 = self.Up4(d5)
        s3 = self.Att4(gate=d4, skip_connection=e3)
        d4 = torch.cat((s3, d4), dim=1)
        d4 = self.UpConv4(d4)

        d3 = self.Up3(d4)

        s2 = self.Att3(gate=d3, skip_connection=e2)
        d3 = torch.cat((s2, d3), dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
        s1 = self.Att2(gate=d2, skip_connection=e1)
        d2 = torch.cat((s1, d2), dim=1)
        d2 = self.UpConv2(d2)

        out = self.Conv(d2)

        return out


class AttentionUNetBase(nn.Module):

    def __init__(self, img_ch=1, output_ch=8):
        super(AttentionUNetBase, self).__init__()

        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.n_channels = 1
        self.n_classes = 8

        self.Conv1 = ConvBlock(img_ch, 64)
        self.Conv2 = ConvBlock(64, 128)
        self.Conv3 = ConvBlock(128, 256)
        self.Conv4 = ConvBlock(256, 512)
        self.Conv5 = ConvBlock(512, 1024)

        self.Up5 = UpConv(1024, 512)
        self.Att5 = AttentionBlock(F_g=512, F_l=512, n_coefficients=256)
        self.UpConv5 = ConvBlock(1024, 512)

        self.Up4 = UpConv(512, 256)
        self.Att4 = AttentionBlock(F_g=256, F_l=256, n_coefficients=128)
        self.UpConv4 = ConvBlock(512, 256)

        self.Up3 = UpConv(256, 128)
        self.Att3 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64)
        self.UpConv3 = ConvBlock(256, 128)

        self.Up2 = UpConv(128, 64)
        self.Att2 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32)
        self.UpConv2 = ConvBlock(128, 64)

        self.Conv = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x, edad):
        """
        e : encoder layers
        d : decoder layers
        s : skip-connections from encoder layers to decoder layers
        """
        e1 = self.Conv1(x)

        e2 = self.MaxPool(e1)
        e2 = self.Conv2(e2)

        e3 = self.MaxPool(e2)
        e3 = self.Conv3(e3)

        e4 = self.MaxPool(e3)
        e4 = self.Conv4(e4)

        e5 = self.MaxPool(e4)
        e5 = self.Conv5(e5)

        # dim=(1, e5.size(1),e5.size(2),e5.size(3))
        # y1=PositionalEncoding(dim,Rango(22))
        # y2=PositionalEncoding(dim,Rango(30))
        # #print(edad)
        # y1=torch.tensor(y1).cuda()
        # y2=torch.tensor(y2).cuda()

        # if edad[0].item()<24:
        #     en1=y1
        # else:
        #     en1=y2
        # if edad[1].item()<24:
        #     en2=y1
        # else:
        #     en2=y2

        # encoding=torch.cat((en1,en2),0)

        # for i in range(e5.size(0)-2):
        #     if edad[i+2].item()<24:
        #         e=y1
        #     else:
        #         e=y2
        #     encoding=torch.cat((encoding,e),0)
        #e5=torch.add(e5,en1).type(torch.cuda.FloatTensor)

        d5 = self.Up5(e5)

        s4 = self.Att5(gate=d5, skip_connection=e4)
        d5 = torch.cat((s4, d5), dim=1) # concatenate attention-weighted skip connection with previous layer output
        d5 = self.UpConv5(d5)

        d4 = self.Up4(d5)
        s3 = self.Att4(gate=d4, skip_connection=e3)
        d4 = torch.cat((s3, d4), dim=1)
        d4 = self.UpConv4(d4)

        d3 = self.Up3(d4)

        s2 = self.Att3(gate=d3, skip_connection=e2)
        d3 = torch.cat((s2, d3), dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
        s1 = self.Att2(gate=d2, skip_connection=e1)
        d2 = torch.cat((s1, d2), dim=1)
        d2 = self.UpConv2(d2)

        out = self.Conv(d2)

        return out

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        #print(edad)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        dim=(1, x5.size(1),x5.size(2),x5.size(3))
        y1=PositionalEncoding(dim,Rango(22))
        y2=PositionalEncoding(dim,Rango(30))
        #print(edad)
        y1=torch.tensor(y1).cuda()
        y2=torch.tensor(y2).cuda()

        if edad[0].item()<24:
            e1=y1
        else:
            e1=y2
        #if edad[1].item()<24:
        #    e2=y1
        #else:
        #    e2=y2

        #encoding=torch.cat((e1,e2),0)

        #for i in range(x5.size(0)-2):
        #    if edad[i+2].item()<24:
        #        e=y1
        #    else:
        #        e=y2
        #    encoding=torch.cat((encoding,e),0)
        
        x5=torch.add(x5,e1).type(torch.cuda.FloatTensor)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class UNetBase(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        #print(edad)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        dim=(1, x5.size(1),x5.size(2),x5.size(3))
        y1=PositionalEncoding(dim,Rango(22))
        y2=PositionalEncoding(dim,Rango(30))
        #print(edad)
        y1=torch.tensor(y1).cuda()
        y2=torch.tensor(y2).cuda()

        if edad[0].item()<24:
            e1=y1
        else:
            e1=y2
        #if edad[1].item()<24:
        #    e2=y1
        #else:
        #    e2=y2

        #encoding=torch.cat((e1,e2),0)

        #for i in range(x5.size(0)-2):
        #    if edad[i+2].item()<24:
        #        e=y1
        #    else:
        #        e=y2
        #    encoding=torch.cat((encoding,e),0)
        
        x5=torch.add(x5,e1).type(torch.cuda.FloatTensor)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

