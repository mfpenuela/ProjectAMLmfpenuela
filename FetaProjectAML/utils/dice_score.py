import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


def distributionAge(edad):
    if edad <= 24:
        part = torch.tensor([0.956230808,0.012620446,0.00723119,0.015634293,0.004152267,0.000929381,0.002283201,0.000918414])
        #part = torch.tensor([0.956230808,1-0.956230808])
    else:
        part = torch.tensor([0.889709304,0.038334469,0.017307918,0.038298294,0.007138341,0.002724011,0.004787459,0.001700204])
        #part = torch.tensor([0.889709304,1-0.889709304])
    return part    
   


def distribution(dis):

    a=np.zeros(8)
    j=0
    for i in range(8):
        
        if i in dis[0]:
  
            a[i]=dis[1][j]
            j=j+1
        else:
            a[i]=0.00001
            
    return torch.tensor(a)

def distribution2(dis,t):

    a=np.zeros(2)
    if 0 in dis[0]:
        if dis[1][0]==t:
            a[1]=0.0001
            a[0]=0.9999
        else:
            a[0]=dis[1][0]/t
            a[1]= 1-(dis[1][0])/t
    else:
        a[0]=0.0001
        a[1]=0.9999
            
    return torch.tensor(a)
        
class SizeParches(nn.Module):
    def __init__(self, gamma=2):
        super(SizeParches, self).__init__()

    def forward(self, inputs, edad, targets, x, tri2, tri3):
        #print(tri3.shape)
        
        prediction = F.softmax(inputs, dim=1)
        prediction = torch.argmax(prediction, dim=1)
        
        b,s,y=prediction.size()

        loss=torch.zeros(b)

        for i in range(b):
            pre = prediction[i] > 0
            suma = torch.sum(pre)/(s*y)
            if edad[i].item()<=24:
                disT1=tri2
            else:
                disT1=tri3
        
            disT1=disT1[:, :, :, int(x[i].item())]
            disT1=torch.tensor(disT1).cuda()

            b,s,y=disT1.shape

            disT1=disT1>0

            disTarget1=torch.sum(disT1, dim=1)
            disTarget1=torch.sum(disTarget1, dim=1)

            disTarget1=disTarget1/(s*y)

            max1=torch.max(disTarget1)
            min1=torch.min(disTarget1)

            pen1=0.

            if suma > max1:
                pen1=(suma-max1)**2
            elif suma < min1:
                pen1=(suma-min1)**2
            loss[i]=pen1
        loss=torch.mean(loss)
        #print(loss)

        return loss*25

       
class SizeParchesL1(nn.Module):
    def __init__(self, gamma=2):
        super(SizeParchesL1, self).__init__()

    def forward(self, inputs, edad, targets, x, tri2, tri3):
        #print(tri3.shape)
        
        prediction = F.softmax(inputs, dim=1)
        prediction = torch.argmax(prediction, dim=1)
        
        b,s,y=prediction.size()

        loss=torch.zeros(b)

        for i in range(b):
            pre = prediction[i] == 1
            suma = torch.sum(pre)/(s*y)
            if edad[i].item()<=24:
                disT1=tri2
            else:
                disT1=tri3
        
            disT1=disT1[:, :, :, int(x[i].item())]
            disT1=torch.tensor(disT1).cuda()

            b,s,y=disT1.shape

            disT1=disT1==1

            disTarget1=torch.sum(disT1, dim=1)
            disTarget1=torch.sum(disTarget1, dim=1)

            disTarget1=disTarget1/(s*y)

            max1=torch.max(disTarget1)
            min1=torch.min(disTarget1)

            pen1=0.

            if suma > max1:
                pen1=(suma-max1)**2
            elif suma < min1:
                pen1=(suma-min1)**2
            loss[i]=pen1
        loss=torch.mean(loss)
        #print(loss)

        return loss*25
        


class KLParches(nn.Module):
    def __init__(self, gamma=2):
        super(KLParches, self).__init__()
        self.kl= nn.KLDivLoss(log_target=True)

    def forward(self, inputs, edad, targets, x, tri2,tri3):
        prediction = F.softmax(inputs, dim=1)
        prediction = torch.argmax(prediction, dim=1)
        b,s,y=prediction.size()
        loss=torch.zeros(b)
        for i in range(b):
            dis1=torch.unique(prediction[i],return_counts=True)
            dis1=torch.log(distribution(dis1)/(s*y))

            if edad[i].item()<=24:
                disT1=tri2
            else:
                disT1=tri3

            disT1=disT1[:, :, :, int(x[i].item())]
            disT1=torch.tensor(disT1).cuda()

            b,s,y=disT1.shape

            #print(b,s,y,z)

            disTarget1=torch.unique(disT1,return_counts=True)
        
            disTarget1=torch.log(distribution(disTarget1)/(s*y*disT1.size(0)))
        
            loss1=self.kl(dis1.cuda(),disTarget1.cuda())

            loss[i]=loss1
        #print(dis1,disTarget1)
        loss=torch.mean(loss)
        #print(loss)
        return loss*2.5