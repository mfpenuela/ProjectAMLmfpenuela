import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from unet import UNet
from unet import AttentionUNet
from unet import UNetBase
from unet import AttentionUNetBase

from PIL import Image
import skimage.io as io
import nibabel as nib

def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    smooth = 0.0001
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
def dice_coef_multilabel(y_true, y_pred, numLabels):
    dice=0
    l=[]
    for index in range(numLabels):
        t=y_true==index
        p=y_pred==index
        dice += dice_coef(t, p)
        l.append(dice_coef(t, p))
    return dice/numLabels, l
def diceDemo(img):
    names=[img]
    dice=0
    base='/home/mfpenuela/FetaProjectAML/Results/DemoImagesBase/'
    Attention='/home/mfpenuela/FetaProjectAML/Results/DemoImagesAttention/'

    for i in range(len(names)):
        vol=np.zeros((128,128,128))
        volgt = np.zeros((128, 128, 128))
        c = 0
        for j in range(128):
            a=io.imread(base+names[i]+'-'+str(c)+'.png')
            vol[:,:,j]=a

            seg = Image.open('/home/mfpenuela/FetaProjectAML/Test/masksImages/' + names[i] + '-' + str(c) + '_mask.png')
            seg = seg.resize((128, 128))
            seg = np.asarray(seg)
            volgt[:, :, j] = seg
            c = c + 2

        d, l = dice_coef_multilabel(vol, volgt, 8)

        dice = dice + d

 
        print('Attention Base DSC:'+str(dice))
    dice=0
    
    for i in range(len(names)):
        vol=np.zeros((128,128,128))
        volgt = np.zeros((128, 128, 128))
        c = 0
        for j in range(128):
            a=io.imread(Attention+names[i]+'-'+str(c)+'.png')
            vol[:,:,j]=a

            seg = Image.open('/home/mfpenuela/FetaProjectAML/Test/masksImages/' + names[i] + '-' + str(c) + '_mask.png')
            seg = seg.resize((128, 128))
            seg = np.asarray(seg)
            volgt[:, :, j] = seg
            c = c + 2

        d, l = dice_coef_multilabel(vol, volgt, 8)

        dice = dice + d

        dicef=dice/10
        print('Attention Fest DSC:'+str(dice))
def diceTest():
    names=['sub-016','sub-023', 'sub-024','sub-027','sub-028','sub-032','sub-035','sub-036','sub-059','sub-064']
    dice=0
    base='/home/mfpenuela/FetaProjectAML/Results/testImagesBase/'
    Attention='/home/mfpenuela/FetaProjectAML/Results/testImagesAttention/'

    for i in range(len(names)):
        vol=np.zeros((128,128,128))
        volgt = np.zeros((128, 128, 128))
        c = 0
        for j in range(128):
            a=io.imread(base+names[i]+'-'+str(c)+'.png')
            vol[:,:,j]=a

            seg = Image.open('/home/mfpenuela/FetaProjectAML/Test/masksImages/' + names[i] + '-' + str(c) + '_mask.png')
            seg = seg.resize((128, 128))
            seg = np.asarray(seg)
            volgt[:, :, j] = seg
            c = c + 2

        d, l = dice_coef_multilabel(vol, volgt, 8)

        dice = dice + d
    dicef=dice/10


    print('Attention Base DSC:'+str(dicef))
    dice=0
    
    for i in range(len(names)):
        vol=np.zeros((128,128,128))
        volgt = np.zeros((128, 128, 128))
        c = 0
        for j in range(128):
            a=io.imread(Attention+names[i]+'-'+str(c)+'.png')
            vol[:,:,j]=a

            seg = Image.open('/home/mfpenuela/FetaProjectAML/Test/masksImages/' + names[i] + '-' + str(c) + '_mask.png')
            seg = seg.resize((128, 128))
            seg = np.asarray(seg)
            volgt[:, :, j] = seg
            c = c + 2

        d, l = dice_coef_multilabel(vol, volgt, 8)

        dice = dice + d

    dicef=dice/10
    print('Attention Fest DSC:'+str(dicef))

def guardarTest():
    base='/home/mfpenuela/FetaProjectAML/Results/testVolsBase/'
    Attention='/home/mfpenuela/FetaProjectAML/Results/testVolsAttention/'
    baseR='/home/mfpenuela/FetaProjectAML/Results/testImagesBase/'
    AttentionR='/home/mfpenuela/FetaProjectAML/Results/testImagesAttention/'

    
    names=['sub-016', 'sub-023', 'sub-024','sub-027','sub-028','sub-032','sub-035','sub-036','sub-059','sub-064']
    #names=['sub-001','sub-005','sub-020','sub-029','sub-037','sub-038','sub-052','sub-070','sub-075','sub-077']

    for i in range(len(names)):
        vol=np.zeros((128,128,128))
        c = 0
        for j in range(128):
            a=io.imread(baseR+names[i]+'-'+str(c)+'.png')
            vol[:,:,j]=a
            c=c+2

        vol = np.array(vol, dtype=np.float32)
        vol2 = nib.Nifti1Image(vol, np.eye(4))
        nib.save(vol2, base+names[i]+'.nii.gz')

    for i in range(len(names)):
        vol=np.zeros((128,128,128))
        c = 0
        for j in range(128):
            a=io.imread(AttentionR+names[i]+'-'+str(c)+'.png')
            vol[:,:,j]=a
            c=c+2

        vol = np.array(vol, dtype=np.float32)
        vol2 = nib.Nifti1Image(vol, np.eye(4))
        nib.save(vol2, Attention+names[i]+'.nii.gz')



def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5,
                edad=20):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    edad=torch.tensor([edad])
    img = img.to(device=device, dtype=torch.float32)
    edad = edad.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img, edad)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        #full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return probs.argmax(dim=0).cpu().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images')
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--mode', default='test')
    parser.add_argument('--img', default='sub-016')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        seg1 = np.array(mask, dtype=np.uint8)
        S1 = Image.fromarray(seg1, 'L')
        return S1
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


if __name__ == '__main__':
    modelAttentioFest='/home/mfpenuela/FetaProjectAML/checkpoint/checkpoint_Attention-UNet.pth'
    modelAttentionBase='/home/mfpenuela/Attention-UNet/checkpoints/sizekl25/checkpoint_epoch200.pth'
    args = get_args()
    in_files = args.input
    #out_files = get_output_filenames(args)

    if args.mode=='test':
        net = AttentionUNetBase(img_ch=1, output_ch=8)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #logging.info(f'Loading model {args.model}')
        logging.info(f'Using device {device}')

        net.to(device=device)
        net.load_state_dict(torch.load(modelAttentionBase, map_location=device))

        logging.info('Model loaded!')
        names=['sub-016-','sub-023-','sub-024-','sub-027-','sub-028-','sub-032-','sub-035-','sub-036-','sub-059-','sub-064-']
        edad=[23.3,23.7,30.4,26.5,31.1,32.3,38.3,22.7,34.8,27.8]
    #names=['sub-001-','sub-005-','sub-020-','sub-029-','sub-037-','sub-038-','sub-052-','sub-070-','sub-075-','sub-077-']
    #edad=[27.9,22.6,25.8,32.5,23.4,26.9,21.2,20.1,29.0,26.9]

    
        print('Base test...')
        for j in range (len(names)):

            for i in range(256):
            #logging.info(f'\nPredicting image {filename} ...')
                filename='/home/mfpenuela/FetaProjectAML/Test/images/'
                out_filename='/home/mfpenuela/FetaProjectAML/Results/testImagesBase/'
                img = Image.open(filename+names[j]+str(i)+'.png')

                mask = predict_img(net=net,
                                full_img=img,
                                scale_factor=args.scale,
                                out_threshold=args.mask_threshold,
                                device=device,
                                edad=edad[j])
            #print(mask.shape,edad[j],names[j])

                if not args.no_save:

                    result = mask_to_image(mask)
                    result.save(out_filename+names[j]+str(i)+'.png')
                    logging.info(f'Mask saved to {out_filename}')

        net = AttentionUNet(img_ch=1, output_ch=8)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #logging.info(f'Loading model {args.model}')
        logging.info(f'Using device {device}')

        net.to(device=device)
        net.load_state_dict(torch.load(modelAttentioFest, map_location=device))

        logging.info('Model loaded!')
        names=['sub-016-','sub-023-','sub-024-','sub-027-','sub-028-','sub-032-','sub-035-','sub-036-','sub-059-','sub-064-']
        edad=[23.3,23.7,30.4,26.5,31.1,32.3,38.3,22.7,34.8,27.8]
    #names=['sub-001-','sub-005-','sub-020-','sub-029-','sub-037-','sub-038-','sub-052-','sub-070-','sub-075-','sub-077-']
    #edad=[27.9,22.6,25.8,32.5,23.4,26.9,21.2,20.1,29.0,26.9]

        print('Attention test...')
        for j in range (len(names)):

            for i in range(256):
            #logging.info(f'\nPredicting image {filename} ...')
                filename='/home/mfpenuela/FetaProjectAML/Test/images/'
                out_filename='/home/mfpenuela/FetaProjectAML/Results/testImagesAttention/'
                img = Image.open(filename+names[j]+str(i)+'.png')

                mask = predict_img(net=net,
                                full_img=img,
                                scale_factor=args.scale,
                                out_threshold=args.mask_threshold,
                                device=device,
                                edad=edad[j])
            #print(mask.shape,edad[j],names[j])

                if not args.no_save:

                    result = mask_to_image(mask)
                    result.save(out_filename+names[j]+str(i)+'.png')
                    logging.info(f'Mask saved to {out_filename}')
        diceTest()
        guardarTest()
    if args.mode=='demo':
        net = AttentionUNetBase(img_ch=1, output_ch=8)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #logging.info(f'Loading model {args.model}')
        logging.info(f'Using device {device}')

        net.to(device=device)
        net.load_state_dict(torch.load(modelAttentionBase, map_location=device))

        logging.info('Model loaded!')
        names=['sub-016-','sub-023-','sub-024-','sub-027-','sub-028-','sub-032-','sub-035-','sub-036-','sub-059-','sub-064-']
        edad=[23.3,23.7,30.4,26.5,31.1,32.3,38.3,22.7,34.8,27.8]
    #names=['sub-001-','sub-005-','sub-020-','sub-029-','sub-037-','sub-038-','sub-052-','sub-070-','sub-075-','sub-077-']
    #edad=[27.9,22.6,25.8,32.5,23.4,26.9,21.2,20.1,29.0,26.9]

    
        print('Base test...')
        for j in range (len(names)):
            if args.img+'-'!=names[j]:
                continue 

            for i in range(256):
            #logging.info(f'\nPredicting image {filename} ...')
                filename='/home/mfpenuela/FetaProjectAML/Test/images/'
                out_filename='/home/mfpenuela/FetaProjectAML/Results/DemoImagesBase/'
                img = Image.open(filename+names[j]+str(i)+'.png')

                mask = predict_img(net=net,
                                full_img=img,
                                scale_factor=args.scale,
                                out_threshold=args.mask_threshold,
                                device=device,
                                edad=edad[j])
            #print(mask.shape,edad[j],names[j])

                if not args.no_save:

                    result = mask_to_image(mask)
                    result.save(out_filename+names[j]+str(i)+'.png')
                    logging.info(f'Mask saved to {out_filename}')

        net = AttentionUNet(img_ch=1, output_ch=8)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #logging.info(f'Loading model {args.model}')
        logging.info(f'Using device {device}')

        net.to(device=device)
        net.load_state_dict(torch.load(modelAttentioFest, map_location=device))

        logging.info('Model loaded!')
        names=['sub-016-','sub-023-','sub-024-','sub-027-','sub-028-','sub-032-','sub-035-','sub-036-','sub-059-','sub-064-']
        edad=[23.3,23.7,30.4,26.5,31.1,32.3,38.3,22.7,34.8,27.8]
    #names=['sub-001-','sub-005-','sub-020-','sub-029-','sub-037-','sub-038-','sub-052-','sub-070-','sub-075-','sub-077-']
    #edad=[27.9,22.6,25.8,32.5,23.4,26.9,21.2,20.1,29.0,26.9]

        print('Attention test...')
        for j in range (len(names)):
            if args.img+'-'!=names[j]:
                continue 

            for i in range(256):
            #logging.info(f'\nPredicting image {filename} ...')
                filename='/home/mfpenuela/FetaProjectAML/Test/images/'
                out_filename='/home/mfpenuela/FetaProjectAML/Results/DemoImagesAttention/'
                img = Image.open(filename+names[j]+str(i)+'.png')

                mask = predict_img(net=net,
                                full_img=img,
                                scale_factor=args.scale,
                                out_threshold=args.mask_threshold,
                                device=device,
                                edad=edad[j])
            #print(mask.shape,edad[j],names[j])

                if not args.no_save:

                    result = mask_to_image(mask)
                    result.save(out_filename+names[j]+str(i)+'.png')
                    logging.info(f'Mask saved to {out_filename}')
        diceDemo(args.img)


