import argparse
import os
import pandas as pd
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import skimage.color
import skimage.io
import skimage.measure
from datautils_argument import TrainDatasetFromFolder, ValDatasetFromFolder
import argparse
from STAN12 import STN
from torch.optim import lr_scheduler


os.environ["CUDA_VISIBLE_DEVICES"] = "2"

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--crop_size', default=200, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                    help='super resolution upscale factor')
parser.add_argument('--num_epochs', default=400, type=int, help='train epoch number')
parser.add_argument('--relation', default=5, type=int, help='frames relation')
parser.add_argument('--longrelationtimes', default=6, type=int, help='long frames relation times')

opt = parser.parse_args()

CROP_SIZE = opt.crop_size
UPSCALE_FACTOR = opt.upscale_factor
NUM_EPOCHS = opt.num_epochs
RELATION = opt.relation

train_set = TrainDatasetFromFolder('/frame3', crop_size=CROP_SIZE, upscale_factor=UPSCALE_FACTOR,relation=RELATION)
val_set = ValDatasetFromFolder('/home/lixianbo/Desktop/RISTN(ADV)/HR1', upscale_factor=UPSCALE_FACTOR,relation=RELATION)
train_loader = DataLoader(dataset=train_set, num_workers=8, batch_size=8, shuffle=True)
# val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

netG = STN(relation=RELATION)

print('# generator parameters:', sum(param.numel() for param in netG.parameters()))

generator_criterion = torch.nn.MSELoss()
generator_criterion_Diff = torch.nn.MSELoss()

if torch.cuda.is_available():
    netG.cuda()
    generator_criterion.cuda()
    generator_criterion_Diff.cuda()

optimizerG = optim.Adam(netG.parameters(),lr=0.0002)
#optimizerG = torch.nn.DataParallel(optimizerG,device_ids=[0,1]).module

PSNR_results = {'psnr_calendar': [],'psnr_city': [],'psnr_foliage': [],'psnr_walk': []}
SSIM_results = {'ssim_calendar': [],'ssim_city': [],'ssim_foliage': [],'ssim_walk': []}
TDiff_results = {'tdiff_calendar': [],'tdiff_city': [],'tdiff_foliage': [],'tdiff_walk': []}

scheduler = lr_scheduler.StepLR(optimizerG,step_size=10,gamma=0.95)
Maxaverage = 0
for epoch in range(1, NUM_EPOCHS + 1):
    train_bar = tqdm(train_loader)
    running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
    scheduler.step()
    netG.train()

    netG.trainMode=True
    k = 0
    for data,data_bilr, target in train_bar:

        batch_size = data.size(0)
        running_results['batch_sizes'] += batch_size

        real_img = Variable(target)
        if torch.cuda.is_available():
            real_img = real_img.cuda()
        z = Variable(data)
        if torch.cuda.is_available():
            z = z.cuda()

        fake_img = netG(z,None)

        real_img = real_img.transpose(0, 1)

        # calc MSE LOss 2 is center
        totalloss = generator_criterion(fake_img[RELATION//2], real_img[RELATION//2])

        for q in range(len(fake_img) // 2):
            gimg = fake_img[q]
            gimg_b = fake_img[RELATION - q - 1]

            real = real_img[q]
            real_b = real_img[RELATION - q - 1]
            g1 = generator_criterion(gimg, real)
            gb = generator_criterion(gimg_b, real_b)


            totalloss += (g1 + gb)

        totalloss.backward()

        optimizerG.step()

        netG.zero_grad()

        train_bar.set_description(desc='%f' % (totalloss.data[0]))

    ############################
    #  evluation method
    ###########################
    netG.eval()
    netG.trainMode = False

    result = 0
    result1 = 0
    totalaverage = 0
    average = 0

    from TemporalEVal import evalTemporal

    for j in range(4):
        val_lr,val_bilr, val_hr = val_set.generatebatch(j)
        batch_size = val_lr.size(0)

        lr = Variable(val_lr, volatile=True)
        val_bilr = Variable(val_bilr, volatile=True)
        hr = Variable(val_hr, volatile=True)
        if torch.cuda.is_available():
            lr = lr.cuda()
            val_bilr = val_bilr.cuda()

        result = 0.0
        result1 = 0.0
        resultT = 0.0

        totalnumber = 0
        for k in range(batch_size):
            lrseq = lr[k]
            hrseq = hr.cpu()[k]
            biseq = val_bilr[k]

            lrseq = torch.unsqueeze(lrseq,dim=0)

            longrange = None

            if k % opt.longrelationtimes == 0:
                longrange = None

            srs, longrange = netG(lrseq,longrange)
            srs = srs.detach()
            for s in range(RELATION):
                srimg = srs[s]
                hrimg = hrseq[s]
                biimg = biseq[s][0]
                tdiff= 0.0

                if s >=1:
                    tdiff = evalTemporal(hrseq[s-1],hrseq[s],srs[s-1],srimg)
                    tdiff = tdiff.cpu().numpy()

                srdata = srimg.data[0].permute(1, 2, 0).cpu().numpy()
                hrdata = hrimg.permute(1, 2, 0).cpu().numpy()

                srdata[srdata < 0.0] = 0.0
                srdata[srdata > 1.0] = 1.0

                skimage.io.imsave("./tempout/%d_%d.jpg" % (j, totalnumber), srdata)
                cmpsr = skimage.color.rgb2ycbcr(srdata).astype('uint8')
                hrdata = skimage.color.rgb2ycbcr(hrdata).astype('uint8')
                cc = skimage.measure.compare_ssim(hrdata[:, :, 0], cmpsr[:, :, 0])
                bb = skimage.measure.compare_psnr(hrdata[:, :, 0], cmpsr[:, :, 0])

                result += bb
                result1 += cc
                resultT += tdiff
                totalnumber += 1

                del tdiff

        if j == 0:
            totalaverage += result/(batch_size*RELATION)
            PSNR_results['psnr_calendar'].append(result/(batch_size*RELATION))
            SSIM_results['ssim_calendar'].append(result1/(batch_size*RELATION))
            TDiff_results['tdiff_calendar'].append(resultT/(batch_size*RELATION))
        elif j == 1:
            totalaverage += result / (batch_size*RELATION)
            PSNR_results['psnr_city'].append(result/(batch_size*RELATION))
            SSIM_results['ssim_city'].append(result1 / (batch_size*RELATION))
            TDiff_results['tdiff_city'].append(resultT / (batch_size * RELATION))
        elif j == 2:
            totalaverage += result / (batch_size*RELATION)
            PSNR_results['psnr_foliage'].append(result/(batch_size*RELATION))
            SSIM_results['ssim_foliage'].append(result1 / (batch_size*RELATION))
            TDiff_results['tdiff_foliage'].append(resultT / (batch_size * RELATION))
        elif j==3:
            totalaverage += result / (batch_size*RELATION)
            PSNR_results['psnr_walk'].append(result/(batch_size*RELATION))
            SSIM_results['ssim_walk'].append(result1 / (batch_size*RELATION))
            TDiff_results['tdiff_walk'].append(resultT / (batch_size * RELATION))
    average = totalaverage/4.0

    if average>Maxaverage:
        torch.save(netG,'./epochs/STAN10MSE_SL5_5.pkl')
        Maxaverage = average

    out_path = 'statistics/'
    data_frame = pd.DataFrame(
        data={'psnr_calendar': PSNR_results['psnr_calendar'],'psnr_city': PSNR_results['psnr_city'],'psnr_foliage': PSNR_results['psnr_foliage'],'psnr_walk': PSNR_results['psnr_walk'],
              'ssim_calendar':SSIM_results['ssim_calendar'],'ssim_city':SSIM_results['ssim_city'],'ssim_foliage':SSIM_results['ssim_foliage'],'ssim_walk':SSIM_results['ssim_walk']},
        index=range(1, epoch+ 1))
    data_frame.to_csv(out_path + 'STAN10MSE_SL5' + str(UPSCALE_FACTOR) + '5.csv', index_label='Epoch')

    data_frame = pd.DataFrame(
        data={'tdiff_calendar': TDiff_results['tdiff_calendar'],'tdiff_city': TDiff_results['tdiff_city'],'tdiff_foliage': TDiff_results['tdiff_foliage'],'tdiff_walk': TDiff_results['tdiff_walk']},
        index=range(1, epoch+ 1))
    data_frame.to_csv(out_path + 'STAN10MSE_SL5_Tidff' + str(UPSCALE_FACTOR) + '5.csv', index_label='Epoch')