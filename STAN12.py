from models.RRIN import SRRIN
from  RDCRNN import RDCRNN
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as  np
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F




class CA(nn.Module):
    def __init__(self,inchannel,outchannel):
        super(CA, self).__init__()

        self.W1  = nn.Sequential(nn.Conv2d(inchannel,inchannel,3,1,1),
                                 nn.ReLU()
        )
        self.ebed = nn.Linear(inchannel, inchannel // 2)
        self.map = nn.Tanh()
        self.decode = nn.Linear(inchannel // 2, outchannel)

    def forward(self,x):
        x = self.W1(x)
        pooledfeatures = F.adaptive_avg_pool2d(x, 1)
        pooledfeatures = torch.squeeze(pooledfeatures, dim=3)
        pooledfeatures = torch.squeeze(pooledfeatures, dim=2)

        xemb = self.ebed(pooledfeatures)
        map = self.map(xemb)
        decode1 = self.decode(map)

        attention1 = F.tanh(decode1)
        attention1 = torch.unsqueeze(attention1, dim=2)
        attention1 = torch.unsqueeze(attention1, dim=3)
        return attention1


class Reconstruction(nn.Module):
    def __init__(self,inchannel,headLevel=4):
        super(Reconstruction, self).__init__()


        self.up = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(128,128,3,1,padding=1),
            nn.PReLU(),
            nn.PixelShuffle(2),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=512, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(),
        )

        self.reconsturction = nn.Sequential(
            nn.Conv2d(32,3,3,1,1),
            nn.ReLU()
        )

    def forward(self, x):
        A = self.CA(x)
        x = A*x + x
        x = self.fusion(x)
        x = self.up(x)
        return self.reconsturction(x)



class STN(nn.Module):
    def __init__(self,relation):
        super(STN, self).__init__()
        self.realtionlen = relation


        self.sptioCNN = SRRIN()
        self.sptioCNN  = torch.nn.DataParallel(self.sptioCNN)
        self.sptioCNN.load_state_dict(torch.load('../Spatial/netG_epoch_4_209.pth'))
        #self.sptioCNN.cuda()

        self.temporalRNN = RDCRNN(256, 3, 16, 12)
        # self.recon = Reconsturcture(256)
        self.trainMode = True

        self.eaualization = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

        self.CA = CA(384, 384)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        #self.reconstruction = Reconstruction(320,4)

    def calc_sp(self, x):
        t2 = None
        for i in range(self.realtionlen):
            ax = x[i]

            ax_s = self.sptioCNN(ax)
            ax_s = torch.unsqueeze(ax_s, 0)

            if i == 0:
                t2 = ax_s
            else:
                t2 = torch.cat((t2, ax_s), 0)
        return  t2

    def featureforward(self, x,longrange):
        # now is seq_len,B,C,H,W
        x = x.transpose(0, 1)
        if self.trainMode:
            s = self.calc_sp(x)
            self.temporalRNN.disablelongrange()
            t, _ = self.temporalRNN(s)
        else:
            s = checkpoint(self.calc_sp, x)
            self.temporalRNN.setlongrange(longrange)
            hidden, newlongrange = self.temporalRNN.featureFoward(s)

            return hidden, newlongrange


    def forward(self, x,longrange):

        # now is seq_len,B,C,H,W
        x = x.transpose(0, 1)
        if self.trainMode:
            s = self.calc_sp(x)
            self.temporalRNN.disablelongrange()
            t,_ = self.temporalRNN(s)
        else:
            s = checkpoint(self.calc_sp, x)
            self.temporalRNN.setlongrange(longrange)
            t,newlongrange = self.temporalRNN(s)

        out = []
        for i in range(len(x)):
            axs = s[i]
            axs = self.eaualization(axs)

            ax = t[i]

            totalfeature = axs+ax

            #Feature fusiom
            #totalfeature = torch.cat((axs,totalfeature),1)

            A = self.CA(totalfeature)
            totalfeature = A * totalfeature + totalfeature
            totalfeature = self.fusion(totalfeature)
            if self.trainMode:
                rec = self.sptioCNN.module.reconstructure(totalfeature)
            else:
                rec = checkpoint(self.sptioCNN.module.reconstructure, totalfeature)

            out.append(torch.unsqueeze(rec,0))


        if self.trainMode == False:
            return torch.cat(out,0),newlongrange
        return torch.cat(out,0)


