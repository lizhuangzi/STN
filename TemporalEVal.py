import torch
import Flowextraction as Flowextraction

def evalTemporal(xt1,xt2,WX1,WX2):

    xt1= torch.unsqueeze(xt1,0)
    xt2 = torch.unsqueeze(xt2, 0)
    Compimg = Flowextraction.GetFlowImage(xt1,xt2,WX1)
    Compimg = Compimg.detach()
    G2 = Flowextraction.process(WX2)
    G2 = G2.detach()

    diff = (Compimg - G2)**2
    diff = torch.sqrt(diff)
    diff = torch.mean(diff, 0)
    diff = torch.mean(diff, 0)
    diff = torch.mean(diff, 0)
    diff = torch.mean(diff, 0)

    del G2
    del Compimg
    return diff
