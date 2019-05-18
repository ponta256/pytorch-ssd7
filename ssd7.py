import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc_ssd7
import os

class SSD7(nn.Module):

    def __init__(self, phase, size, num_classes):

        def multibox(base, num_classes):
            loc_layers = []
            conf_layers = []
            source = [12, 16, 20, 24]

            for k, v in enumerate(source):
                loc_layers += [nn.Conv2d(base[v].out_channels, 4*4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(base[v].out_channels, 4*num_classes, kernel_size=3, padding=1)]

            return (loc_layers, conf_layers)

        
        super(SSD7, self).__init__()

        # SSD network
        base = [nn.Conv2d(3, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1)), # 1 (0)
                nn.BatchNorm2d(32), # (1)
                nn.ReLU(inplace=True), # (2)
                nn.MaxPool2d(2, 2), # (3)
            
                nn.Conv2d(32, 48, kernel_size=(3,3), stride=(1,1), padding=(1,1)), # 2 (4)
                nn.BatchNorm2d(48), # (5)
                nn.ReLU(inplace=True), # (6)
                nn.MaxPool2d(2, 2), # (7)
            
                nn.Conv2d(48, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1)), # 3 (8)
                nn.BatchNorm2d(64), # (9)
                nn.ReLU(inplace=True), # (10)
                nn.MaxPool2d(2, 2), # (11)
            
                nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1)), # 4 (12)
                nn.BatchNorm2d(64), # (13)
                nn.ReLU(inplace=True), # (14*)
                nn.MaxPool2d(2, 2), # (15)
            
                nn.Conv2d(64, 48, kernel_size=(3,3), stride=(1,1), padding=(1,1)), # 5 (16)
                nn.BatchNorm2d(48), # (17)
                nn.ReLU(inplace=True), # (18*)
                nn.MaxPool2d(2, 2), # (19)
            
                nn.Conv2d(48, 48, kernel_size=(3,3), stride=(1,1), padding=(1,1)), # 6 (20)
                nn.BatchNorm2d(48), # (21)
                nn.ReLU(inplace=True), # (22*)
                nn.MaxPool2d(2, 2), # (23)
            
                nn.Conv2d(48, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1)), # 7 (24)
                nn.BatchNorm2d(32), # (25)
                nn.ReLU(inplace=True)] # (26*)
    
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = voc_ssd7
        print(self.cfg)

        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size
        
        self.base = nn.ModuleList(base)
        loc, conf = multibox(self.base, num_classes)
        self.loc = nn.ModuleList(loc)
        self.conf = nn.ModuleList(conf)

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()

        # [14, 18, 22, 26]        
        for k in range(15):
            x = self.base[k](x)
        sources.append(x)
        for k in range(15, 19):
            x = self.base[k](x)
        sources.append(x)
        for k in range(19, 23):
            x = self.base[k](x)
        sources.append(x)
        for k in range(23, 27):
            x = self.base[k](x)
        sources.append(x)            

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        # print(self.priors.type(type(x.data)))
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output
    
    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')



def build_ssd7(phase, num_classes=21):
    size = 300

    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return

    return SSD7(phase, size, num_classes)
