'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self,vgg_name, num_classes):
        super(VGG, self).__init__()
        self.features = make_layers(cfg_list[vgg_name])
        self.classifier = make_layers([('L', 8192, 1024), ('L', 1024, num_classes)]) ###vgg8

        #self.classifier = make_layers([('L', 512, 4096), ('L', 4096, 4096), ('L', 4096, num_classes)]) ###vgg11, vgg16,vgg19
        
        

    def forward(self, x):
        x = self.features(x)
        #x = self.Avgpool2d(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v[0] == 'M':
            layers += [nn.MaxPool2d(kernel_size=v[1], stride=v[2])]
        if v[0] == 'C':
            out_channels = v[1]
            if v[3] == 'same':
                padding = v[2]//2
            else:
                padding = 0
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=v[2], bias = False, padding=padding)
            non_linearity_activation =  nn.ReLU()
            layers += [conv2d,non_linearity_activation]
            in_channels = out_channels
        if v[0] == 'L':
            linear = nn.Linear(in_features=v[1], out_features=v[2], bias = False)
            if i < len(cfg)-1:
                non_linearity_activation =  nn.ReLU()
                layers += [linear, non_linearity_activation]
            else:
                layers += [linear]
    return nn.Sequential(*layers)



cfg_list = {
    'vgg8': [('C', 128, 3, 'same', 2.0),
                ('C', 128, 3, 'same', 16.0),
                ('M', 2, 2),
                ('C', 256, 3, 'same', 16.0),
                ('C', 256, 3, 'same', 16.0),
                ('M', 2, 2),
                ('C', 512, 3, 'same', 16.0),
                ('C', 512, 3, 'same', 32.0),
                ('M', 2, 2)],
                
    'vgg11': [('C', 64, 3, 'same', 2.0),
                ('M', 2, 2),
                ('C', 128, 3, 'same', 16.0),
                ('M', 2, 2),
                ('C', 256, 3, 'same', 16.0),
                ('C', 256, 3, 'same', 16.0),
                ('M', 2, 2),
                ('C', 512, 3, 'same', 16.0),
                ('C', 512, 3, 'same', 32.0),
                ('M', 2, 2),
                ('C', 512, 3, 'same', 16.0),
                ('C', 512, 3, 'same', 32.0),
                ('M', 2, 2)],
    
    'vgg16': [('C', 64, 3, 'same', 2.0),
                ('C', 64, 3, 'same', 2.0),
                ('M', 2, 2),
                ('C', 128, 3, 'same', 16.0),
                ('C', 128, 3, 'same', 16.0),
                ('M', 2, 2),
                ('C', 256, 3, 'same', 16.0),
                ('C', 256, 3, 'same', 16.0),
                ('C', 256, 3, 'same', 16.0),
                ('M', 2, 2),
                ('C', 512, 3, 'same', 16.0),
                ('C', 512, 3, 'same', 32.0),
                ('C', 512, 3, 'same', 32.0),
                ('M', 2, 2),
                ('C', 512, 3, 'same', 16.0),
                ('C', 512, 3, 'same', 32.0),
                ('C', 512, 3, 'same', 32.0),
                ('M', 2, 2)],
    
    'vgg19': [('C', 64, 3, 'same', 2.0),
                ('C', 64, 3, 'same', 2.0),
                ('M', 2, 2),
                ('C', 128, 3, 'same', 16.0),
                ('C', 128, 3, 'same', 16.0),
                ('M', 2, 2),
                ('C', 256, 3, 'same', 16.0),
                ('C', 256, 3, 'same', 16.0),
                ('C', 256, 3, 'same', 16.0),
                ('C', 256, 3, 'same', 16.0),
                ('M', 2, 2),
                ('C', 512, 3, 'same', 16.0),
                ('C', 512, 3, 'same', 32.0),
                ('C', 512, 3, 'same', 16.0),
                ('C', 512, 3, 'same', 16.0),
                ('M', 2, 2),
                ('C', 512, 3, 'same', 16.0),
                ('C', 512, 3, 'same', 32.0),
                ('C', 512, 3, 'same', 32.0),
                ('C', 512, 3, 'same', 32.0),
                ('M', 2, 2)]
}

