from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Inception3', 'inception_v3']


model_urls = {
    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}

_InceptionOuputs = namedtuple('InceptionOuputs', ['logits', 'aux_logits'])


def inception_v3(pretrained=False, progress=True, **kwargs):
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        aux_logits (bool): If True, add an auxiliary branch that can improve training.
            Default: *True*
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: *False*
    """
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        if 'aux_logits' in kwargs:
            original_aux_logits = kwargs['aux_logits']
            kwargs['aux_logits'] = True
        else:
            original_aux_logits = True
        model = Inception3(**kwargs)
        state_dict = load_state_dict_from_url(model_urls['inception_v3_google'],
                                              progress=progress)
        model.load_state_dict(state_dict)
        if not original_aux_logits:
            model.aux_logits = False
            del model.AuxLogits
        return model

    return Inception3(**kwargs)


class Inception3(nn.Module):

    def __init__(self, num_classes=10, aux_logits=False, transform_input=False):
        super(Inception3, self).__init__()
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = BasicConv2d(3, 96, kernel_size=3, stride=1, padding=1)
        self.Mixed_2a = Inception(96, 32, 32)
        self.Mixed_2b = Inception(32+32, 32, 48)
        self.Mixed_2c = Downsample(32+48, 80)
        self.Mixed_3a = Inception(32+48+80, 112, 48)
        self.Mixed_3b = Inception(112+48, 96, 64)
        self.Mixed_3c = Inception(96+64, 80, 80)
        self.Mixed_3d = Inception(80+80, 48, 96)
        self.Mixed_3e = Downsample(48+96, 96)
        self.Mixed_4a = Inception(48+96+96, 176, 160)
        self.Mixed_4b = Inception(176+160, 176, 160)
        self.fc = nn.Linear(176+160, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)            
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)

        x = self.Conv2d_1a_3x3(x)
        x = self.Mixed_2a(x)
        x = self.Mixed_2b(x)
        x = self.Mixed_2c(x)
        x = self.Mixed_3a(x)
        x = self.Mixed_3b(x)
        x = self.Mixed_3c(x)
        x = self.Mixed_3d(x)
        x = self.Mixed_3e(x)
        x = self.Mixed_4a(x)
        x = self.Mixed_4b(x)
        x = F.avg_pool2d(x, (7, 7))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Inception(nn.Module):

    def __init__(self, in_channels, out_channels_k1x1, out_channels_k3x3):
        super(Inception, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, out_channels_k1x1, kernel_size=1, stride=1)
        self.branch3x3 = BasicConv2d(in_channels, out_channels_k3x3, kernel_size=3, stride=1, padding=1)
        self.info = in_channels, out_channels_k1x1, out_channels_k3x3

    def forward(self, x):
        #print(self.info)
        #print(x.shape)
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        #print(branch1x1.shape)
        #print(branch3x3.shape)
        outputs = [branch1x1, branch3x3]
        return torch.cat(outputs, 1)

class Downsample(nn.Module):

    def __init__(self, in_channels, out_channels_k3x3):
        super(Downsample, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, out_channels_k3x3, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        #print(x.shape)
        #print(branch3x3.shape)
        #print(branch_pool.shape)
        outputs = [branch3x3, branch_pool]
        return torch.cat(outputs, 1)

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=True, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
