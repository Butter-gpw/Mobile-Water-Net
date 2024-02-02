import torch.nn as nn
import torch
import torchvision.models as models


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(weights='VGG19_Weights.DEFAULT').features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


def batch_normalize(tensor):
    batch = tensor.size(0)
    tensor = tensor.view(batch, -1)
    tensor = nn.functional.normalize(tensor, dim=1)
    return tensor


class Contrast_Learning_Loss(nn.Module):
    def __init__(self, device, T, requires_grad=False):
        super(Contrast_Learning_Loss, self).__init__()
        self.T = T
        self.device = device
        self.vgg = Vgg19(requires_grad=requires_grad).to(self.device)
        self.CrossEntropyLoss = nn.CrossEntropyLoss().to(self.device)
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, a, p, n):
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0

        for i in range(len(a_vgg)):
            anc = batch_normalize(a_vgg[i])
            pos = batch_normalize(p_vgg[i].detach())
            neg = batch_normalize(n_vgg[i].detach())

            l_pos = torch.einsum("nc,nc->n", [anc, pos]).unsqueeze(-1)
            l_neg = torch.einsum("nc,nc->n", [anc, neg]).unsqueeze(-1)


            logits = torch.concat([l_pos, l_neg], dim=1) / self.T
            labels = torch.zeros(logits.shape[0], dtype=torch.long)
            labels = labels.to(self.device)

            contrastive = self.CrossEntropyLoss(logits, labels)

            loss += self.weights[i] * contrastive
        return loss





if __name__ == '__main__':
    a = torch.randn((4, 3, 256, 256))
    p = torch.randn((4, 3, 256, 256))
    n = torch.randn((4, 3, 256, 256))


    a_v = batch_normalize(a)
    p_v = batch_normalize(p)
    n_v = batch_normalize(n)



