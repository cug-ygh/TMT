import math

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from core.utils import calculate_accuracy

from torch import Tensor


class GanTrainLoss(nn.Module):
    def __init__(self, alpha, beta, delta, batch_size=16, device='cuda'):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.device = device
        self.SpecificLoss_Fn = nn.CrossEntropyLoss()
        self.InvariantLoss_Fn = nn.CrossEntropyLoss()
        self.ClsLoss_Fn = nn.CrossEntropyLoss()
        # self.Specific_Visual_Label = torch.zeros(batch_size, 2).scatter_(1, torch.zeros((batch_size, 1), dtype = torch.int64), 1).to(self.device)
        # self.Specific_Audio_Label = torch.zeros(batch_size, 2).scatter_(1, torch.zeros((batch_size, 1), dtype = torch.int64), 1).to(self.device)
        # self.Invariant_Visual_Label = torch.zeros(batch_size, 2).scatter_(1, torch.ones((batch_size, 1), dtype = torch.int64), 1).to(self.device)
        # self.Invariant_Audio_Label = torch.zeros(batch_size, 2).scatter_(1, torch.ones((batch_size, 1), dtype = torch.int64), 1).to(self.device)


    def forward(self, D0_visual_op, D0_audio_op, D2_visual_op, D2_audio_op, cls_output, cls_label, epoch):
        batch_size = D0_visual_op.shape[0]
        Specific_Visual_Label = torch.zeros((batch_size)).long().to(self.device)
        Specific_Audio_Label = torch.ones((batch_size)).long().to(self.device)
        Invariant_Visual_Label = torch.zeros((batch_size)).long().to(self.device)
        Invariant_Audio_Label = torch.ones((batch_size)).long().to(self.device)
        # print(D0_visual_op, self.Specific_Visual_Label, cls_label)

        self.beta = 0.0002 * ( (2 / (1 + math.exp(-10 * (0 + epoch * 0.001)))) - 1)
        self.alpha = self.beta

        SpecificLoss = self.alpha * ( self.SpecificLoss_Fn(D2_visual_op, Specific_Visual_Label)  +
                                      self.SpecificLoss_Fn(D2_audio_op, Specific_Audio_Label) )
        # print(cls_output, cls_label, cls_label)
        InvariantLoss = self.beta * ( self.InvariantLoss_Fn(D0_visual_op, Invariant_Visual_Label)  +
                                      self.InvariantLoss_Fn(D0_audio_op, Invariant_Audio_Label) )

        ClsLoss = self.delta * self.ClsLoss_Fn(cls_output, cls_label)

        loss =  SpecificLoss + InvariantLoss + ClsLoss

        # loss = ClsLoss

        ClsAcc = calculate_accuracy(cls_output, cls_label)
        Specific_Cls_Acc = ( calculate_accuracy(D2_visual_op, Specific_Visual_Label) + calculate_accuracy(D2_audio_op, Specific_Audio_Label) ) / 2
        Invariant_Cls_Acc = ( calculate_accuracy(D0_visual_op, Invariant_Visual_Label) + calculate_accuracy(D0_audio_op, Invariant_Audio_Label) ) / 2

        return loss, SpecificLoss, InvariantLoss, ClsLoss, ClsAcc, Specific_Cls_Acc, Invariant_Cls_Acc


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.01, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)



class MultimodalLoss_Mixup(nn.Module):
    def __init__(self, alpha, beta, delta, batch_size=16, device='cuda'):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.device = device
        self.SpecificLoss_Fn = nn.CrossEntropyLoss()
        self.InvariantLoss_Fn = nn.CrossEntropyLoss()
        self.ClsLoss_Fn = nn.CrossEntropyLoss()


    def forward(self, D0_visual_op, D0_audio_op, D0_text_op, D2_visual_op, D2_audio_op, D2_text_op,
                cls_output, cls_label_a, cls_label_b, lam, epoch):
        batch_size = D0_visual_op.shape[0]
        Specific_Visual_Label = torch.zeros((batch_size)).long().to(self.device)
        Specific_Audio_Label = torch.ones((batch_size)).long().to(self.device)
        Specific_Text_Label = torch.ones((batch_size)).long().to(self.device) * 2
        Invariant_Visual_Label = torch.zeros((batch_size)).long().to(self.device)
        Invariant_Audio_Label = torch.ones((batch_size)).long().to(self.device)
        Invariant_Text_Label = torch.ones((batch_size)).long().to(self.device) * 2

        SpecificLoss = self.alpha * ( self.SpecificLoss_Fn(D2_visual_op, Specific_Visual_Label)  +
                                      self.SpecificLoss_Fn(D2_audio_op, Specific_Audio_Label) + self.SpecificLoss_Fn(D2_text_op, Specific_Text_Label))

        InvariantLoss = self.beta * ( self.InvariantLoss_Fn(D0_visual_op, Invariant_Visual_Label)  +
                                      self.InvariantLoss_Fn(D0_audio_op, Invariant_Audio_Label) + self.InvariantLoss_Fn(D0_text_op, Invariant_Text_Label))

        ClsLoss = self.delta * (lam * self.ClsLoss_Fn(cls_output, cls_label_a) + (1-lam) * self.ClsLoss_Fn(cls_output, cls_label_b))

        loss =  SpecificLoss + InvariantLoss + ClsLoss

        _, predicted = torch.max(cls_output, 1)
        ClsAcc = (lam * predicted.eq(cls_label_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(cls_label_b.data).cpu().sum().float())

        Specific_Cls_Acc = ( calculate_accuracy(D2_visual_op, Specific_Visual_Label) + calculate_accuracy(D2_audio_op, Specific_Audio_Label) + calculate_accuracy(D2_text_op, Specific_Text_Label)) / 3
        Invariant_Cls_Acc = ( calculate_accuracy(D0_visual_op, Invariant_Visual_Label) + calculate_accuracy(D0_audio_op, Invariant_Audio_Label) + calculate_accuracy(D0_text_op, Invariant_Text_Label)) / 3

        return loss, SpecificLoss, InvariantLoss, ClsLoss, ClsAcc, Specific_Cls_Acc, Invariant_Cls_Acc


class MultimodalLoss(nn.Module):
    def __init__(self, alpha, beta, delta, batch_size=16, device='cuda'):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        if alpha == 0:
            self.is_alpha_0 = True
        else:
            self.is_alpha_0 = False
        if beta == 0:
            self.is_beta_0 = True
        else:
            self.is_beta_0 = False
        self.device = device
        self.SpecificLoss_Fn = nn.CrossEntropyLoss()
        self.InvariantLoss_Fn = nn.CrossEntropyLoss()
#        self.ClsLoss_Fn = nn.MSELoss()
        self.ClsLoss_Fn = nn.CrossEntropyLoss(reduction='mean')


    def forward(self, D0_visual_op, D0_audio_op, D0_text_op, D2_visual_op, D2_audio_op, D2_text_op,
                cls_output, cls_label, epoch):
        batch_size = D0_visual_op.shape[0]
        Specific_Visual_Label = torch.zeros((batch_size)).long().to(self.device)
        Specific_Audio_Label = torch.ones((batch_size)).long().to(self.device)
        Specific_Text_Label = torch.ones((batch_size)).long().to(self.device) * 1
        Invariant_Visual_Label = torch.zeros((batch_size)).long().to(self.device)
        Invariant_Audio_Label = torch.ones((batch_size)).long().to(self.device)
        Invariant_Text_Label = torch.ones((batch_size)).long().to(self.device) * 1
        # print(D0_visual_op, self.Specific_Visual_Label, cls_label)

        if self.is_alpha_0 == False:
            self.alpha = 0.002 * ( (2 / (1 + math.exp(-10 * (0 + epoch * 0.02)))) - 1)
        else:
            self.alpha = 0
        if self.is_beta_0 == False:
            self.beta = 0.002 * ( (2 / (1 + math.exp(-10 * (0 + epoch * 0.02)))) - 1)
        else:
            self.alpha = 0
        # print(self.alpha, self.beta)

        self.alpha = 0
        self.beta = 0

        SpecificLoss = self.alpha * ( self.SpecificLoss_Fn(D2_visual_op, Specific_Visual_Label)  +
                                      self.SpecificLoss_Fn(D2_audio_op, Specific_Audio_Label) + 
                                      self.SpecificLoss_Fn(D2_text_op, Specific_Text_Label) ) 

        InvariantLoss = self.beta * ( self.InvariantLoss_Fn(D0_visual_op, Invariant_Visual_Label)  +
                                      self.InvariantLoss_Fn(D0_audio_op, Invariant_Audio_Label) + 
                                      self.InvariantLoss_Fn(D0_text_op, Invariant_Text_Label) ) 

        ClsLoss = self.delta * self.ClsLoss_Fn(cls_output, cls_label)

        loss =  SpecificLoss + InvariantLoss + ClsLoss
        # print(cls_output)
        # print(cls_label)
        # exit(0)
        ClsAcc = calculate_accuracy(cls_output, cls_label)
        Specific_Cls_Acc = ( calculate_accuracy(D2_visual_op, Specific_Visual_Label) + calculate_accuracy(D2_audio_op, Specific_Audio_Label) + calculate_accuracy(D2_text_op, Specific_Text_Label) ) / 3
        Invariant_Cls_Acc = ( calculate_accuracy(D0_visual_op, Invariant_Visual_Label) + calculate_accuracy(D0_audio_op, Invariant_Audio_Label) + calculate_accuracy(D0_text_op, Invariant_Text_Label)) / 3

        return loss, SpecificLoss, InvariantLoss, ClsLoss, ClsAcc, Specific_Cls_Acc, Invariant_Cls_Acc



class MultimodalLoss_Reg(nn.Module):
    def __init__(self, alpha, beta, delta, batch_size=16, device='cuda'):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        if alpha == 0:
            self.is_alpha_0 = True
        else:
            self.is_alpha_0 = False
        if beta == 0:
            self.is_beta_0 = True
        else:
            self.is_beta_0 = False
        self.device = device
        self.SpecificLoss_Fn = nn.CrossEntropyLoss()
        self.InvariantLoss_Fn = nn.CrossEntropyLoss()
        self.RegLoss_Fn = nn.L1Loss()
        # self.RegLoss_Fn = nn.MSELoss()


    def forward(self, D0_visual_op, D0_audio_op, D0_text_op, D2_visual_op, D2_audio_op, D2_text_op,
                cls_output, reg_label, epoch):
        batch_size = D0_visual_op.shape[0]
        Specific_Visual_Label = torch.zeros((batch_size)).long().to(self.device)
        Specific_Audio_Label = torch.ones((batch_size)).long().to(self.device)
        Specific_Text_Label = torch.ones((batch_size)).long().to(self.device)
        Invariant_Visual_Label = torch.zeros((batch_size)).long().to(self.device)
        Invariant_Audio_Label = torch.ones((batch_size)).long().to(self.device)
        Invariant_Text_Label = torch.ones((batch_size)).long().to(self.device)
        # print(D0_visual_op, self.Specific_Visual_Label, cls_label)

        if self.is_alpha_0 == False:
            self.alpha = 0.002 * ( (2 / (1 + math.exp(-10 * (0 + epoch * 0.02)))) - 1)
        else:
            self.alpha = 0
        if self.is_beta_0 == False:
            self.beta = 0.002 * ( (2 / (1 + math.exp(-10 * (0 + epoch * 0.02)))) - 1)
        else:
            self.alpha = 0
        # print(self.alpha, self.beta)

        SpecificLoss = self.alpha * ( self.SpecificLoss_Fn(D2_visual_op, Specific_Visual_Label)  +
                                      self.SpecificLoss_Fn(D2_audio_op, Specific_Audio_Label) + 
                                      self.SpecificLoss_Fn(D2_text_op, Specific_Text_Label) )

        InvariantLoss = self.beta * ( self.InvariantLoss_Fn(D0_visual_op, Invariant_Visual_Label)  +
                                      self.InvariantLoss_Fn(D0_audio_op, Invariant_Audio_Label) + 
                                      self.InvariantLoss_Fn(D0_text_op, Invariant_Text_Label) )

        ClsLoss = self.delta * self.RegLoss_Fn(cls_output, reg_label)

        loss =  SpecificLoss - InvariantLoss + ClsLoss

        ClsAcc = 0
        Specific_Cls_Acc = ( calculate_accuracy(D2_visual_op, Specific_Visual_Label) + calculate_accuracy(D2_audio_op, Specific_Audio_Label) ) / 2
        Invariant_Cls_Acc = ( calculate_accuracy(D0_visual_op, Invariant_Visual_Label) + calculate_accuracy(D0_audio_op, Invariant_Audio_Label) ) / 2

        return loss, SpecificLoss, InvariantLoss, ClsLoss, ClsAcc, Specific_Cls_Acc, Invariant_Cls_Acc

class MultimodalLoss_VA(nn.Module):
    def __init__(self, alpha, beta, delta, batch_size=16, device='cuda'):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        if alpha == 0:
            self.is_alpha_0 = True
        else:
            self.is_alpha_0 = False
        if beta == 0:
            self.is_beta_0 = True
        else:
            self.is_beta_0 = False
        self.device = device
        self.SpecificLoss_Fn = nn.CrossEntropyLoss()
        self.InvariantLoss_Fn = nn.CrossEntropyLoss()
        self.ClsLoss_Fn = nn.CrossEntropyLoss(reduction='mean')


    def forward(self, D0_visual_op, D0_audio_op, D2_visual_op, D2_audio_op,
                cls_output, cls_label, epoch):
        batch_size = D0_visual_op.shape[0]
        Specific_Visual_Label = torch.zeros((batch_size)).long().to(self.device)
        Specific_Audio_Label = torch.ones((batch_size)).long().to(self.device)
        # Specific_Text_Label = torch.ones((batch_size)).long().to(self.device) * 1
        Invariant_Visual_Label = torch.zeros((batch_size)).long().to(self.device)
        Invariant_Audio_Label = torch.ones((batch_size)).long().to(self.device)
        # Invariant_Text_Label = torch.ones((batch_size)).long().to(self.device) * 1
        # print(D0_visual_op, self.Specific_Visual_Label, cls_label)

        if self.is_alpha_0 == False:
            self.alpha = 0.002 * ( (2 / (1 + math.exp(-10 * (0 + epoch * 0.02)))) - 1)
        else:
            self.alpha = 0
        if self.is_beta_0 == False:
            self.beta = 0.002 * ( (2 / (1 + math.exp(-10 * (0 + epoch * 0.02)))) - 1)
        else:
            self.alpha = 0
        # print(self.alpha, self.beta)

        self.alpha = 0.01
        self.beta = 0.01
        SpecificLoss = self.alpha * ( self.SpecificLoss_Fn(D2_visual_op, Specific_Visual_Label)  +
                                      self.SpecificLoss_Fn(D2_audio_op, Specific_Audio_Label) )
                                      
        # SpecificLoss = self.alpha * ( self.SpecificLoss_Fn(D2_visual_op, Specific_Visual_Label)  +
        #                               self.SpecificLoss_Fn(D2_audio_op, Specific_Audio_Label) + 
        #                               self.SpecificLoss_Fn(D2_text_op, Specific_Text_Label) ) 

        # InvariantLoss = self.beta * ( self.InvariantLoss_Fn(D0_visual_op, Invariant_Visual_Label)  +
        #                               self.InvariantLoss_Fn(D0_audio_op, Invariant_Audio_Label) + 
        #                               self.InvariantLoss_Fn(D0_text_op, Invariant_Text_Label) ) 
        InvariantLoss = self.beta * ( self.InvariantLoss_Fn(D0_visual_op, Invariant_Visual_Label)  +
                                      self.InvariantLoss_Fn(D0_audio_op, Invariant_Audio_Label) ) 
        ClsLoss = self.delta * self.ClsLoss_Fn(cls_output, cls_label)

        loss =  SpecificLoss + InvariantLoss + ClsLoss

        ClsAcc = calculate_accuracy(cls_output, cls_label)
        Specific_Cls_Acc = ( calculate_accuracy(D2_visual_op, Specific_Visual_Label) + calculate_accuracy(D2_audio_op, Specific_Audio_Label) ) / 2
        Invariant_Cls_Acc = ( calculate_accuracy(D0_visual_op, Invariant_Visual_Label) + calculate_accuracy(D0_audio_op, Invariant_Audio_Label) ) / 2
        return loss, SpecificLoss, InvariantLoss, ClsLoss, ClsAcc, Specific_Cls_Acc, Invariant_Cls_Acc


if __name__ == '__main__':
    a = torch.randn(16, 1, 512).cuda()
    b = torch.randn(16, 1, 512).cuda()
    loss_fn = TrainLoss(alpha=0.5, bata=0.5, batch_size=16, device='cuda', temperature=0.5)
    # loss_fn = ContrastiveLoss(batch_size=4)
    loss_con, acc = loss_fn(a, b, a, b)
    print(loss_con, acc)
