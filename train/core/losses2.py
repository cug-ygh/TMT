import math

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from core.utils import calculate_accuracy
# from utils import calculate_accuracy
from torch import Tensor



class OrthLoss(nn.Module):

    def __init__(self):
        super(OrthLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)
        

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss
        

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    将源域数据和目标域数据转化为核矩阵,即上文中的K
    Params: 
	    source: 源域数据(n * len(x))
	    target: 目标域数据(m * len(y))
	    kernel_mul: 
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		sum(kernel_val): 多个核矩阵之和
    '''
    n_samples = int(source.size()[0])+int(target.size()[0])# 求矩阵的行数,一般source和target的尺度是一样的,这样便于计算
    total = torch.cat([source, target], dim=0)#将source,target按列方向合并
    #将total复制(n+m）份
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #将total的每一行都复制成(n+m）行,即每个数据都扩展成(n+m）份
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    #求任意两个数据之间的和,得到的矩阵中坐标(i,j）代表total中第i行数据和第j行数据之间的l2 distance(i==j时为0）
    L2_distance = ((total0-total1)**2).sum(2) 
    #调整高斯核函数的sigma值
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    #以fix_sigma为中值,以kernel_mul为倍数取kernel_num个bandwidth值(比如fix_sigma为1时,得到[0.25,0.5,1,2,4]
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    #高斯核函数的数学表达式
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    #得到最终的核矩阵
    return sum(kernel_val)#/len(kernel_val)


def MmdLoss(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    计算源域数据和目标域数据的MMD距离
    Params: 
	    source: 源域数据(n * len(x))
	    target: 目标域数据(m * len(y))
	    kernel_mul: 
	    kernel_num: 取不同高斯核的数量
	    fix_sigma: 不同高斯核的sigma值
	Return:
		loss: MMD loss
    '''
    batch_size = int(source.size()[0])#一般默认为源域和目标域的batchsize相同
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    #根据式(3）将核矩阵分成4部分
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss#因为一般都是n==m,所以L矩阵一般不加入计算


class MultimodalLoss(nn.Module):
    def __init__(self, alpha, beta, delta, batch_size=16, device='cuda'):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.device = device
        self.cls_fn =  nn.CrossEntropyLoss() 
        #self.cls_fn =  nn.MSELoss() 
        self.orth_fn = OrthLoss()


    def forward(self, x_invariant, x_specific_a, x_specific_v, x_specific_t, x_visual, x_audio, x_text, cls_output, cls_label, epoch):

        orth_loss1 = (self.orth_fn(x_invariant.mean(dim=1), x_specific_a.mean(dim=1)) + self.orth_fn(x_invariant.mean(dim=1), x_specific_v.mean(dim=1)) \
                     + self.orth_fn(x_invariant.mean(dim=1), x_specific_t.mean(dim=1))) / 3

        orth_loss2 = (self.orth_fn(x_specific_a.mean(dim=1), x_specific_v.mean(dim=1)) + self.orth_fn(x_specific_v.mean(dim=1), x_specific_t.mean(dim=1)) \
                     + self.orth_fn(x_specific_t.mean(dim=1), x_specific_a.mean(dim=1))) / 3

        orth_loss = orth_loss1 + orth_loss2
        

        sim_loss = (MmdLoss(x_audio.mean(dim=1), x_specific_a.mean(dim=1)) + MmdLoss(x_visual.mean(dim=1), x_specific_v.mean(dim=1)) \
                   + MmdLoss(x_text.mean(dim=1), x_specific_t.mean(dim=1))) / 3
        
        # print(cls_output.shape)
        # exit(0) 
        cls_loss = self.cls_fn(cls_output, cls_label)
        # print(cls_loss)
        # exit(0)       
        loss =  self.alpha * orth_loss + self.beta * sim_loss + self.delta * cls_loss

        acc = calculate_accuracy(cls_output, cls_label)

        return loss, orth_loss, sim_loss, cls_loss, acc
