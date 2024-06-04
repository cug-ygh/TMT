import os
import torch
import numpy as np
from tqdm import tqdm
from core.dataset_new import MMDataLoader
from core.losses2 import MultimodalLoss
# from core.losses import MultimodalLoss, MultimodalLoss_Mixup, MultimodalLoss_Reg
from core.optimizer import get_optimizer
from core.scheduler import get_scheduler
from core.utils import AverageMeter, save_model, setup_seed, save_model_reg, mixup_data
from tensorboardX import SummaryWriter
#from models.MyModel_difusion import MyMultimodal_Integrally
from models.MyModel import MyMultimodal_Integrally
import argparse
from core.metric import MetricsTop
from torch.autograd import Variable
from core.metric import cal_acc5

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(device)


parser = argparse.ArgumentParser()
# dataset
parser.add_argument("--datasetName", type=str, default="sims", required=False)
parser.add_argument("--dataPath", type=str, default=r"//home/yuanyuan/ygh/unaligned_39.pkl", required=False)
parser.add_argument("--use_bert", type=bool, default=False, required=False)
parser.add_argument("--need_data_aligned", type=bool, default=True, required=False)
parser.add_argument("--need_truncated", type=bool, default=True, required=False)
parser.add_argument("--data_missing", type=bool, default=False, required=False)
parser.add_argument("--seq_lens", type=tuple, default=[30,30,30], required=False)
#parser.add_argument("--batch_size", type=int, default=16, required=False)
parser.add_argument("--batch_size", type=int, default=64, required=False)
parser.add_argument("--num_workers", type=int, default=0, required=False)
parser.add_argument("--train_mode", type=str, default='regression', required=False)  # regression   classification 
args = parser.parse_args()

project_name = "train5_26" 
batchsize = args.batch_size
max_acc = 0
Has0_acc_2=0
Non0_acc_2=0
Has0_F1_score=0
F1_score_3=0
Acc_3=0

n_epochs=200
learning_rate = 1e-4

Mult_acc_2=0
Mult_acc_3=0
Mult_acc_5=0
F1_score_5=0
F1_score=0
MAE=0.99
Corr=0
def main():

    log_path = os.path.join(os.path.join("./log", project_name))
    print("log_path :", log_path)

    save_path = os.path.join("./checkpoint",  project_name)
    print("model_save_path :", save_path)
    model = MyMultimodal_Integrally(num_classs = 5,
                        visual_seq_len=50,
                        audio_seq_len=50,
                        batchsize=64,
                        trans_depth=2,
                        cross_depth=4
                        ).to(device)
    
    # exit(0)
    # model = MMT_semi(num_classs = 3 if args.train_mode == 'classification' else 1,
    #           visual_seq_len=50,
    #           audio_seq_len=50,
    #           cross_depth=4
    #          ).to(device)

    dataLoader = MMDataLoader(args)

    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    scheduler_warmup = get_scheduler(optimizer, n_epochs)
    #0.001 0.01 0.1 1.0 10 100
    #alpha
    # 0.001 46.83
    #  0.01 48.14
    #   0.1 46.60
    #     1  46.82
    #    10  46.39  
    
    #beta
    # 0.001 47.70
    # 0.01 48.14
    # 0.1  47.48
    #1.0  46.82
    #10 46.78
    loss_fn = MultimodalLoss(alpha=0.01, beta=1,delta=10, batch_size=batchsize, device=device)
    # if args.train_mode == 'classification':
    #     loss_fn = MultimodalLoss(alpha=0.01, beta=0.01, delta=1, batch_size=batchsize, device=device)
    #     loss_fn = MultimodalLoss(alpha=0.01, beta=0.01, delta=1, batch_size=batchsize, device=device)
    #     loss_fn_mixup = MultimodalLoss_Mixup(alpha=0.01, beta=0.01, delta=1, batch_size=batchsize, device=device)
    # else:
    #     loss_fn = MultimodalLoss_Reg(alpha=0.01, beta=0.01, delta=1, batch_size=batchsize, device=device) 

    writer = SummaryWriter(logdir=log_path)

    for epoch in range(1, n_epochs+1):
        train(model, dataLoader['train'], optimizer, loss_fn, epoch, writer)
        evaluate(model, dataLoader['test'], optimizer, loss_fn, epoch, writer, save_path)
        scheduler_warmup.step()
    writer.close()



def train(model, train_loader, optimizer, loss_fn, epoch, writer):

    train_pbar = tqdm(enumerate(train_loader))
    losses = AverageMeter()
    losses_cls = AverageMeter()
    losses_specific = AverageMeter()
    losses_invariant = AverageMeter()
    acc_specific = AverageMeter()
    acc_invariant = AverageMeter()
    acc_cls = AverageMeter()

    import random
    model.train()
    for cur_iter, sample in train_pbar:
        x_vision = sample['vision'].to(device)
        x_audio = sample['audio'].to(device)
        x_text = sample['text'].to(device)
        # if args.train_mode == 'classification':
        #     label = sample['labels']['M'].long().squeeze().to(device)
        # else:
        #     label = sample['labels']['M'].squeeze().to(device)
        label = sample['labels']['M']
        label = torch.where(label==0, torch.tensor(2.), label)
        label = torch.where(label==1.0, torch.tensor(4.), label)
        label = torch.where(label==-1.0, torch.tensor(0.), label)

        label = torch.where(label==-0.8, torch.tensor(0.), label)
        label = torch.where(label==-0.6, torch.tensor(1.), label)
        label = torch.where(label==-0.4, torch.tensor(1.), label)
        label = torch.where(label==-0.2, torch.tensor(1.), label)
        label = torch.where(label==0.6, torch.tensor(3.), label)
        label = torch.where(label==0.4, torch.tensor(3.), label)
        label = torch.where(label==0.2, torch.tensor(3.), label)
        label = torch.where(label==0.8, torch.tensor(4.), label)
        label = label.long().squeeze().to(device)

        x_vision, x_audio, x_text, lebel_a, lebel_b, lam = mixup_data(x_vision, x_audio, x_text, label, 1.0, device=device)
        x_vision, x_audio, x_text, lebel_a, lebel_b = map(Variable, (x_vision, x_audio, x_text, lebel_a, lebel_b))
        
        label = label.view(-1).long()
        # from torchstat import stat
        # # print(x_text.shape)
        # # exit(0)
        # from thop import profile
        # # input = torch.randn(64,20,709).to(device)
        # # input2= torch.randn(64,20,33).to(device)
        # # input3=torch.randn(64,20,768).to(device)
        # flops, params = profile(model, inputs=(input,input2,input3))
        # # print('FLOPs = ' + str(flops/1000**3) + 'G')
        # # print('Params = ' + str(params/1000**2) + 'M')

        # input_a = (64,20,709)
        # # input_a =(20,709)
        # # input_t=(20,768)
        # stat(model,input_a)
        # exit(0)
        x_invariant, x_specific_a, x_specific_v, x_specific_t, cls_output, x_visual, x_audio, x_text = model(x_vision, x_audio, x_text)
        # print(cls_output.shape)
        # exit(0)
        loss, orth_loss, sim_loss, cls_loss, acc_batch = loss_fn(
            x_invariant, x_specific_a, 
            x_specific_v, x_specific_t,
            x_visual, x_audio, x_text,
            cls_output, label, epoch)
        # print(loss)
        # exit(0)
        #D0_visual_op, D0_audio_op, D0_text_op, D2_visual_op, D2_audio_op, D2_text_op, cls_output,a,b,c = model(x_vision, x_audio, x_text)

        if args.train_mode == 'classification':
            label = label.view(-1).long()
        else:
            label = label.view(-1, 1)
        # print(loss)
        # exit(0)
            # cls_output = torch.clip(cls_output, min=-1., max=1.)

        # loss, specific_loss, invariant_loss, cls_loss, cls_acc, Specific_Cls_Acc, Invariant_Cls_Acc = loss_fn(
        #     D0_visual_op, D0_audio_op, D0_text_op,
        #     D2_visual_op, D2_audio_op, D2_text_op,
        #     cls_output, lebel_a, lebel_b, lam, epoch)
        # loss, specific_loss, invariant_loss, cls_loss, cls_acc, Specific_Cls_Acc, Invariant_Cls_Acc = loss_fn(
            # D0_visual_op, D0_audio_op, D0_text_op,
            # D2_visual_op, D2_audio_op, D2_text_op,
            # cls_output, label, epoch)
        # losses.update(loss.item(), batchsize)
        # losses_specific.update(specific_loss.item(), batchsize)
        # losses_invariant.update(invariant_loss.item(), batchsize)
        # losses_cls.update(cls_loss.item(), batchsize)
        # acc_specific.update(Specific_Cls_Acc, batchsize)
        # acc_invariant.update(Invariant_Cls_Acc, batchsize)
        # acc_cls.update(cls_acc, batchsize)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


        # train_pbar.set_description('train')
        # train_pbar.set_postfix({'epoch': '{}'.format(epoch),
        #                            'loss': '{:.5f}'.format(losses.value_avg),
        #                            'specific_loss': '{:.5f}'.format(losses_specific.value_avg),
        #                            'invariant_loss': '{:.5f}'.format(losses_invariant.value_avg),
        #                            'cls_loss': '{:.5f}'.format(losses_cls.value_avg),
        #                            'acc': '{:.5f}, {:.5f}, {:.5f}'.format(acc_cls.value_avg, acc_specific.value_avg,
        #                                                                   acc_invariant.value_avg),
        #                            'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])
        #                            })
    # lr_schduler.step()
    # warmup_scheduler.dampen()

    # writer.add_scalar('train/loss', losses.value_avg, epoch)
    # writer.add_scalar('train/specific_loss', losses_specific.value_avg, epoch)
    # writer.add_scalar('train/invariant_loss', losses_invariant.value_avg, epoch)
    # writer.add_scalar('train/cls_loss', losses_cls.value_avg, epoch)
    # writer.add_scalar('train/cls_acc', acc_cls.value_avg, epoch)
    # writer.add_scalar('train/Specific_Cls_Acc', acc_specific.value_avg, epoch)
    # writer.add_scalar('train/Invariant_Cls_Acc', acc_invariant.value_avg, epoch)


def evaluate(model, eval_loader, optimizer, loss_fn, epoch, writer, save_path):
    global max_acc,Has0_acc_2,Has0_F1_score,Non0_acc_2,Acc_3,Mult_acc_2,Mult_acc_3,Mult_acc_5,F1_score,MAE,Corr,F1_score_3,F1_score_5
    eval_pbar = tqdm(enumerate(eval_loader))
    losses = AverageMeter()
    losses_cls = AverageMeter()
    losses_specific = AverageMeter()
    losses_invariant = AverageMeter()
    acc_specific = AverageMeter()
    acc_invariant = AverageMeter()
    acc_cls = AverageMeter()
    y_pred = []
    y_true = []

    model.eval()
    
    for cur_iter, sample in eval_pbar:
        x_vision = sample['vision']
        x_audio = sample['audio']
        x_text = sample['text']
        label = sample['labels']['M']

        label = torch.where(label==0, torch.tensor(2.), label)
        label = torch.where(label==1.0, torch.tensor(4.), label)
        label = torch.where(label==-1.0, torch.tensor(0.), label)

        label = torch.where(label==-0.8, torch.tensor(0.), label)
        label = torch.where(label==-0.6, torch.tensor(1.), label)
        label = torch.where(label==-0.4, torch.tensor(1.), label)
        label = torch.where(label==-0.2, torch.tensor(1.), label)
        
        label = torch.where(label==0.6, torch.tensor(3.), label)
        label = torch.where(label==0.4, torch.tensor(3.), label)
        label = torch.where(label==0.2, torch.tensor(3.), label)
        label = torch.where(label==0.8, torch.tensor(4.), label)
        # if args.train_mode == 'classification':
        #     label = sample['labels']['M'].long().squeeze().to(device)
        # else:
        #     label = sample['labels']['M'].squeeze().to(device)
        # # label = torch.where(label>=0, torch.tensor(1.), label)
        # # label = torch.where(label<0, torch.tensor(0.), label)
        # # label = torch.where(label==0, torch.tensor(2.), label)
        # # label = label.squeeze().long().to(device)

        with torch.no_grad():
            x_invariant, x_specific_a, x_specific_v, x_specific_t, cls_output, x_visual, x_audio, x_text = model(x_vision.to(device), x_audio.to(device), x_text.to(device))
            #D0_visual_op, D0_audio_op, D0_text_op, D2_visual_op, D2_audio_op, D2_text_op, cls_output,a,b,c = model(x_vision.to(device), x_audio.to(device), x_text.to(device))
        label = label.view(-1).long()
        if args.train_mode == 'classification':
            label = label.view(-1).long()
        else:
            # cls_output = torch.clip(cls_output, min=-1., max=1.)
            label = label.view(-1, 1)

        y_pred.append(cls_output.cpu())
        y_true.append(label.cpu())

    #     loss, specific_loss, invariant_loss, cls_loss, cls_acc, Specific_Cls_Acc, Invariant_Cls_Acc = loss_fn(
    #         D0_visual_op, D0_audio_op, D0_text_op,
    #         D2_visual_op, D2_audio_op, D2_text_op,
    #         cls_output, label, epoch)

    #     losses.update(loss.item(), batchsize)
    #     losses_specific.update(specific_loss.item(), batchsize)
    #     losses_invariant.update(invariant_loss.item(), batchsize)
    #     losses_cls.update(cls_loss.item(), batchsize)
    #     acc_specific.update(Specific_Cls_Acc, batchsize)
    #     acc_invariant.update(Invariant_Cls_Acc, batchsize)
    #     acc_cls.update(cls_acc, batchsize)



    #     eval_pbar.set_description('eval')
    #     eval_pbar.set_postfix({'epoch': '{}'.format(epoch),
    #                             'loss': '{:.5f}'.format(losses.value_avg),
    #                             'specific_loss': '{:.5f}'.format(losses_specific.value_avg),
    #                             'invariant_loss': '{:.5f}'.format(losses_invariant.value_avg),
    #                             'cls_loss': '{:.5f}'.format(losses_cls.value_avg),
    #                             'acc': '{:.5f}, {:.5f}'.format(acc_cls.value_avg, max_acc),
    #                             'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])
    #                             })

    # writer.add_scalar('eval/loss', losses.value_avg, epoch)
    # writer.add_scalar('eval/specific_loss', losses_specific.value_avg, epoch)
    # writer.add_scalar('eval/invariant_loss', losses_invariant.value_avg, epoch)
    # writer.add_scalar('eval/cls_loss', losses_cls.value_avg, epoch)
    # writer.add_scalar('eval/cls_acc', acc_cls.value_avg, epoch)
    # writer.add_scalar('eval/Specific_Cls_Acc', acc_specific.value_avg, epoch)
    # writer.add_scalar('eval/Invariant_Cls_Acc', acc_invariant.value_avg, epoch)

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    eval_results = cal_acc5(pred, true)
    if eval_results['Mult_acc_5'] > Mult_acc_5:
        Mult_acc_5 = eval_results['Mult_acc_5']
        #save_model(save_path, epoch, model, optimizer, eval_results['Mult_acc_5'], eval_results['Mult_acc_5'], eval_results['F1_score_5'], 2)
    if eval_results['F1_score_5'] > F1_score_5:
        F1_score_5 = eval_results['F1_score_5']
    print("eval_results; Mult_acc_5: ", Mult_acc_5, "F1_score_5: ", F1_score_5)

if __name__ == '__main__':
    setup_seed(12345)
    main()
