import os
import torch
import numpy as np
from tqdm import tqdm
from core.dataset_new import MMDataLoader
from core.losses import MultimodalLoss, MultimodalLoss_Reg
from core.optimizer import get_optimizer
from core.scheduler import get_scheduler
from core.utils import AverageMeter, save_model, setup_seed, save_model_reg
from tensorboardX import SummaryWriter
from models.MyModel_Ablation import MMT_Test
import argparse
from core.metric import MetricsTop
import pandas as pd


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(device)


parser = argparse.ArgumentParser()
# dataset
parser.add_argument("--datasetName", type=str, default="sims", required=False)
parser.add_argument("--dataPath", type=str, default=r"G:\datasets\MOSI\unaligned_39.pkl", required=False)
parser.add_argument("--use_bert", type=bool, default=False, required=False)
parser.add_argument("--need_data_aligned", type=bool, default=True, required=False)
parser.add_argument("--need_truncated", type=bool, default=True, required=False)
parser.add_argument("--data_missing", type=bool, default=False, required=False)
parser.add_argument("--seq_lens", type=tuple, default=[30,30,30], required=False)
parser.add_argument("--batch_size", type=int, default=1, required=False)
parser.add_argument("--num_workers", type=int, default=0, required=False)
parser.add_argument("--train_mode", type=str, default='classification', required=False)  # regression    
args = parser.parse_args()

project_name = "CodeTest0318" 
batchsize = 1
max_acc = 0
Has0_acc_2=0
Non0_acc_2=0
Has0_F1_score=0

Acc_3=0

n_epochs=200
learning_rate = 1e-4

Mult_acc_2=0
Mult_acc_3=0
Mult_acc_5=0
F1_score=0
MAE=0.99
Corr=0
F1_score_3=0


def save_excel(fea,file_name):
    data_df = pd.DataFrame(fea)
    writer = pd.ExcelWriter(file_name)
    data_df.to_excel(writer,'page_1')
    writer.save()


def main():

    model = MMT_Test(num_classs = 3,
              visual_seq_len=50,
              audio_seq_len=50,
              cross_depth=4
             ).to(device)
    checkpoint = torch.load(r'checkpoint\CodeTest0524_acc2\save_28_0.7812_0.7812_0.6805_d2.pth')['state_dict']
    model.load_state_dict(checkpoint)

    dataLoader = MMDataLoader(args)

    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    if args.train_mode == 'classification':
        loss_fn = MultimodalLoss(alpha=0.01, beta=0.01, delta=1, batch_size=batchsize, device=device)
    else:
        loss_fn = MultimodalLoss_Reg(alpha=0.01, beta=0.01, delta=1, batch_size=batchsize, device=device) 

    evaluate(model, dataLoader['test'], optimizer, loss_fn, 1)


def evaluate(model, eval_loader, optimizer, loss_fn, epoch):
    global max_acc,Has0_acc_2,Has0_F1_score,Non0_acc_2,Acc_3,Mult_acc_2,Mult_acc_3,Mult_acc_5,F1_score,MAE,Corr,F1_score_3
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

    label_list = []
    pred_list = []
    feature_list = []
    all_feature_list= []
    all_label = []

    model.eval()
    
    for cur_iter, sample in eval_pbar:
        x_vision = sample['vision']
        x_audio = sample['audio']
        x_text = sample['text']
        if args.train_mode == 'classification':
            label = sample['labels']['M'].long().squeeze().to(device)
        else:
            label = sample['labels']['M'].squeeze().to(device)

        with torch.no_grad():
            D0_visual_op, D0_audio_op, D0_text_op, D2_visual_op, D2_audio_op, D2_text_op, cls_output, feat, \
            x_visual_specific, x_audio_specific, x_text_specific, x_visual_invariant, x_audio_invariant, x_text_invariant, x_vision, x_audio, x_text = model(x_vision.to(device), x_audio.to(device), x_text.to(device))
        
        label_list.append(label.item())
        pred_list.append(torch.max(cls_output, dim=1)[1].item())
        feature_list.append(feat.squeeze().cpu().numpy())

        # all_feature_list.append(x_visual_specific.mean(1).squeeze(0).cpu().numpy())
        # all_label.append(0)
        # all_feature_list.append(x_audio_specific.mean(1).squeeze(0).cpu().numpy())
        # all_label.append(1)
        # all_feature_list.append(x_visual_invariant.mean(1).squeeze(0).cpu().numpy())
        # all_label.append(2)
        # all_feature_list.append(x_audio_invariant.mean(1).squeeze(0).cpu().numpy())
        # all_label.append(3)
        # all_feature_list.append(x_text_specific.mean(1).squeeze(0).cpu().numpy())
        # all_label.append(4)
        # all_feature_list.append(x_text_invariant.mean(1).squeeze(0).cpu().numpy())
        # all_label.append(5)

        # all_feature_list.append(x_vision.mean(1).squeeze(0).cpu().numpy())
        # all_label.append(0)
        # all_feature_list.append(x_audio.mean(1).squeeze(0).cpu().numpy())
        # all_label.append(1)
        # all_feature_list.append(x_text.mean(1).squeeze(0).cpu().numpy())
        # all_label.append(4)

        for f in range(x_visual_specific.shape[1]):
            all_feature_list.append(x_visual_specific[:,f].squeeze(0).cpu().numpy())
            all_label.append(0)
        for f in range(x_audio_specific.shape[1]):
            all_feature_list.append(x_audio_specific[:,f].squeeze(0).cpu().numpy())
            all_label.append(1)
        for f in range(x_visual_invariant.shape[1]):
            all_feature_list.append(x_visual_invariant[:,f].squeeze(0).cpu().numpy())
            all_label.append(2)
        for f in range(x_audio_invariant.shape[1]):
            all_feature_list.append(x_audio_invariant[:,f].squeeze(0).cpu().numpy())
            all_label.append(3)
        for f in range(x_text_specific.shape[1]):
            all_feature_list.append(x_visual_invariant[:,f].squeeze(0).cpu().numpy())
            all_label.append(4)
        for f in range(x_text_invariant.shape[1]):
            all_feature_list.append(x_audio_invariant[:,f].squeeze(0).cpu().numpy())
            all_label.append(5)

        if args.train_mode == 'classification':
            label = label.view(-1).long()
        else:
            label = label.view(-1, 1)
        y_pred.append(cls_output.cpu())
        y_true.append(label.cpu())

        loss, specific_loss, invariant_loss, cls_loss, cls_acc, Specific_Cls_Acc, Invariant_Cls_Acc = loss_fn(
            D0_visual_op, D0_audio_op, D0_text_op,
            D2_visual_op, D2_audio_op, D2_text_op,
            cls_output, label, epoch)

        losses.update(loss.item(), batchsize)
        losses_specific.update(specific_loss.item(), batchsize)
        losses_invariant.update(invariant_loss.item(), batchsize)
        losses_cls.update(cls_loss.item(), batchsize)
        acc_specific.update(Specific_Cls_Acc, batchsize)
        acc_invariant.update(Invariant_Cls_Acc, batchsize)
        acc_cls.update(cls_acc, batchsize)



        eval_pbar.set_description('eval')
        eval_pbar.set_postfix({'epoch': '{}'.format(epoch),
                                'loss': '{:.5f}'.format(losses.value_avg),
                                'specific_loss': '{:.5f}'.format(losses_specific.value_avg),
                                'invariant_loss': '{:.5f}'.format(losses_invariant.value_avg),
                                'cls_loss': '{:.5f}'.format(losses_cls.value_avg),
                                'acc': '{:.5f}, {:.5f}'.format(acc_cls.value_avg, max_acc),
                                'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])
                                })

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    eval_results = MetricsTop(args.train_mode).getMetics('SIMS')(pred, true)
    
    if args.train_mode == 'classification':
        # if acc_cls.value_avg > max_acc:
        #     max_acc = acc_cls.value_avg 
        # if eval_results['Has0_acc_2'] > Has0_acc_2:
        #     Has0_acc_2 = eval_results['Has0_acc_2']
        # if eval_results['Non0_acc_2'] > Non0_acc_2:
        #     Non0_acc_2 = eval_results['Non0_acc_2']
        # if eval_results['Acc_3'] > Acc_3:
        #     Acc_3 = eval_results['Acc_3']
        # if eval_results['Has0_F1_score'] > Has0_F1_score:
        #     Has0_F1_score = eval_results['Has0_F1_score']
        # if eval_results['F1_score_3'] > F1_score_3:
        #     F1_score_3 = eval_results['F1_score_3']

        if acc_cls.value_avg > max_acc:
            max_acc = acc_cls.value_avg 
            Has0_acc_2 = eval_results['Has0_acc_2']
            Non0_acc_2 = eval_results['Non0_acc_2']
            Acc_3 = eval_results['Acc_3']
            Has0_F1_score = eval_results['Has0_F1_score']
            F1_score_3 = eval_results['F1_score_3']

        print("eval_results; Has0_acc_2: ", Has0_acc_2, "Non0_acc_2: ", Non0_acc_2, "Has0_F1_score:", Has0_F1_score, "Acc_3: ", Acc_3, "F1_score_3:", F1_score_3)
    else:
        if eval_results['Mult_acc_2'] > Mult_acc_2:
            Mult_acc_2 = eval_results['Mult_acc_2']
        if eval_results['Mult_acc_3'] > Mult_acc_3:
            Mult_acc_3 = eval_results['Mult_acc_3']
        if eval_results['Mult_acc_5'] > Mult_acc_5:
            Mult_acc_5 = eval_results['Mult_acc_5']
        if eval_results['F1_score'] > F1_score:
            F1_score = eval_results['F1_score']
        if eval_results['MAE'] < MAE:
            MAE = eval_results['MAE']
        if eval_results['Corr'] > Corr:
            Corr = eval_results['Corr']

        print("eval_results; Mult_acc_2: ", Mult_acc_2, "Mult_acc_3: ", Mult_acc_3, "Mult_acc_5:", Mult_acc_5, "F1_score: ", F1_score, "MAE: ", MAE, "Corr: ", Corr)

    # path1 = "./output/confusion_mat/{}".format(args.datasetName)
    # if not os.path.exists(path1):
    #     os.makedirs(path1)
    # path2 = "./output/confusion_mat/{}".format(args.datasetName)
    # if not os.path.exists(path2):
    #     os.makedirs(path2)
    # with open(os.path.join(path1, 'label_list.txt'), 'w+') as f:
    #     f.write(str(label_list))
    # with open(os.path.join(path1, 'pred_list.txt'), 'w+') as f:
    #     f.write(str(pred_list))

    # path3 = './output/tsne/{}'.format(args.datasetName) 
    # if not os.path.exists(path3):
    #     os.makedirs(path3)
    # save_excel(feature_list, os.path.join(path3, 'test_feat.xlsx'))
    # save_excel(label_list, os.path.join(path3, 'test_label.xlsx'))
    # save_excel(all_feature_list, os.path.join(path3, 'all_feat.xlsx'))
    # save_excel(all_label, os.path.join(path3, 'all_label.xlsx'))

    save_excel(all_feature_list, './output/tsne/{}//all_feat_acc2.xlsx'.format(args.datasetName))
    save_excel(all_label, './output/tsne/{}/all_label_acc2.xlsx'.format(args.datasetName))

if __name__ == '__main__':
    # setup_seed(12345)
    main()
