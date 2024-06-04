import torchsummary

import torch
from torch import nn
from einops import rearrange
from models.vit import Transformer, CrossTransformer
# from vit import Transformer, CrossTransformer
import random
from torch.autograd import Function
from subNets import BertTextEncoder

class GradReverse(torch.autograd.Function):

    # 重写父类方法的时候，最好添加默认参数，不然会有warning（为了好看。。）
    @ staticmethod
    def forward(ctx, x, constant):
        #　其实就是传入dict{'lambd' = lambd}
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 传入的是tuple，我们只需要第一个
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)
class MMT_semi(nn.Module):
    def __init__(self, num_classs, visual_seq_len=50, audio_seq_len=50, cross_depth=1):
        super(MMT_semi, self).__init__()

        self.visual_seq_len = visual_seq_len
        self.audio_seq_len = audio_seq_len

        # self.proj_l = nn.Conv1d(300, 256, kernel_size=1, padding=(1-1)//2, bias=False)
        # self.proj_a = nn.Conv1d(5, 256, kernel_size=5, padding=(5-1)//2, bias=False)
        # self.proj_v = nn.Conv1d(20, 256, kernel_size=3, padding=(3-1)//2, bias=False)

        self.proj_l = nn.Linear(768, 256)
        self.proj_a = nn.Linear(33, 256)
        self.proj_v = nn.Linear(709, 256)
        #v2.0
        # self.proj_l = nn.Linear(768, 256)
        # self.proj_a = nn.Linear(25, 256)
        # self.proj_v = nn.Linear(177, 256)
        
        self.proj_l2 = nn.Linear(768, 256)
        self.proj_a2 = nn.Linear(25, 256)
        self.proj_v2 = nn.Linear(177, 256)
        self.bertmodel = BertTextEncoder(use_finetune=False, transformers='bert', pretrained='bert-base-uncased')
        self.dropout = nn.Dropout(0.5)
        self.specific_projection_v = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128)
        )


        self.specific_projection_a = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128)
        )

        self.specific_projection_l = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128)
        )

        self.invariant_projection = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128)
        )

        self.D0 = nn.Sequential(
            nn.Linear(128, 3)
        )


        self.common_fusion_layer = nn.Sequential(
                                                nn.Linear(384, 128),
                                                nn.LeakyReLU(0.1),
                                                nn.Linear(128, 128)
                                                )

        self.compensation_layer = CrossTransformer(source_num_frames = 150,
                                                tgt_num_frames = 50,
                                                dim=128,
                                                depth=cross_depth,
                                                heads=8,
                                                mlp_dim=256,
                                                dropout=0.5,
                                                emb_dropout=0.1
                                                )
        self.audio_cls=nn.Sequential(
            # nn.BatchNorm1d(128),
            # nn.Linear(128, 128),
            # nn.LeakyReLU(),
            # nn.Dropout(0.5),
            nn.Linear(128, num_classs)
        )
        self.vision_cls=nn.Sequential(
            # nn.BatchNorm1d(128),
            # nn.Linear(128, 128),
            # nn.LeakyReLU(),
            # nn.Dropout(0.5),
            nn.Linear(128, num_classs)
        )
        self.text_cls=nn.Sequential(
            # nn.BatchNorm1d(128),
            # nn.Linear(128, 128),
            # nn.LeakyReLU(),
            # nn.Dropout(0.5),
            nn.Linear(128, num_classs)
        )
        self.cls_head = nn.Sequential(
            # nn.BatchNorm1d(128),
            # nn.Linear(128, 128),
            # nn.LeakyReLU(),
            # nn.Dropout(0.5),
            nn.Linear(128, num_classs)
        )


    def forward(self, x_vision, x_audio, x_text,num=0):
        # x_vision = sample['vision']
        # x_audio = sample['audio']
        # x_text = sample['text']

        # x_vision = x_vision.permute(0, 2, 1)
        # x_audio = x_audio.permute(0,2,1)
        # x_text = x_text.permute(0,2,1)
    
        # x_vision = self.proj_v(x_vision).permute(0, 2, 1)
        # x_audio = self.proj_a(x_audio).permute(0, 2, 1)
        # x_text = self.proj_l(x_text).permute(0, 2, 1)
        # print(x_audio.shape)
        # print(x_vision.shape)
        # print(x_text.shape)
        # exit(0)
        if num==0:
            x_vision = self.proj_v(x_vision)
            x_audio = self.proj_a(x_audio)
            x_text = self.proj_l(x_text)
        else:
            x_text = self.bertmodel(x_text)
            x_vision = self.proj_v2(x_vision)
            x_audio = self.proj_a2(x_audio)
            x_text = self.proj_l2(x_text)


        x_vision = self.dropout(x_vision)
        x_audio = self.dropout(x_audio)
        x_text = self.dropout(x_text)

        x_vision_D = x_vision.detach()
        x_audio_D = x_audio.detach()
        x_text_D = x_text.detach()

        # x_vision_D = x_vision
        # x_audio_D = x_audio
        # x_text_D = x_text

        x_vision_specific_D = self.specific_projection_v(x_vision_D)
        x_audio_specific_D = self.specific_projection_a(x_audio_D)
        x_text_specific_D = self.specific_projection_l(x_text_D)
        x_vision_invariant_D = self.invariant_projection(x_vision_D)
        x_audio_invariant_D = self.invariant_projection(x_audio_D)
        x_text_invariant_D = self.invariant_projection(x_text_D)

        x_vision_invariant_D0 = x_vision_invariant_D.mean(dim=1)
        x_vision_invariant_D0 = GradReverse.apply(x_vision_invariant_D0, 1.0)
        x_audio_invariant_D0 = x_audio_invariant_D.mean(dim=1)
        x_audio_invariant_D0 = GradReverse.apply(x_audio_invariant_D0, 1.0)
        x_text_invariant_D0 = x_text_invariant_D.mean(dim=1)
        x_text_invariant_D0 = GradReverse.apply(x_text_invariant_D0, 1.0)


        D0_visual_op = self.D0(x_vision_invariant_D0)
        D0_audio_op = self.D0(x_audio_invariant_D0)
        D0_text_op = self.D0(x_text_invariant_D0)

        D2_visual_op = self.D0(x_vision_specific_D.mean(dim=1))
        D2_audio_op = self.D0(x_audio_specific_D.mean(dim=1))
        D2_text_op = self.D0(x_text_specific_D.mean(dim=1))

        x_visual_specific = self.specific_projection_v(x_vision)
        x_audio_specific = self.specific_projection_a(x_audio)
        x_text_specific = self.specific_projection_l(x_text)
        x_visual_invariant = self.invariant_projection(x_vision)
        x_audio_invariant = self.invariant_projection(x_audio)
        x_text_invariant = self.invariant_projection(x_text)

        feat_common = torch.cat((x_visual_invariant, x_audio_invariant, x_text_invariant), dim=2)
        feat_common = self.common_fusion_layer(feat_common)

        feat_exclusive = torch.cat((x_visual_specific, x_audio_specific, x_text_specific), dim=1)
        feat = self.compensation_layer(feat_exclusive, feat_common).mean(dim=1)

        # feat_common = torch.cat((x_visual_invariant, x_audio_invariant), dim=2)
        # feat_common = self.common_fusion_layer(feat_common)

        # feat_exclusive = torch.cat((x_visual_specific, x_audio_specific), dim=1)
        # feat = self.compensation_layer(feat_exclusive, feat_common).mean(dim=1)
        visual_final=self.vision_cls(x_visual_specific.mean(dim=1))
        audio_final=self.audio_cls(x_audio_specific.mean(dim=1))
        text_final=self.text_cls(x_text_specific.mean(dim=1))

        cls_output = self.cls_head(feat)


        # return D0_visual_op, D0_audio_op,D0_text_op, D2_visual_op, D2_audio_op, D2_text_op, cls_output,visual_final,audio_final,text_final


        return D0_visual_op, D0_audio_op,D0_text_op, D2_visual_op, D2_audio_op, D2_text_op,cls_output,visual_final,audio_final,text_final

class MMT(nn.Module):
    def __init__(self, num_classs, visual_seq_len=50, audio_seq_len=50, cross_depth=1):
        super(MMT, self).__init__()

        self.visual_seq_len = visual_seq_len
        self.audio_seq_len = audio_seq_len

        # self.proj_l = nn.Conv1d(300, 256, kernel_size=1, padding=(1-1)//2, bias=False)
        # self.proj_a = nn.Conv1d(5, 256, kernel_size=5, padding=(5-1)//2, bias=False)
        # self.proj_v = nn.Conv1d(20, 256, kernel_size=3, padding=(3-1)//2, bias=False)

        self.proj_l = nn.Linear(768, 256)
        self.proj_a = nn.Linear(33, 256)
        self.proj_v = nn.Linear(709, 256)

        self.dropout = nn.Dropout(0.5)
        self.specific_projection_v = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128)
        )


        self.specific_projection_a = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128)
        )

        self.specific_projection_l = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128)
        )

        self.invariant_projection = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128)
        )

        self.D0 = nn.Sequential(
            nn.Linear(128, 3)
        )


        self.common_fusion_layer = nn.Sequential(
                                                nn.Linear(384, 128),
                                                nn.LeakyReLU(0.1),
                                                nn.Linear(128, 128)
                                                )

        self.compensation_layer = CrossTransformer(source_num_frames = 150,
                                                tgt_num_frames = 50,
                                                dim=128,
                                                depth=cross_depth,
                                                heads=8,
                                                mlp_dim=256,
                                                dropout=0.5,
                                                emb_dropout=0.1
                                                )

        self.cls_head = nn.Sequential(
            # nn.BatchNorm1d(128),
            # nn.Linear(128, 128),
            # nn.LeakyReLU(),
            # nn.Dropout(0.5),
            nn.Linear(128, num_classs)
        )


    def forward(self, x_vision, x_audio, x_text):
        # x_vision = sample['vision']
        # x_audio = sample['audio']
        # x_text = sample['text']

        # x_vision = x_vision.permute(0, 2, 1)
        # x_audio = x_audio.permute(0,2,1)
        # x_text = x_text.permute(0,2,1)
    
        # x_vision = self.proj_v(x_vision).permute(0, 2, 1)
        # x_audio = self.proj_a(x_audio).permute(0, 2, 1)
        # x_text = self.proj_l(x_text).permute(0, 2, 1)

        # print(x_audio.shape)
        # exit(0)
        x_vision = self.proj_v(x_vision)
        x_audio = self.proj_a(x_audio)
        x_text = self.proj_l(x_text)

        x_vision = self.dropout(x_vision)
        x_audio = self.dropout(x_audio)
        x_text = self.dropout(x_text)

        x_vision_D = x_vision.detach()
        x_audio_D = x_audio.detach()
        x_text_D = x_text.detach()

        # x_vision_D = x_vision
        # x_audio_D = x_audio
        # x_text_D = x_text

        x_vision_specific_D = self.specific_projection_v(x_vision_D)
        x_audio_specific_D = self.specific_projection_a(x_audio_D)
        x_text_specific_D = self.specific_projection_l(x_text_D)
        x_vision_invariant_D = self.invariant_projection(x_vision_D)
        x_audio_invariant_D = self.invariant_projection(x_audio_D)
        x_text_invariant_D = self.invariant_projection(x_text_D)

        x_vision_invariant_D0 = x_vision_invariant_D.mean(dim=1)
        x_vision_invariant_D0 = GradReverse.apply(x_vision_invariant_D0, 1.0)
        x_audio_invariant_D0 = x_audio_invariant_D.mean(dim=1)
        x_audio_invariant_D0 = GradReverse.apply(x_audio_invariant_D0, 1.0)
        x_text_invariant_D0 = x_text_invariant_D.mean(dim=1)
        x_text_invariant_D0 = GradReverse.apply(x_text_invariant_D0, 1.0)


        D0_visual_op = self.D0(x_vision_invariant_D0)
        D0_audio_op = self.D0(x_audio_invariant_D0)
        D0_text_op = self.D0(x_text_invariant_D0)

        D2_visual_op = self.D0(x_vision_specific_D.mean(dim=1))
        D2_audio_op = self.D0(x_audio_specific_D.mean(dim=1))
        D2_text_op = self.D0(x_text_specific_D.mean(dim=1))

        x_visual_specific = self.specific_projection_v(x_vision)
        x_audio_specific = self.specific_projection_a(x_audio)
        x_text_specific = self.specific_projection_l(x_text)
        x_visual_invariant = self.invariant_projection(x_vision)
        x_audio_invariant = self.invariant_projection(x_audio)
        x_text_invariant = self.invariant_projection(x_text)

        feat_common = torch.cat((x_visual_invariant, x_audio_invariant, x_text_invariant), dim=2)
        feat_common = self.common_fusion_layer(feat_common)

        feat_exclusive = torch.cat((x_visual_specific, x_audio_specific, x_text_specific), dim=1)
        feat = self.compensation_layer(feat_exclusive, feat_common).mean(dim=1)

        # feat_common = torch.cat((x_visual_invariant, x_audio_invariant), dim=2)
        # feat_common = self.common_fusion_layer(feat_common)

        # feat_exclusive = torch.cat((x_visual_specific, x_audio_specific), dim=1)
        # feat = self.compensation_layer(feat_exclusive, feat_common).mean(dim=1)

        cls_output = self.cls_head(feat)

        return D0_visual_op, D0_audio_op,D0_text_op, D2_visual_op, D2_audio_op, D2_text_op, cls_output



class MMT_Test(nn.Module):
    def __init__(self, num_classs, visual_seq_len=50, audio_seq_len=50, cross_depth=1):
        super(MMT_Test, self).__init__()

        self.visual_seq_len = visual_seq_len
        self.audio_seq_len = audio_seq_len

        # self.proj_l = nn.Conv1d(300, 256, kernel_size=1, padding=(1-1)//2, bias=False)
        # self.proj_a = nn.Conv1d(5, 256, kernel_size=5, padding=(5-1)//2, bias=False)
        # self.proj_v = nn.Conv1d(20, 256, kernel_size=3, padding=(3-1)//2, bias=False)

        self.proj_l = nn.Linear(768, 256)
        self.proj_a = nn.Linear(33, 256)
        self.proj_v = nn.Linear(709, 256)

        self.dropout = nn.Dropout(0.5)
        self.specific_projection_v = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128)
        )


        self.specific_projection_a = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128)
        )

        self.specific_projection_l = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128)
        )

        self.invariant_projection = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128)
        )

        self.D0 = nn.Sequential(
            nn.Linear(128, 3),
            # nn.LeakyReLU(0.1),
            # nn.Linear(128, 3)
        )


        self.common_fusion_layer = nn.Sequential(
                                                nn.Linear(384, 128),
                                                nn.LeakyReLU(0.1),
                                                nn.Linear(128, 128)
                                                )

        self.compensation_layer = CrossTransformer(source_num_frames = 150,
                                                tgt_num_frames = 50,
                                                dim=128,
                                                depth=cross_depth,
                                                heads=8,
                                                mlp_dim=256,
                                                dropout=0.5,
                                                emb_dropout=0.1
                                                )

        self.cls_head = nn.Sequential(
            # nn.BatchNorm1d(128),
            # nn.Linear(128, 128),
            # nn.LeakyReLU(),
            # nn.Dropout(0.5),
            nn.Linear(128, num_classs)
        )


    def forward(self, x_vision, x_audio, x_text):
        # x_vision = sample['vision']
        # x_audio = sample['audio']
        # x_text = sample['text']

        # x_vision = x_vision.permute(0, 2, 1)
        # x_audio = x_audio.permute(0,2,1)
        # x_text = x_text.permute(0,2,1)
    
        # x_vision = self.proj_v(x_vision).permute(0, 2, 1)
        # x_audio = self.proj_a(x_audio).permute(0, 2, 1)
        # x_text = self.proj_l(x_text).permute(0, 2, 1)

        x_vision = self.proj_v(x_vision)
        x_audio = self.proj_a(x_audio)
        x_text = self.proj_l(x_text)

        x_vision = self.dropout(x_vision)
        x_audio = self.dropout(x_audio)
        x_text = self.dropout(x_text)

        x_vision_D = x_vision.detach()
        x_audio_D = x_audio.detach()
        x_text_D = x_text.detach()

        x_vision_specific_D = self.specific_projection_v(x_vision_D)
        x_audio_specific_D = self.specific_projection_a(x_audio_D)
        x_text_specific_D = self.specific_projection_l(x_text_D)
        x_vision_invariant_D = self.invariant_projection(x_vision_D)
        x_audio_invariant_D = self.invariant_projection(x_audio_D)
        x_text_invariant_D = self.invariant_projection(x_text_D)

        x_vision_invariant_D0 = x_vision_invariant_D.mean(dim=1)
        x_vision_invariant_D0 = GradReverse.apply(x_vision_invariant_D0, 1.0)
        x_audio_invariant_D0 = x_audio_invariant_D.mean(dim=1)
        x_audio_invariant_D0 = GradReverse.apply(x_audio_invariant_D0, 1.0)
        x_text_invariant_D0 = x_text_invariant_D.mean(dim=1)
        x_text_invariant_D0 = GradReverse.apply(x_text_invariant_D0, 1.0)


        D0_visual_op = self.D0(x_vision_invariant_D0)
        D0_audio_op = self.D0(x_audio_invariant_D0)
        D0_text_op = self.D0(x_text_invariant_D0)

        D2_visual_op = self.D0(x_vision_specific_D.mean(dim=1))
        D2_audio_op = self.D0(x_audio_specific_D.mean(dim=1))
        D2_text_op = self.D0(x_text_specific_D.mean(dim=1))

        x_visual_specific = self.specific_projection_v(x_vision)
        x_audio_specific = self.specific_projection_a(x_audio)
        x_text_specific = self.specific_projection_l(x_text)
        x_visual_invariant = self.invariant_projection(x_vision)
        x_audio_invariant = self.invariant_projection(x_audio)
        x_text_invariant = self.invariant_projection(x_text)

        feat_common = torch.cat((x_visual_invariant, x_audio_invariant, x_text_invariant), dim=2)
        feat_common = self.common_fusion_layer(feat_common)
        # feat_common = x_visual_invariant + x_audio_invariant

        feat_exclusive = torch.cat((x_visual_specific, x_audio_specific, x_text_specific), dim=1)
        # feat_exclusive = self.LayerNorm1(feat_exclusive)

        feat = self.compensation_layer(feat_exclusive, feat_common).mean(dim=1)

        cls_output = self.cls_head(feat)

        return D0_visual_op, D0_audio_op, D0_text_op, D2_visual_op, D2_audio_op, D2_text_op, cls_output, feat, x_visual_specific, x_audio_specific, x_text_specific, x_visual_invariant, x_audio_invariant, x_text_invariant, x_vision, x_audio, x_text



class MMT_Ablation(nn.Module):
    def __init__(self, num_classs, visual_seq_len=50, audio_seq_len=50, cross_depth=1):
        super(MMT_Ablation, self).__init__()

        self.visual_seq_len = visual_seq_len
        self.audio_seq_len = audio_seq_len

        # self.proj_l = nn.Conv1d(300, 256, kernel_size=1, padding=(1-1)//2, bias=False)
        # self.proj_a = nn.Conv1d(5, 256, kernel_size=5, padding=(5-1)//2, bias=False)
        # self.proj_v = nn.Conv1d(20, 256, kernel_size=3, padding=(3-1)//2, bias=False)

        self.proj_l = nn.Linear(768, 256)
        self.proj_a = nn.Linear(33, 256)
        self.proj_v = nn.Linear(709, 256)

        self.dropout = nn.Dropout(0.5)

        self.compensation_layer = Transformer(num_frames=50,
                                                num_classes=num_classs,
                                                dim=768,
                                                depth=cross_depth,
                                                heads=8,
                                                mlp_dim=768,
                                                dropout=0.5,
                                                emb_dropout=0.1
                                                )

        self.cls_head = nn.Sequential(
            # nn.BatchNorm1d(128),
            # nn.Linear(128, 128),
            # nn.LeakyReLU(),
            # nn.Dropout(0.5),
            nn.Linear(128, num_classs)
        )


    def forward(self, x_vision, x_audio, x_text):
        # x_vision = sample['vision']
        # x_audio = sample['audio']
        # x_text = sample['text']

        # x_vision = x_vision.permute(0, 2, 1)
        # x_audio = x_audio.permute(0,2,1)
        # x_text = x_text.permute(0,2,1)
    
        # x_vision = self.proj_v(x_vision).permute(0, 2, 1)
        # x_audio = self.proj_a(x_audio).permute(0, 2, 1)
        # x_text = self.proj_l(x_text).permute(0, 2, 1)

        x_vision = self.proj_v(x_vision)
        x_audio = self.proj_a(x_audio)
        x_text = self.proj_l(x_text)

        x_vision = self.dropout(x_vision)
        x_audio = self.dropout(x_audio)
        x_text = self.dropout(x_text)

        x_vision_D = x_vision.detach()
        x_audio_D = x_audio.detach()
        x_text_D = x_text.detach()

        x_vision_specific_D = self.specific_projection_v(x_vision_D)
        x_audio_specific_D = self.specific_projection_a(x_audio_D)
        x_text_specific_D = self.specific_projection_l(x_text_D)
        x_vision_invariant_D = self.invariant_projection(x_vision_D)
        x_audio_invariant_D = self.invariant_projection(x_audio_D)
        x_text_invariant_D = self.invariant_projection(x_text_D)

        x_vision_invariant_D0 = x_vision_invariant_D.mean(dim=1)
        x_vision_invariant_D0 = GradReverse.apply(x_vision_invariant_D0, 1.0)
        x_audio_invariant_D0 = x_audio_invariant_D.mean(dim=1)
        x_audio_invariant_D0 = GradReverse.apply(x_audio_invariant_D0, 1.0)
        x_text_invariant_D0 = x_text_invariant_D.mean(dim=1)
        x_text_invariant_D0 = GradReverse.apply(x_text_invariant_D0, 1.0)


        D0_visual_op = self.D0(x_vision_invariant_D0)
        D0_audio_op = self.D0(x_audio_invariant_D0)
        D0_text_op = self.D0(x_text_invariant_D0)

        D2_visual_op = self.D0(x_vision_specific_D.mean(dim=1))
        D2_audio_op = self.D0(x_audio_specific_D.mean(dim=1))
        D2_text_op = self.D0(x_text_specific_D.mean(dim=1))

        x_visual_specific = self.specific_projection_v(x_vision)
        x_audio_specific = self.specific_projection_a(x_audio)
        x_text_specific = self.specific_projection_l(x_text)
        x_visual_invariant = self.invariant_projection(x_vision)
        x_audio_invariant = self.invariant_projection(x_audio)
        x_text_invariant = self.invariant_projection(x_text)

        feat_common = torch.cat((x_visual_invariant, x_audio_invariant, x_text_invariant), dim=2)
        feat_common = self.common_fusion_layer(feat_common)
        # feat_common = x_visual_invariant + x_audio_invariant

        feat_exclusive = torch.cat((x_visual_specific, x_audio_specific, x_text_specific), dim=1)
        # feat_exclusive = self.LayerNorm1(feat_exclusive)

        feat = self.compensation_layer(feat_exclusive, feat_common).mean(dim=1)

        cls_output = self.cls_head(feat)

        return D0_visual_op, D0_audio_op,D0_text_op, D2_visual_op, D2_audio_op, D2_text_op, cls_output



def Generate_MMT(opt):
    model = MMT(num_classs = opt.num_classs,
                visual_seq_len=opt.visual_seq_len,
                audio_seq_len=opt.audio_seq_len,
                trans_depth=opt.transformer_depth,
                cross_depth=opt.crosstransformer_depth
                )

    return model




if __name__ == '__main__':
    model = MMT(num_classs = 6,
                        visual_seq_len=50,
                        audio_seq_len=50,
                        trans_depth=2,
                        cross_depth=2
                        )
    print(model)
