import torchsummary
import torch
from torch import nn
from einops import rearrange
from models.vit import Transformer, CrossTransformer, ProjetTransformer
# from vit import Transformer, CrossTransformer
from models.bert import BertTextEncoder
import random
from torch.autograd import Function


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


class MyMultimodal_Integrally(nn.Module):
    def __init__(self, num_classs, visual_seq_len=8, audio_seq_len=8, batchsize=8, trans_depth=2, cross_depth=2):
        super(MyMultimodal_Integrally, self).__init__()

        self.visual_seq_len = visual_seq_len
        self.audio_seq_len = audio_seq_len

        self.bertmodel = BertTextEncoder(use_finetune=True, transformers='bert', pretrained='bert-base-uncased')

        # mosi
        # self.proj_l = nn.Linear(768, 256)
        # self.proj_a = nn.Linear(5, 256)
        # self.proj_v = nn.Linear(20, 256)

        # mosei
        self.proj_l = nn.Linear(768, 256)
        self.proj_a = nn.Linear(74, 256)
        self.proj_v = nn.Linear(35, 256)
        #ch-sims
        self.proj_l = nn.Linear(768, 256)
        self.proj_a = nn.Linear(33, 256)
        self.proj_v = nn.Linear(709, 256)
        
        self.project_layer = ProjetTransformer(num_frames=60, dim=256, depth=2, heads=8, mlp_dim=512, dim_head = 64, len_invariant = 6, len_specific = 6, dropout = 0.5, emb_dropout = 0.1)

        self.dropout = nn.Dropout(0.5)

        self.fusion_layer_1 = CrossTransformer(source_num_frames = 150,
                                                   tgt_num_frames = 150,
                                                   dim=256,
                                                   depth=cross_depth,
                                                   heads=8,
                                                   mlp_dim=256,
                                                   dropout=0.5,
                                                   emb_dropout=0.1
                                                   )

        self.fusion_layer_2 = CrossTransformer(source_num_frames = 150,
                                                   tgt_num_frames = 150,
                                                   dim=256,
                                                   depth=cross_depth,
                                                   heads=8,
                                                   mlp_dim=256,
                                                   dropout=0.5,
                                                   emb_dropout=0.1
                                                   )

        self.cls_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.Linear(128, num_classs)
        )


    def forward(self, x_visual, x_audio, x_text):

        # x_text = self.bertmodel(x_text)
        x_visual = self.dropout(x_visual)
        x_audio = self.dropout(x_audio)
        x_text = self.dropout(x_text)

        x_visual = self.proj_v(x_visual)
        x_audio = self.proj_a(x_audio)
        x_text = self.proj_l(x_text)

        x = torch.cat((x_visual, x_audio, x_text), dim=1)

        x = self.project_layer(x)
        # print(x.shape)
        # exit(0)
        x_invariant, x_specific_v, x_specific_a, x_specific_t = x[:, :6], x[:, 6:8], x[:, 8:10], x[:, 10:12]
        
        feat_specific = torch.cat((x_specific_v, x_specific_a, x_specific_t), dim=1)
        feat_1 = self.fusion_layer_1(x_invariant, feat_specific).mean(dim=1)
        feat_2 = self.fusion_layer_2(feat_specific, x_invariant).mean(dim=1)
        feat = feat_1 + feat_2

        cls_output = self.cls_head(feat)

        return x_invariant, x_specific_a, x_specific_v, x_specific_t, cls_output, x_visual, x_audio, x_text


def generate_model(opt):
    model = MyMultimodal_Integrally(num_classs = opt.num_classs,
                        visual_seq_len=opt.visual_seq_len,
                        audio_seq_len=opt.audio_seq_len,
                        batchsize=opt.batchsize,
                        trans_depth=opt.transformer_depth,
                        cross_depth=opt.crosstransformer_depth
                        )

    return model


if __name__ == '__main__':
    model = MyMultimodal_Integrally(num_classs = 6,
                        train_stage=4,
                        visual_seq_len=8,
                        audio_seq_len=8,
                        batchsize=8,
                        trans_depth=1,
                        cross_depth=1
                        )
    # print(model)暂时注释

    visual = torch.randn((8, 8, 3, 224,224))
    audio = torch.randn((8, 8, 3, 64,64))
    out = model(visual, audio)
    pass
