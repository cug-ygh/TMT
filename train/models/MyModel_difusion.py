import torchsummary
import torch
from torch import nn
from einops import rearrange
from models.vit import Transformer, CrossTransformer, ProjetTransformer
# from vit import Transformer, CrossTransformer
from models.bert import BertTextEncoder
import random
from torch.autograd import Function
class LMF(nn.Module):
    '''
    Low-rank Multimodal Fusion
    '''

    def __init__(self):
        '''
        Args:
            input_dims - a length-3 tuple, contains (audio_dim, video_dim, text_dim)
            hidden_dims - another length-3 tuple, hidden dims of the sub-networks
            text_out - int, specifying the resulting dimensions of the text subnetwork
            dropouts - a length-4 tuple, contains (audio_dropout, video_dropout, text_dropout, post_fusion_dropout)
            output_dim - int, specifying the size of output
            rank - int, specifying the size of rank in LMF
        Output:
            (return value in forward) a scalar value between -3 and 3
        '''
        super(LMF, self).__init__()

        # dimensions are specified in the order of audio, video and text
        self.text_in, self.audio_in = 128, 128
        self.text_hidden, self.audio_hidden = 128, 128

        self.text_out= self.text_hidden // 2
        self.output_dim = 1
        self.rank = 3

        self.audio_prob, self.video_prob, self.text_prob, self.post_fusion_prob = 0.3, 0.3, 0.3, 0.3

        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        # self.post_fusion_layer_1 = nn.Linear((self.text_out + 1) * (self.video_hidden + 1) * (self.audio_hidden + 1), self.post_fusion_dim)
        self.audio_factor = nn.Parameter(torch.Tensor(self.rank, self.audio_hidden + 1, self.output_dim))
        self.text_factor = nn.Parameter(torch.Tensor(self.rank, 128 + 1, self.output_dim))
        self.fusion_weights = nn.Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = nn.Parameter(torch.Tensor(1, self.output_dim))

        # init teh factors
        nn.init.xavier_normal_(self.audio_factor)
        nn.init.xavier_normal_(self.text_factor)
        nn.init.xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, text_x, audio_x):
        '''
        Args:
            audio_x: tensor of shape (batch_size, audio_in)
            video_x: tensor of shape (batch_size, video_in)
            text_x: tensor of shape (batch_size, sequence_len, text_in)
        '''
        audio_h = audio_x.mean(1)
        text_h = text_x.mean(1)

        batch_size = audio_h.data.shape[0]

        # next we perform low-rank multimodal fusion
        # here is a more efficient implementation than the one the paper describes
        # basically swapping the order of summation and elementwise product
        # next we perform "tensor fusion", which is essentially appending 1s to the tensors and take Kronecker product
        add_one = torch.ones(size=[batch_size, 1], requires_grad=False).type_as(audio_h).to(text_x.device)
        _audio_h = torch.cat((add_one, audio_h), dim=1)
        _text_h = torch.cat((add_one, text_h), dim=1)

        # print(_audio_h.shape, self.audio_factor.shape, _text_h.shape, self.text_factor.shape)
        fusion_audio = torch.matmul(_audio_h, self.audio_factor)
        fusion_text = torch.matmul(_text_h, self.text_factor)
        fusion_zy = fusion_audio * fusion_text

        # output = torch.sum(fusion_zy, dim=0).squeeze()
        # use linear transformation instead of simple summation, more flexibility
        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)

        return output
class TFN(nn.Module):
    '''
    Implements the Tensor Fusion Networks for multimodal sentiment analysis as is described in:
    Zadeh, Amir, et al. "Tensor fusion network for multimodal sentiment analysis." EMNLP 2017 Oral.
    '''

    def __init__(self):
        '''
        Args:
            input_dims - a length-3 tuple, contains (audio_dim, video_dim, text_dim)
            hidden_dims - another length-3 tuple, similar to input_dims
            text_out - int, specifying the resulting dimensions of the text subnetwork
            dropouts - a length-4 tuple, contains (audio_dropout, video_dropout, text_dropout, post_fusion_dropout)
            post_fusion_dim - int, specifying the size of the sub-networks after tensorfusion
        Output:
            (return value in forward) a scalar value between -3 and 3
        '''
        super(TFN, self).__init__()

        # dimensions are specified in the order of audio, video and text
        self.text_in, self.audio_in, self.video_in = 256, 256, 256
        self.text_hidden, self.audio_hidden, self.video_hidden = 256, 256,256
        self.output_dim = 3

        self.text_out= 256
        self.post_fusion_dim = 256

        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=0.5)
        self.post_fusion_layer_1 = nn.Linear((self.text_out + 1) * (self.audio_hidden + 1), self.post_fusion_dim)
        self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, self.output_dim)

        # in TFN we are doing a regression with constrained output range: (-3, 3), hence we'll apply sigmoid to output
        # shrink it to (0, 1), and scale\shift it back to range (-3, 3)
        self.output_range = nn.Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = nn.Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, text_x, audio_x):
        '''
        Args:
            audio_x: tensor of shape (batch_size, audio_in)
            video_x: tensor of shape (batch_size, video_in)
            text_x: tensor of shape (batch_size, sequence_len, text_in)
        '''
        audio_x = audio_x.mean(1)

        audio_h = audio_x
        text_h = text_x.mean(1)
        batch_size = audio_h.data.shape[0]

        # next we perform "tensor fusion", which is essentially appending 1s to the tensors and take Kronecker product
        add_one = torch.ones(size=[batch_size, 1], requires_grad=False).type_as(audio_h).to(text_x.device)
        _audio_h = torch.cat((add_one, audio_h), dim=1)
        _text_h = torch.cat((add_one, text_h), dim=1)

        fusion_tensor = torch.bmm(_audio_h.unsqueeze(2), _text_h.unsqueeze(1)).view(batch_size, -1)

        post_fusion_dropped = self.post_fusion_dropout(fusion_tensor)
        post_fusion_y_1 = nn.functional.relu(self.post_fusion_layer_1(post_fusion_dropped), inplace=True)
        post_fusion_y_2 = nn.functional.relu(self.post_fusion_layer_2(post_fusion_y_1), inplace=True)
        output = self.post_fusion_layer_3(post_fusion_y_2)
        if self.output_dim == 1: # regression
            output = torch.sigmoid(output)
            output = output * self.output_range + self.output_shift

        # res = {
        #     'Feature_t': text_h,
        #     'Feature_a': audio_h,
        #     'Feature_v': video_h,
        #     'Feature_f': fusion_tensor,
        #     'M': output
        # }

        return output

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


class   MyMultimodal_Integrally(nn.Module):
    def __init__(self, num_classs, visual_seq_len=8, audio_seq_len=8, batchsize=8, trans_depth=2, cross_depth=2):
        super(MyMultimodal_Integrally, self).__init__()

        self.visual_seq_len = visual_seq_len
        self.audio_seq_len = audio_seq_len

        #self.bertmodel = BertTextEncoder(use_finetune=True, transformers='bert', pretrained='bert-base-uncased')

        # mosi
        # self.proj_l = nn.Linear(768,32)
        # self.proj_a = nn.Linear(5,32)
        # self.proj_v = nn.Linear(20, 32)

        # mosei
        # self.proj_l = nn.Linear(768, 256)
        # self.proj_a = nn.Linear(74, 256)
        # self.proj_v = nn.Linear(35, 256)
        # #ch-sims
        #256
        numlayer=256
        self.proj_l = nn.Linear(768, numlayer)
        self.proj_a = nn.Linear(33, numlayer)
        self.proj_v = nn.Linear(709, numlayer)
        
        self.hidden_size = 256
        depthx=2
        depthy=2
        print(depthx)
        # self.attention_layer = nn.Sequential(
        #     nn.Linear(256 + 256, 256),
        #     nn.Tanh(),
        #     nn.Linear(256, 1),
        #     nn.Softmax(dim=1)
        # )
        self.project_layer = ProjetTransformer(num_frames=60, dim=numlayer, depth=depthx, heads=8, mlp_dim=512, dim_head = 64, len_invariant = 6, len_specific = 6, dropout = 0.5, emb_dropout = 0.1)

        self.dropout = nn.Dropout(0.5)

        # self.fusion_layer_1 = CrossTransformer(source_num_frames = 150,
        #                                            tgt_num_frames = 150,
        #                                            dim=numlayer,
        #                                            depth=depthy,
        #                                            heads=8,
        #                                            mlp_dim=256,
        #                                            dropout=0.5,
        #                                            emb_dropout=0.1
        #                                            )

        # self.fusion_layer_2 = CrossTransformer(source_num_frames = 150,
        #                                            tgt_num_frames = 150,
        #                                            dim=numlayer,
        #                                            depth=depthy,
        #                                            heads=8,
        #                                            mlp_dim=256,
        #                                            dropout=0.5,
        #                                            emb_dropout=0.1
        #                                            )
            
        self.fusion_layer =TFN()
        # print('tfn')
        # exit(0)
        #self.fusion_layer =nn.LSTM(256, 256, batch_first=True)
        #self.fusion_layer =nn.GRU(256, 256, batch_first=True)

        
        self.cls_head = nn.Sequential(
            nn.Linear(numlayer, 128),
            nn.Linear(128, num_classs)
        )
        # self.cls_head = nn.Sequential(
        #     nn.Linear(512, 256),
        #     nn.Linear(256, num_classs)
        # )

    def forward(self, x_visual, x_audio, x_text):

        #x_text = self.bertmodel(x_text)
        
        x_visual = self.dropout(x_visual)
        x_audio = self.dropout(x_audio)
        x_text = self.dropout(x_text)
        # print(x_visual.shape)
        x_visual = self.proj_v(x_visual)
        x_audio = self.proj_a(x_audio)
        x_text = self.proj_l(x_text)

        x = torch.cat((x_visual, x_audio, x_text), dim=1)

        x = self.project_layer(x)
        # print(x.shape)
        # exit(0)
        x_invariant, x_specific_v, x_specific_a, x_specific_t = x[:, :6], x[:, 6:8], x[:, 8:10], x[:, 10:12]
        
        feat_specific = torch.cat((x_specific_v, x_specific_a, x_specific_t), dim=1)


        #TFN()
        # print('tfn')    
        # print(x_invariant.shape)
       
        # print(feat_specific.shape) 
                       
        feat= self.fusion_layer( x_invariant,feat_specific)
        cls_output=feat
        # print(cls_output.shape)
        # exit(0)

        # print(feat.shape)
        # exit(0) 
        #lstm
        #gru
        # print(x_invariant.shape)
        # print(feat_specific.shape)

        # concatf=torch.cat([x_invariant, feat_specific], dim=1)
        # feat,_=self.fusion_layer(concatf)
        # feat = feat.mean(dim=1)

        # feat_1 = self.fusion_layer_1(x_invariant, feat_specific).mean(dim=1)
        # feat_2 = self.fusion_layer_2(feat_specific, x_invariant).mean(dim=1)
        # print(feat_1.shape)
        # print(feat_2.shape)
        # combined_feature = torch.cat((feat_1, feat_2), dim=1)
        # attention_scores = self.attention_layer(combined_feature)
        # feat =  attention_scores * feat_1 + (1 - attention_scores) * feat_2
        #print( fused_feature.shape)
        # feat=torch.cat(( feat_1,feat_2),dim=1)
        #print(feat.shape)
        #exit(0)
        # feat = feat_1 + feat_2
        #print(feat.shape)
        #exit(0)
        # cls_output = self.cls_head(feat)
        #exit(0)
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
