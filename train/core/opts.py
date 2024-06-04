import argparse



parser = argparse.ArgumentParser()

# model
parser.add_argument("--feature_dims", type=tuple, default=[512,512], required=False)
parser.add_argument("--hidden_dims", type=tuple, default=[512,512], required=False)
parser.add_argument("--dropouts", type=tuple, default=[0.5,0.5,0.5], required=False)
parser.add_argument("--rank", type=int, default=3, required=False)
parser.add_argument("--num_classes", type=int, default=6, required=False)
parser.add_argument("--train_mode", type=str, default="classification", required=False)

# dataset
parser.add_argument("--dataset_name", type=str, default="eNTERFACE", required=False)
parser.add_argument("--featurePath", type=str, default="./data/eNTERFACE/eNTERFACE_Set6.pkl", required=False)
parser.add_argument("--seq_lens", type=int, default=8, required=False)
parser.add_argument("--batch_size", type=int, default=64, required=False)
parser.add_argument("--need_normalized", type=bool, default=True, required=False)

# training paraments
parser.add_argument("--n_epoch", type=int, default=200, required=False)
parser.add_argument("--factor_lr", type=float, default=1e-4, required=False)
parser.add_argument("--learning_rate", type=float, default=1e-3, required=False)
parser.add_argument("--weight_decay", type=float, default=1e-4, required=False)
parser.add_argument("--max_acc", type=float, default=0, required=False)
parser.add_argument("--checkpoint_path", type=str, default='./checkpoint/LMF/eNTERFACE/Set6', required=False)
parser.add_argument("--seed", type=int, default=1116, required=False)
args = parser.parse_args()