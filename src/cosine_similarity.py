import json
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from src.args import parse_arguments
from src.eval import eval_single_dataset
from src.linearize import LinearizedImageEncoder, LinearizedWithRelu
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector
import numpy as np
import torch.nn.functional as F
args = parse_arguments()
if args.seed is not None:
    args.save = f"checkpoints_{args.seed}/{args.model}"
else:
    args.save = f"checkpoints/{args.model}"

accuracies = {}


print("*" * 100)
if args.finetuning_mode == "none":
    print("Evaluating pretrained models.")
elif args.finetuning_mode == "standard":
    print("Evaluating non-linear FT models.")
elif args.finetuning_mode == "linear":
    print("Evaluating linear FT models.")
elif args.finetuning_mode == "relu":
    print("Evaluating linear relu FT models.")
elif args.finetuning_mode == "posthoc":
    print("Evaluating post-hoc linearized models.")

eval_datasets = [
    "Cars",
    "DTD",
    "EuroSAT",
    "GTSRB",
    "MNIST",
    "RESISC45",
    "SVHN",
    "SUN397",
]

task_vectors = []        
for dataset in eval_datasets:
    print("*" * 100)
    print(f"Evaluating on {dataset}")

    pretrained_checkpoint = (
        f"{args.save}/{dataset}Val/zeroshot.pt"
        if args.finetuning_mode == "standard" or args.finetuning_mode == "none"
        else f"{args.save}/{dataset}Val/{args.finetuning_mode}_zeroshot.pt"
    )


    finetuned_checkpoint = (
        f"{args.save}/{dataset}Val/finetuned.pt"
        if args.finetuning_mode == "standard" or args.finetuning_mode == "none"
        else f"{args.save}/{dataset}Val/{args.finetuning_mode}_finetuned.pt"
    )

    try:
        task_vector=(
            LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)
            if args.finetuning_mode == "linear"
            else NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
        )
        param = []
        for params in task_vector.vector.items():
            param.append(params[1].view(-1))  # 展平每个参数，并加入列表
        task_vectors.append(torch.cat(param))
            
    except FileNotFoundError:
        print(f"Error: Could not find {finetuned_checkpoint}.")
        continue

i=0
j=0
n=8
cos = np.eye(n)
while i<n:
    j=i+1
    while j<n:
        cos_sim = F.cosine_similarity(task_vectors[i].unsqueeze(0), task_vectors[j].unsqueeze(0), dim=1)
        cos[i][j] = cos_sim
        cos[j][i] = cos[i][j]
        j+=1
    i+=1
    
# 标签
labels = ['Cars', 'DTD', 'EuroSAT', 'GTSRB', 'MNIST', 'RESISC45', 'SUN397', 'SVHN']


if __name__ == "__main__":
    # 使用 seaborn 绘制热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(cos, annot=True, fmt=".2f", cmap='Blues', xticklabels=labels, yticklabels=labels)

    # 添加标题和坐标轴标签
    plt.title(f'Cosine similarity between task vectors - {args.finetuning_mode}')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # 显示图形
    plt.savefig(f'{args.save}/{args.finetuning_mode}_cos_similarity.pdf', dpi=300)