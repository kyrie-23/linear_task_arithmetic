import json
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from src.args import parse_arguments
from src.eval import eval_single_dataset
from src.modeling import ImageClassifier, ImageEncoder
from src.linearize import LinearizedImageEncoder, LinearizedWithRelu
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
import numpy as np
import torch.nn.functional as F
from scipy import stats
args = parse_arguments()
if args.seed is not None:
    args.save = f"checkpoints_{args.seed}/{args.model}"
else:
    args.save = f"checkpoints/{args.model}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

task_vectors_flat = [] 
task_vectors = []   
dataloaders = []   
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
        task_vectors.append(task_vector)
        param = []
        for params in task_vector.vector.items():
            param.append(params[1].view(-1))  # 展平每个参数，并加入列表
        task_vectors_flat.append(torch.cat(param))
            
    except FileNotFoundError:
        print(f"Error: Could not find {finetuned_checkpoint}.")
        continue
    
    dataset_ = get_dataset(
        dataset,
        ImageEncoder(args).val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    dataloaders.append(get_dataloader(dataset_, is_train=False, args=args, image_encoder=None))

    

for i in range(len(dataloaders)):
    image_encoder = task_vectors[i].apply_to(
                pretrained_checkpoint, residual=False
            ).to(device)
    cor = np.zeros((len(dataloaders)))
    cos = np.zeros((len(dataloaders)))
    for j in range(len(dataloaders)):
        if i==j:
            cor[j] =1
            cos[j] =1
            continue
        with torch.no_grad():
            output = 0
            for data1 in iter(dataloaders[j]):
                data1 = maybe_dictionarize(data1)
                x1 = data1["images"].to(device)
                output += (image_encoder(x1)).mean()
            output /= len(dataloaders[j])
            cor[j] = output
            cos[j] = F.cosine_similarity(task_vectors_flat[i].unsqueeze(0), task_vectors_flat[j].unsqueeze(0), dim=1)
    corr = stats.spearmanr(cor, cos)
    print(f"Correlation: {corr.correlation:.4f}, p-value: {corr.pvalue:.4f}")
        