import torch
import numpy as np
import tqdm
from src import utils
from src.heads import get_classification_head
from src.modeling import ImageClassifier, ImageEncoder
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector, linear_to_nonlinear
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.linearize import LinearizedImageEncoder, ReluEncoder
from src.args import parse_arguments
import matplotlib.pyplot as plt
args = parse_arguments()
if args.seed is not None:
    args.save = f"checkpoints_{args.seed}/{args.model}"
else:
    args.save = f"checkpoints/{args.model}"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
task_vectors = []  # Task vectors for each task
dataloaders = []  # Dataloaders for each task
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

# pre = f"{args.save}/CarsVal/zeroshot.pt"
# fine = f"{args.save}/CarsVal/finetuned.pt"
# nonlinear_vector = NonLinearTaskVector(pre, fine)
for dataset in eval_datasets:
    if args.finetuning_mode != "standard":
        pretrained_checkpoint = f"{args.save}/{dataset}Val/{args.finetuning_mode}_zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}Val/{args.finetuning_mode}_finetuned.pt"
        task_vectors.append(
            LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint) 
            if args.finetuning_mode == "linear" else NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
        )
    else:
        pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"
        task_vectors.append(
            NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
        )
    dataset_ = get_dataset(
        dataset,
        ImageEncoder(args).val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    dataloaders.append(get_dataloader(dataset_, is_train=False, args=args, image_encoder=None))

attention_outputs = []
def get_attention_outputs(name):
    def hook(model, input, output):
        print(f"Layer {name} Output: {output}")
        attention_outputs.append(output)
        if 'attention' in name:
            print(torch.mm(input[0][0],model.in_proj_weight[:768]).mean())
    return hook

for i in range(len(dataloaders)):
    # task_vector = sum(task_vectors)-task_vectors[i]
    task_vector = sum(task_vectors)
    image_encoder = task_vector.apply_to(
            pretrained_checkpoint, residual=False
        ).to(device)
    # for name, layer in image_encoder.image_encoder.model.visual.transformer.resblocks.named_children():
    #     layer.attn.register_forward_hook(get_attention_outputs(f"attention_{name}"))
    #     layer.attn.out_proj.register_forward_hook(get_attention_outputs(f"out_proj_{name}"))
    with torch.no_grad():
        outputs = 0
        for data1 in iter(dataloaders[i]):
            data1 = maybe_dictionarize(data1)
            x1 = data1["images"].to(device)
            output = (image_encoder(x1))
            # print(output)
            outputs += output.mean()
    outputs /= len(dataloaders[i])
    print(f'Output for {eval_datasets[i]}: {outputs}')
        

    
    