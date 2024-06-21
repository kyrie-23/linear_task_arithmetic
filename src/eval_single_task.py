import json

from src.args import parse_arguments
from src.eval import eval_single_dataset
from src.linearize import LinearizedImageEncoder, LinearizedWithRelu
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector
import numpy as np
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
for scaling_coef in np.linspace(1.0, args.max_coefficient, args.n_eval_points):
    accuracies[scaling_coef]={}
        
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
        task_vector = (
            LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)
            if args.finetuning_mode == "linear"
            else NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
        )
    except FileNotFoundError:
        print(f"Error: Could not find {finetuned_checkpoint}.")
        continue

    for scaling_coef in np.linspace(1.0, args.max_coefficient, args.n_eval_points):
        if args.finetuning_mode == "none":
            image_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=0.0)
        # elif args.finetuning_mode == "standard" or args.finetuning_mode == "linear" or args.finetuning_mode == "relu":
        #     image_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)
        elif args.finetuning_mode == "posthoc":
            zs_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=0.0)
            ft_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)
            image_encoder = LinearizedImageEncoder(
                init_encoder=zs_encoder, image_encoder=ft_encoder, args=args
            )
        else:
            image_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)

        for split in ["test", "val"]:
            # Evaluate
            print("=" * 100)
            print(f"Evaluating on {split} split.")
            eval_dataset = dataset if split == "test" else f"{dataset}Val"

            accuracies[scaling_coef][eval_dataset] = eval_single_dataset(
                image_encoder, eval_dataset, args
            )["top1"]


# if args.finetuning_mode == "none":
#     # Evaluate zero-shot accuracy on ImageNet
#     for split in ["ImageNetVal", "ImageNet"]:
#         accuracies[split] = eval_single_dataset(image_encoder, split, args)["top1"]

for scaling_coef in np.linspace(1.0, args.max_coefficient, args.n_eval_points):
    accuracies[scaling_coef]["avg"] = np.mean(
        [accuracies[scaling_coef][dataset] for dataset in eval_datasets]
    )

# Save results
if args.finetuning_mode == "none":
    save_path = f"{args.save}/zeroshot_accuracies.json"
elif args.finetuning_mode == "standard":
    save_path = f"{args.save}/ft_accuracies.json"
else:
    save_path = f"{args.save}/{args.finetuning_mode}_ft_accuracies.json"


with open(save_path, "w") as f:
    json.dump(accuracies, f)
