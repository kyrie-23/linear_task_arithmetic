import json
import os

from utils import find_optimal_coef

from src.args import parse_arguments
from src.eval import evaluate_task_vector, evaluate_task_vector_at_coef
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector

args = parse_arguments()
args.save = f"checkpoints/{args.model}"


print("*" * 100)
if args.finetuning_mode == "standard":
    print("Evaluating non-linear FT models.")
    ft_accuracies_path = os.path.join(args.save, "ft_accuracies.json")
elif args.finetuning_mode == "linear":
    print("Evaluating linear FT models.")
    ft_accuracies_path = os.path.join(args.save, "linear_ft_accuracies.json")
elif args.finetuning_mode == "posthoc":
    print("Evaluating post-hoc linearized models.")
    ft_accuracies_path = os.path.join(args.save, "posthoc_ft_accuracies.json")
else:
    raise ValueError(f"Invalid finetuning mode: {args.finetuning_mode}")
print("*" * 100)

with open(ft_accuracies_path) as f:
    args.finetuning_accuracies = json.load(f)['1.0'] if args.finetuning_mode == "standard" else json.load(f)

with open(os.path.join(args.save, "zeroshot_accuracies.json")) as f:
    pretrained_accuracies = json.load(f)

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

pretrained_checkpoint = f"{args.save}/CarsVal/zeroshot.pt"
# finetuned_checkpoint = f"{args.save}/linear_localized.pt" if args.finetuning_mode == "linear" else f"{args.save}/standard_localized.pt"
finetuned_checkpoint = f"{args.save}/vector_localized.pt"

task_vector = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)


args.eval_datasets = [dataset + "Val" for dataset in eval_datasets]
# args.eval_datasets = eval_dataset['1.0'] if args.finetuning_mode == "standard" else eval_dataset
args.control_dataset = None

# We use the validation set to choose the optimal coefficient.
val_metrics = evaluate_task_vector(
    task_vector,
    pretrained_checkpoint,
    args,
    posthoc_linearization=args.finetuning_mode == "posthoc" or "linear",
)

optimal_coef = find_optimal_coef(
    val_metrics,
    metric="avg_normalized_top1",
    minimize=False,
)

# Evaluate on the test set with the optimal coefficient.
args.eval_datasets = [dataset for dataset in eval_datasets]
test_metrics = evaluate_task_vector_at_coef(
    task_vector,
    pretrained_checkpoint,
    args,
    float(optimal_coef),
    posthoc_linearization=args.finetuning_mode == "posthoc" or "linear",
)

print("=" * 100)
print(f"Test normalized accuracy: {test_metrics['avg_normalized_top1']}")
print(f"Test absolute accuracy: {test_metrics['avg_top1']}")
additive_accuracies = {"test": test_metrics, "val": val_metrics}

if args.finetuning_mode == "standard":
    save_file = f"{args.save}/additions_localized.json"
elif args.finetuning_mode == "linear":
    save_file = f"{args.save}/linear_additions_localized.json"
elif args.finetuning_mode == "posthoc":
    save_file = f"{args.save}/posthoc_additions_localized.json"
with open(save_file, "w") as f:
    json.dump(additive_accuracies, f, indent=4)