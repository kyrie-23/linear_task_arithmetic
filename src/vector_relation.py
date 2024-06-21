from src.args import parse_arguments
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector
from src.heads import get_classification_head
from src.modeling import ImageClassifier, ImageEncoder
from src.linearize import LinearizedImageEncoder
from functorch import jacrev, jvp
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
args = parse_arguments()
eval_datasets = [
    "Cars",
    # "DTD",
    # "EuroSAT",
    # "GTSRB",
    # "MNIST",
    # "RESISC45",
    # "SUN397",
    # "SVHN",
]
task_vectors = []  # Task vectors for each task
dataloaders = []  # Dataloaders for each task
for dataset in eval_datasets:
    # if args.finetuning_mode == "linear":
    #     pretrained_checkpoint = f"{args.save}/{dataset}Val/linear_zeroshot.pt"
    #     finetuned_checkpoint = f"{args.save}/{dataset}Val/linear_finetuned.pt"
    #     task_vectors.append(
    #         LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)
    #     )
    # else:
    #     pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
    #     finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"
    #     task_vectors.append(
    #         NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
    #     )
    dataset_ = get_dataset(
        dataset,
        ImageEncoder(args).val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size,
    )
    dataloaders.append(get_dataloader(dataset_, is_train=False, args=args, image_encoder=None))



if args.seed is not None:
    args.save = f"checkpoints_{args.seed}/{args.model}"
else:
    args.save = f"checkpoints/{args.model}"

for dataset in eval_datasets:    
    pretrained_checkpoint = f"{args.save}/{dataset}Val/zeroshot.pt"
    finetuned_checkpoint = f"{args.save}/{dataset}Val/finetuned.pt"
    task_vector = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
    # lin_task_vector = NonLinearTaskVector(
    #         vector=lin_task_vector.get_named_parameters(task_vector.vector.keys())
    #     )
    pretrained_checkpoint = f"{args.save}/{dataset}Val/linear_zeroshot.pt"
    finetuned_checkpoint = f"{args.save}/{dataset}Val/linear_finetuned.pt"
    lin_task_vector = LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint)

    
def evaluate_model(pretrained_checkpoint, task_vector):

    pretrained_encoder = task_vector.apply_to_linear(
            pretrained_checkpoint, scaling_coef=1.0
        )
    


    pretrained_encoder.eval()
    return pretrained_encoder

def projection(model, dataloader):
    proj = 0
    for data1 in iter(dataloader):
        data1 = maybe_dictionarize(data1)
        x1 = data1["images"].to(device)
        # jacobian.append(jacrev(model.model.func0)(tuple(model.model.params0),x1))
        dparams = [p - p0 for p, p0 in zip(model.model.params, model.model.params0)]   
        out, dp = jvp(
            lambda param: model.model.func0(param, x1),
            (tuple(model.model.params0),),
            (tuple(dparams),),
        )
        out, dp = out.mean(), dp.mean()
        proj += ((out-model.model.func0(model.model.params, x1).mean())/dp)
    return proj/len(dataloader)

model = evaluate_model(pretrained_checkpoint, task_vector).cuda()
projection(model, dataloaders[0])