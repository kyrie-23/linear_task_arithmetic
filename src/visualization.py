import torch
import numpy as np
import tqdm
from src import utils
from src.heads import get_classification_head
from src.modeling import ImageClassifier, ImageEncoder
from src.task_vectors import LinearizedTaskVector, NonLinearTaskVector, linear_to_nonlinear
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset
from src.linearize import LinearizedImageEncoder
from src.args import parse_arguments
import matplotlib.pyplot as plt
import torch.nn.functional as F


loss_fn = torch.nn.CrossEntropyLoss()
args = parse_arguments()
if args.seed is not None:
    args.save = f"checkpoints_{args.seed}/{args.model}"
else:
    args.save = f"checkpoints/{args.model}"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def evaluate_model(pretrained_checkpoint, task_vector, dataset_name, args, scaling_coef=1.0):
    image_encoder = task_vector.apply_to(
        pretrained_checkpoint, scaling_coef=scaling_coef
    )
    if args.finetuning_mode == "posthoc":
        pretrained_encoder = task_vector.apply_to(
            pretrained_checkpoint, scaling_coef=0.0
        )
        image_encoder = LinearizedImageEncoder(
            init_encoder=pretrained_encoder, image_encoder=image_encoder, args=args
        )
    # Compute model predictions with modified parameters
    classification_head = get_classification_head(args, dataset_name)
    model = ImageClassifier(image_encoder, classification_head).to(device)
    model.freeze_head()
    return model

def predict(x, model):    
    with torch.no_grad():
        logits = utils.get_logits(x, model)
        pred = logits.argmax(dim=1, keepdim=True).to(device)
    return pred

def prediction_error(y_pred, y_true):
    # Prediction error as a distance metric
    return (y_pred != y_true).float().mean()


def compute_disentanglement_error(pretrained_checkpoint, task_vector, dataloader, eval_dataset, alpha1, alpha2):
    err1 = 0
    err2 = 0
    image_encoder = task_vector[0].apply_to(
        pretrained_checkpoint, scaling_coef=alpha1
    )
    classification_head = get_classification_head(args, eval_dataset[0])
    model = ImageClassifier(image_encoder, classification_head).to(device)
    model.freeze_head()
    sum_vector = task_vector[0]*alpha1 + task_vector[1]*alpha2
    image_encoder = sum_vector.apply_to(
        pretrained_checkpoint, scaling_coef=1.0
    )
    combined_model = ImageClassifier(image_encoder, classification_head).to(device)
    combined_model.freeze_head()
    with torch.no_grad():
        for i, data1 in enumerate(dataloader[0]):
            if i>15:
                break
            data1 = maybe_dictionarize(data1)
            x1 = data1["images"].to(device)
            # y1 = data1["labels"].to(device)

            # y2 = data2["labels"].to(device)
            pred1 = predict(x1, model)
            combined_pred1 = predict(x1, combined_model)
            
            
            err1 += prediction_error(combined_pred1, pred1)
    err1 /= len(dataloader[0])
    classification_head = get_classification_head(args, eval_dataset[1])
    combined_model = ImageClassifier(image_encoder, classification_head).to(device)
    combined_model.freeze_head()
    image_encoder = task_vector[1].apply_to(
        pretrained_checkpoint, scaling_coef=alpha2
    )
    model = ImageClassifier(image_encoder, classification_head).to(device)
    model.freeze_head()
    
    with torch.no_grad():
        for i, data2 in enumerate(dataloaders[1]):
            if i > 15:
                break
            # y1 = data1["labels"].to(device)
            data2 = maybe_dictionarize(data2)
            x2 = data2["images"].to(device)
            # y2 = data2["labels"].to(device)
            pred2 = predict(x2, model)
            
            combined_pred2 = predict(x2, combined_model)
            
            err2 += prediction_error(combined_pred2, pred2)
    
    err2 /= len(dataloader[1])
    return err1+err2

def compute_disentanglement_rep(pretrained_checkpoint, task_vector, dataloader, eval_dataset, alpha1, alpha2):
    err1 = 0
    err2 = 0
    image_encoder = task_vector[0].apply_to(
        pretrained_checkpoint, scaling_coef=alpha1
    )
    model = image_encoder.to(device)
    sum_vector = task_vector[0]*alpha1 + task_vector[1]*alpha2
    image_encoder = sum_vector.apply_to(
        pretrained_checkpoint, scaling_coef=1.0
    )
    combined_model = image_encoder.to(device)
    with torch.no_grad():
        for i, data1 in enumerate(dataloader[0]):
            if i >15:
                break
            data1 = maybe_dictionarize(data1)
            x1 = data1["images"].to(device)
            # y1 = data1["labels"].to(device)

            # y2 = data2["labels"].to(device)
            pred1 = model(x1).view(x1.size(0), -1)
            combined_pred1 = combined_model(x1).view(x1.size(0), -1)
            
            
            err1 += (F.kl_div(F.log_softmax(pred1, dim=1),F.softmax(combined_pred1, dim=1), reduction='batchmean')
            + F.kl_div(F.log_softmax(combined_pred1, dim=1),F.softmax(pred1, dim=1), reduction='batchmean'))/2
    err1 /= len(dataloader[0])
    image_encoder = task_vector[1].apply_to(
        pretrained_checkpoint, scaling_coef=alpha2
    )
    model = image_encoder.to(device)
    
    with torch.no_grad():
        for i, data2 in enumerate(dataloaders[1]):
            if i >15:
                break
            # y1 = data1["labels"].to(device)
            data2 = maybe_dictionarize(data2)
            x2 = data2["images"].to(device)
            # y2 = data2["labels"].to(device)
            pred2 = model(x2).view(x2.size(0), -1)
            combined_pred2 = combined_model(x2).view(x2.size(0), -1)
            
            err2 += (F.kl_div(F.log_softmax(pred2, dim=1),F.softmax(combined_pred2, dim=1), reduction='batchmean')
            + F.kl_div(F.log_softmax(combined_pred2, dim=1),F.softmax(pred2, dim=1), reduction='batchmean'))/2
    
    err2 /= len(dataloader[1])
    return err1+err2

# Example Usage
# Define your model, tasks, and datasets
# alpha1, alpha2 = 0.5, 0.5  # Scaling factors for task vectors
task_vectors = []  # Task vectors for each task
dataloaders = []  # Dataloaders for each task
eval_datasets = [
    "Cars",
    # "DTD",
    # "EuroSAT",
    # "GTSRB",
    # "MNIST",
    "RESISC45",
    # "SVHN",
    # "SUN397",
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
    
# Define the range and size of the matrix
n_points = 15
alpha1_range = np.linspace(-2.0, 2.0, n_points)
alpha2_range = np.linspace(-2.0, 2.0, n_points)

# Initialize the matrix to store the values
err_matrix = np.zeros((n_points, n_points))

# Loop through the range values
for i, alpha1 in enumerate(alpha1_range):
    for j, alpha2 in enumerate(alpha2_range):
        # Assume some function f to calculate a value, here just a simple example
        err_matrix[i, j] = compute_disentanglement_rep(pretrained_checkpoint, task_vectors, dataloaders, eval_datasets, alpha1, alpha2)

if __name__ == "__main__":       
    # Create the heatmap
    fig, ax = plt.subplots()
    cax = ax.imshow(err_matrix, interpolation='nearest', cmap='coolwarm', extent=[-2, 2, -2, 2], vmin=0, vmax=0.2)
    fig.colorbar(cax)

    # # Adding annotations
    # # Arrow
    # ax.annotate('', xy=(5, 5), xytext=(7, 7), arrowprops=dict(facecolor='red', shrink=0.05))
    # # Text
    # ax.text(5, 5, r'$\theta^*$', fontsize=12, ha='center', va='center', color='white')
    # # Rectangle
    # from matplotlib.patches import Rectangle
    # rect = Rectangle((3,3), width=4, height=4, edgecolor='red', facecolor='none', linestyle='dashed')
    # ax.add_patch(rect)

    # Set axis limits and labels as needed
    ax.set_xticks(np.linspace(-2, 2, 5))
    ax.set_yticks(np.linspace(-2, 2, 5))
    ax.set_xlabel(r'$\alpha_1$')
    ax.set_ylabel(r'$\alpha_2$')

    # Title
    ax.set_title('Cars - RESISC45')

    plt.savefig(f'disentanglement_rep{args.finetuning_mode}_3.pdf', dpi=300)
