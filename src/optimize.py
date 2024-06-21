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
from functorch import jvp, make_functional_with_buffers

loss_fn = torch.nn.CrossEntropyLoss()
args = parse_arguments()
if args.seed is not None:
    args.save = f"checkpoints_{args.seed}/{args.model}"
else:
    args.save = f"checkpoints/{args.model}"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def evaluate_model(pretrained_checkpoint, task_vector, dataset_name, args, scaling_coef=0.3):
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

def evaluate_model2(pretrained_checkpoint, task_vector, scaling_coef=0.3):
    pretrained_encoder = task_vector.apply_to(
            pretrained_checkpoint, scaling_coef=0.0
        )

    image_encoder = task_vector.apply_to(
        pretrained_checkpoint, scaling_coef=scaling_coef
    )
    
    return pretrained_encoder.to(device), image_encoder.to(device)

def predict(x, model):    
    with torch.no_grad():
        logits = utils.get_logits(x, model)
        pred = logits.argmax(dim=1, keepdim=True).to(device)
    return pred

def prediction_error(y_pred, y_true):
    # Prediction error as a distance metric
    return (y_pred == y_true).float().mean()


def compute_disentanglement_error(pretrained_checkpoint, task_vector1,  task_vector2, dataloader, eval_dataset):
    xi = 0
    model1 = evaluate_model(pretrained_checkpoint, task_vector1, eval_dataset, args)
    combined_model1 = evaluate_model(pretrained_checkpoint, task_vector2, eval_dataset, args)
    # model2 = evaluate_model(pretrained_checkpoint, task_vectors[1], eval_datasets[1], args)
    # combined_model2 = evaluate_model(pretrained_checkpoint, sum(task_vectors), eval_datasets[1], args)
    for data1 in iter(dataloader):
        data1 = maybe_dictionarize(data1)
        x1 = data1["images"].to(device)
        # y1 = data1["labels"].to(device)

        # y2 = data2["labels"].to(device)
        pred1 = predict(x1, model1)
        combined_pred1 = predict(x1, combined_model1)
        
        
        xi += prediction_error(combined_pred1, pred1)
        
    # for data2 in iter(dataloaders[1]):
    #     # y1 = data1["labels"].to(device)
    #     data2 = maybe_dictionarize(data2)
    #     x2 = data2["images"].to(device)
    #     # y2 = data2["labels"].to(device)
    #     pred2 = predict(x2, model2)
        
    #     combined_pred2 = predict(x2, combined_model2)
        
    #     xi += prediction_error(combined_pred2, pred2)
    
    xi /= len(dataloader)
    return xi

def optimize_disentanglement(pretrained_checkpoint, task_vectors, dataloaders):
    task_vector = sum(task_vectors)
    combined_model = task_vector.apply_to(pretrained_checkpoint, scaling_coef=1).to(device)
    params = [p for p in combined_model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=args.wd)
    loss = 0
    for j in range(50):
        for i in range(len(task_vectors)):
            optimizer.zero_grad()
            eval_model = task_vectors[i].apply_to(pretrained_checkpoint, scaling_coef=1).to(device).eval()
            for data1 in iter(dataloaders[i]):
                data1 = maybe_dictionarize(data1)
                x1 = data1["images"].to(device)
                pred1 = eval_model(x1)
                combined_pred1 = combined_model(x1)
                loss += loss_fn(combined_pred1, pred1)
            loss /= len(dataloaders[i])
            print(f"Loss for task {i}: {loss}")
        loss /= len(task_vectors)
        loss.backward()
        optimizer.step()
    combined_model.save(f"{args.save}/combined.pt")
    

def parameters_to_vector(parameter_list):
    # Flatten and concatenate parameters into a single vector
    param_tensors = [p.view(-1) for p in parameter_list]
    flat_parameters = torch.cat(param_tensors)
    return flat_parameters

def set_flat_parameters(model, flat_parameters):
    # Pointer to keep track of position in flat_parameters
    pointer = 0
    for p in model.parameters():
        # The number of elements in this parameter
        num_elements = p.numel()
        # Replace the existing parameters with the corresponding part of flat_parameters
        p.data = flat_parameters[pointer:pointer + num_elements].view_as(p)
        pointer += num_elements

# def compute_ntk(model, dataloader):
#     # Define a function to compute the output given parameters and inputs
#     def model_apply(x):
#         return model(x.unsqueeze(0))

#     # Vectorize the Jacobian computation over batches
#     batch_jacobian = functorch.vmap(functorch.jacrev(model_apply))

#     ntk_matrix = 0
#     total_samples = 0
#     for batch in iter(dataloader):
#         batch = maybe_dictionarize(batch)
#         images = batch["images"].to(device)
#         # Compute the Jacobians for the entire batch
#         jacobians = batch_jacobian(images)
#         # Compute the NTK for the batch using einsum
#         ntk = torch.einsum('bik,bjk->bij', jacobians, jacobians)
#         ntk_matrix += ntk.sum(0)
#         total_samples += images.size(0)
    
#     ntk_matrix /= total_samples
#     return ntk_matrix

def compute_ntk(pretrained_encoder, image_encoder, dataloader):
    dps = 0
    for batch in iter(dataloader):
        batch = maybe_dictionarize(batch)
        images = batch["images"].to(device)
        out = image_encoder(images)-pretrained_encoder(images)
        dps += out
    dparams = [p - p0 for p, p0 in zip(image_encoder.parameters(), pretrained_encoder.parameters())]
    dparams = parameters_to_vector(dparams).unsqueeze(1)
    return torch.matmul(dps,torch.inverse(dparams))/len(dataloader)

class LinearizedWithRelu(torch.nn.Module):
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.bias = None
        
    def forward(self, x):
        return torch.nn.functional.relu(self.model(x))
 
def _get_submodules(model, key):   
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name

def optimize_jvp(pretrained_checkpoint, task_vectors, dataloaders):
    lamda = 1
    task_vector = sum(task_vectors)
    combined_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=0.3).to(device)
    _, params, _ = make_functional_with_buffers(
            combined_encoder, disable_autograd_tracking=True
            )
    
    # pretrained_encoder = task_vectors[0].apply_to(pretrained_checkpoint, scaling_coef=0).to(device)
            
    # func0, params0, buffers0 = make_functional_with_buffers(
    #                 pretrained_encoder.eval(), disable_autograd_tracking=True
    #             )
    # func = lambda params, x: func0(params, buffers0, x)
            
            
    # params0 = torch.nn.ParameterList(params0)
    # for p in params0:
    #     p.requires_grad = False
    for params in combined_encoder.parameters():
        params.requires_grad = False
    
    for key, module in combined_encoder.named_modules():
        if "out_proj" in key and isinstance(module, torch.nn.Linear):
            for params in module.parameters():
                params.requires_grad = True
            parent, target, target_name = _get_submodules(combined_encoder,key)
            setattr(parent, target_name, LinearizedWithRelu(target))
    
    params = torch.nn.ParameterList(params)
    for p in params:
        p.requires_grad = True
    optimizer = torch.optim.AdamW(params, lr=0.01, weight_decay=args.wd)
    for i in range(1):
        for j in range(len(dataloaders)):
            classification_head = get_classification_head(args, eval_datasets[j]).to(device)
            classification_head.weight.requires_grad_(False)
            classification_head.bias.requires_grad_(False)
            single_encoder = task_vectors[j].apply_to(pretrained_checkpoint, scaling_coef=1).to(device)
            
            func0, params0, buffers0 = make_functional_with_buffers(
                    single_encoder.eval(), disable_autograd_tracking=True
                )
            func = lambda params, x: func0(params, buffers0, x)
            
            
            params0 = torch.nn.ParameterList(params0)
            for p in params0:
                p.requires_grad = False

        # The params are.
            
            

            for q in range(10):
                avg_loss = 0
                for batch in iter(dataloaders[j]):
                    dparams = [p - p0 for p, p0 in zip(params, params0)]
                    optimizer.zero_grad()
                    batch = maybe_dictionarize(batch)
                    images = batch["images"].to(device)
                    _, dp = jvp(
                        lambda param: func(param, images),
                        (tuple(params0),),
                        (tuple(dparams),),
                    )
                    pred = classification_head(dp)
                    zero = torch.ones(pred.shape[0], pred.shape[1]) / pred.shape[1]
                    loss = torch.nn.functional.kl_div(pred.log_softmax(dim=1), zero.to(device).softmax(dim=1),reduce='mean')
                    avg_loss += loss
                    # print(f"Loss: {loss}")
                    loss.backward()
                    optimizer.step()
                avg_loss /= len(dataloaders[j])
                print(f"Epoch {q}, Average Loss for task {j}: {avg_loss}")
    # Assuming 'combined_encoder' is recreated or obtained similarly as before
            _, original_params, _ = make_functional_with_buffers(
            combined_encoder, disable_autograd_tracking=True
            )

    # Iterate over the original model's parameters and replace them with loaded ones
            for orig_param, loaded_param in zip(original_params, params):
                orig_param.data.copy_(loaded_param.data)
        
            acc = eval(combined_encoder, classification_head, dataloaders[j])
            print(f"Single Accuracy for task {j}: {100*acc:.2f}%")
            
        for j in range(len(dataloaders)):
            classification_head = get_classification_head(args, eval_datasets[j]).to(device)
            acc = eval(combined_encoder, classification_head, dataloaders[j])
            print(f"Additional Accuracy for task {j}: {100*acc:.2f}%")
        

        combined_encoder.save(f"{args.save}/{args.finetuning_mode}_localized_epoch{i}.pt")


def optimize_vector(pretrained_checkpoint, task_vectors, dataloaders):
    sum_vector = sum(task_vectors)
    for task_vector in task_vectors:
        sum_vector_ = sum_vector-task_vector
        params=[]
        for param in sum_vector_.vector.values():
            param.requires_grad = True
            params.append(param)
        optimizer = torch.optim.Adam(params, lr=0.01)
        
        for i in range(10):
            for key, value in task_vector.vector.items():
                optimizer.zero_grad()
                cos = torch.nn.functional.cosine_similarity(task_vector.vector[key], sum_vector_.vector[key], dim=0)
                # print(cos)
                loss = torch.mean(torch.abs(cos))
                loss.backward()
                optimizer.step()
    model = sum_vector.apply_to(pretrained_checkpoint, scaling_coef=1).to(device)
    model.save(f"{args.save}/vector_localized.pt")

        

# Eigenfunction Decomposition
def eigen_decomposition(ntk_matrix):
    eigenvalues, eigenvectors = torch.linalg.eigh(ntk_matrix)
    return eigenvalues, eigenvectors

def compute_local_energy(ntk_x, eigenvectors):
    eigenfunctions_x = torch.matmul(ntk_x, eigenvectors)
    local_energy = torch.sum(eigenfunctions_x ** 2, dim=-1)
    return local_energy

def localization():

    optimize_jvp(pre, task_vectors, dataloaders)
    # optimize_vector(pre, task_vectors, dataloaders)
    
def eval(image_encoder, classification_head, dataloader):
    model = ImageClassifier(image_encoder, classification_head)

    model.eval()
    with torch.no_grad():
        top1, correct, n = 0.0, 0.0, 0.0
        for _, data in enumerate(tqdm.tqdm(dataloader)):
            data = maybe_dictionarize(data)
            x = data["images"].to(device)
            y = data["labels"].to(device)

            logits = utils.get_logits(x, model)

            pred = logits.argmax(dim=1, keepdim=True).to(device)

            correct += pred.eq(y.view_as(pred)).sum().item()

            n += y.size(0)

        top1 = correct / n

    return top1

# Example Usage
# Define your model, tasks, and datasets
alpha1, alpha2 = 0.5, 0.5  # Scaling factors for task vectors
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

pre = f"{args.save}/CarsVal/zeroshot.pt"
fine = f"{args.save}/CarsVal/finetuned.pt"
nonlinear_vector = NonLinearTaskVector(pre, fine)
for dataset in eval_datasets:
    if args.finetuning_mode == "linear":
        pretrained_checkpoint = f"{args.save}/{dataset}Val/linear_zeroshot.pt"
        finetuned_checkpoint = f"{args.save}/{dataset}Val/linear_finetuned.pt"
        task_vectors.append(
            linear_to_nonlinear(LinearizedTaskVector(pretrained_checkpoint, finetuned_checkpoint),nonlinear_vector.vector.keys())
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
    
# dataloaders = [dataloaders[1]]

# Compute disentanglement error

# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.decomposition import PCA

# # Example: Visualize Task Vectors Using PCA
# def visualize_task_vectors(task_vectors):
#     pca = PCA(n_components=2)
#     reduced_vectors = pca.fit_transform(task_vectors)
#     plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1])
#     for i, txt in enumerate(range(len(task_vectors))):
#         plt.annotate(txt, (reduced_vectors[i, 0], reduced_vectors[i, 1]))
#     plt.xlabel('PCA Component 1')
#     plt.ylabel('PCA Component 2')
#     plt.title('Task Vector Visualization')
#     plt.show()

# # Example: Visualize Eigenfunction Localization Using Heatmap
# def visualize_eigenfunction_localization(eigenfunctions, points):
#     # Assuming eigenfunctions is a 2D array where each row corresponds to a point
#     plt.imshow(eigenfunctions, cmap='hot', interpolation='nearest')
#     plt.colorbar()
#     plt.title('Eigenfunction Localization')
#     plt.show()

# Usage Example
# task_vectors = Your task vectors
# visualize_task_vectors(task_vectors)

# eigenfunctions = Computed eigenfunctions for certain points
# points = Points or regions of interest
# visualize_eigenfunction_localization(eigenfunctions, points)

if __name__ == "__main__":
    # optimize_disentanglement(pretrained_checkpoint, task_vectors, dataloaders)
    localization()
    # print(energy)