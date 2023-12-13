import torch
from task_vectors import NonLinearTaskVector

pretrained_checkpoint = f'concept-ablation/assets/pretrained_models/sd-v1-4.ckpt'
finetuned_checkpoint = f'concept-ablation/assets/pretrained_models/delta_snoopy_ablated.ckpt'

if __name__ == '__main__':
    # Create the task vector
    task_vector = NonLinearTaskVector(pretrained_checkpoint, finetuned_checkpoint)
    # Negate the task vector
    neg_task_vector = -task_vector
    # Apply the task vector
    model = neg_task_vector.apply_to(pretrained_checkpoint, scaling_coef=5)
    torch.save(model, f'concept-ablation/assets/pretrained_models/snoopy_5enhanced.ckpt')
