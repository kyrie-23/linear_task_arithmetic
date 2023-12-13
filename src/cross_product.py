import torch
from task_vectors import NonLinearTaskVector
import torch.nn.functional as F



pretrained_checkpoint = f'../concept-ablation/assets/pretrained_models/sd-v1-4.ckpt'
finetuned1_checkpoint = f'../concept-ablation/assets/pretrained_models/delta_grumpy_cat_ablated.ckpt'
finetuned2_checkpoint = f'../concept-ablation/assets/pretrained_models/delta_snoopy_ablated.ckpt'

if __name__ == '__main__':
    # Create the task vector
    task_vector1 = NonLinearTaskVector(pretrained_checkpoint, finetuned1_checkpoint)
    task_vector2 = NonLinearTaskVector(pretrained_checkpoint, finetuned2_checkpoint)
    # Negate the task vector
    cos_sim = 0
    for key in task_vector1.vector:
        cos_sim += F.cosine_similarity(task_vector1.vector[key], task_vector2.vector[key], dim=0)
    print(cos_sim/len(task_vector1.vector))
    cross_product_task_vector = (-task_vector1).cross_product(-task_vector2)
    # Apply the task vector
    model = cross_product_task_vector.apply_to(pretrained_checkpoint, scaling_coef=1.0)
    torch.save(model, f'../concept-ablation/assets/pretrained_models/snoopy_grumpycat_cross.ckpt')
