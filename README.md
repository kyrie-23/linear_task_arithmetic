# Fine-Tuning Linear Layers Only Is a Simple yet Effective Way for Task Arithmetic

This is the source code to reproduce the experiments of the paper "[Fine-Tuning Linear Layers Only Is a Simple yet Effective Way for Task Arithmetic](https://arxiv.org/abs/2407.07089)"

Task arithmetic has recently emerged as a cost-effective and scalable approach to edit pre-trained models directly in weight space, by adding the fine-tuned weights of different tasks. The performance has been further improved by a linear property which is illustrated by weight disentanglement. Yet, conventional linearization methods (e.g., NTK linearization) not only double the time and training cost but also have a disadvantage on single-task performance.
We propose a simple yet effective and efficient method that only fine-tunes linear layers, which improves weight disentanglement and efficiency simultaneously. Specifically, our study reveals that finetuning attention modules occurs in a linear regime by fine-tuning only the linear layers, significantly improving weight disentanglement. 
To further understand how our method improves the disentanglement of task arithmetic, we present a comprehensive study of task arithmetic by differentiating the role of representation model and task-specific model. Furthermore, we illustrate that weight disentanglement emerges from the representation model, while the performance of task arithmetic has been constrained by task-specific models, such as classification heads.
Overall, our work uncovers novel insights into the fundamental mechanisms of task arithmetic and offers a more reliable and effective approach to edit pre-trained models.

## Dependencies

To run the code, please install all its dependencies:
```sh
conda env create
conda activate tangent-arithmetic
```
and add the `src` directory to the `PYTHONPATH`:
```sh
cd tangent_task_arithmetic
export PYTHONPATH="$PYTHONPATH:$PWD"
```

## Repository content

This repository is heavily based on the code from [Ilharco et al. (2022)](https://github.com/mlfoundations/task_vectors) and follows the same structure.

### Task vectors

The task vector logic in [src/task_vectors.py](src/task_vectors.py) has been extended to distinguish between `NonLinearTaskVector`s and `LinearizedTaskVector`s which can be applied to non-linear `ImageEncoder`s and `LinearizedImageEncoder`s, respectively. Given a pre-trained checkpoint and a fine-tuned checkpoint, you can create a linearized/standard task vector as:

```python
from src.task_vectors import NonLinearTaskVector, LinearizedTaskVector

# Non-linear task vector.
zeroshot_checkpoint = ... # Pre-trained non-linear image encoder.
finetuned_checkpoint = ... # Non-linearly fine-tuned checkpoint.

nonlinear_task_vector = NonLinearTaskVector(zeroshot_checkpoint, finetuned_checkpoint)

# Tangent task vector.
linear_zeroshot_checkpoint = ... # Pre-trained linearized image encoder.
linear_finetuned_checkpoint = ... # Linearly fine-tuned checkpoint.

linear_task_vector = LinearizedTaskVector(linear_zeroshot_checkpoint, linear_finetuned_checkpoint)
```

Once created, we can modify and combine the task vectors through arithmetic operations in Python, e.g.,
```python
negated_task_vector = -task_vector # Negating a task vector.
multi_task_vector = 0.5 * task_vector_1 + 0.7 * task_vector_2 # Adding two vectors.
```
and apply them to a pre-trained encoder as:
```python
edited_encoder = task_vector.apply_to(pretrained_checkpoint, scaling_coef=0.8)
```

Sometimes, we may want to apply a non-linear task vector to a `LinearizedImageEncoder` (to obtain posthoc linearized models for example), or viceversa. Both `NonLinearTaskVector` and `LinearizedTaskVector` can be casted and applied to encoders from the complementary class as
```python
linear_edited_encoder = nonlinear_task_vector.apply_to_linear(linear_pretrained_encoder, scaling_coef=0.8)
```

### Training

The script `src/finetune.py` can be used to reproduce the training protocol we used to fine-tune our models on all our downstream tasks (both linearly and non-linearly).
```sh 
python src/finetune.py --finetuning-mode=standard --model=ViT-B-32 --world-size=2 # Finetune non-linearly on 2 GPUs
python src/finetune.py --finetuning-mode=linear --model=ViT-B-32 --world-size=2 # Finetune non-linearly on 2 GPUs
python src/finetune.py --finetuning-mode=linear-2 --model=ViT-B-32 --world-size=2 # Finetune non-linearly on 2 GPUs (our method)
```

### Evaluation

We provide different scripts to evaluate the different task vectors obtained using the previous scripts.

#### Single-task accuracy
Having run `src/finetune.py` for a given model, you can evaluate the performance of the fine-tuned weights on each single task by running
```sh 
# Evaluate pre-trained models.
python src/eval_single_task.py --model=ViT-B-32 --finetuning-mode=none

# Evaluate non-linearly fine-tuned models.
python src/eval_single_task.py --model=ViT-B-32 --finetuning-mode=standard

# Evaluate linearly fine-tuned models.
python src/eval_single_task.py --model=ViT-B-32 --finetuning-mode=linear

# Evaluate our method. Requires having run finetune.py with --finetuning=mode=standard.
python src/eval_single_task.py --model=ViT-B-32 --finetuning-mode=linear-2
```

#### Task addition
Once evaluated on the single tasks, we can evaluate the task arithmetic performance of the different strategies on the addition benchmark.
```sh 
# Evaluate non-linearly fine-tuned models.
python src/eval_task_addition.py --model=ViT-B-32 --finetuning-mode=standard

# Evaluate linearly fine-tuned models.
python src/eval_task_addition.py --model=ViT-B-32 --finetuning-mode=linear

# Evaluate our method.
python src/eval_task_addition.py --model=ViT-B-32 --finetuning-mode=linear-2
```


## Datasets
To download and prepare the datasets, please use the code in 'src/datasets' directly.

## Reference
If you find this code useful, please cite the following paper:
```bibtex
@misc{jin2024finetuninglinearlayerssimple,
      title={Fine-Tuning Linear Layers Only Is a Simple yet Effective Way for Task Arithmetic}, 
      author={Ruochen Jin and Bojian Hou and Jiancong Xiao and Weijie Su and Li Shen},
      year={2024},
      eprint={2407.07089},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.07089}, 
}
```



