import matplotlib.pyplot as plt


colors = ['blue', 'orange', 'green']
models = ['ViT-B-32', 'ViT-B-16', 'ViT-L-14']

data_updated = {
    "Non-linear": {
        # "ViT-B-32": [0.7839, 0.792, 0.9904, 0.9921, 0.9967, 0.9598, 0.9735, 0.7505],
        "ViT-B-32": [0.9048625,	0.9035],
        # "ViT-B-16": [0.8712, 0.816, 0.9878, 0.989, 0.9967, 0.9679, 0.9791, 0.7866],
        "ViT-B-16": [0.9242875,	0.922675],
        # "ViT-L-14": [0.923, 0.8431, 0.9919, 0.9918, 0.9977, 0.9741, 0.9816, 0.8216]
        "ViT-L-14":[0.9406,	0.941675]
    },
    "Ours": {
        # "ViT-B-32": [0.7116, 0.7878, 0.9844, 0.9797, 0.9966, 0.9463, 0.9631, 0.7465],
        "ViT-B-32":[0.8954875,	0.8948125],
        # "ViT-B-16": [0.844, 0.8186, 0.9907, 0.9889, 0.9963, 0.9644, 0.9757, 0.7747],
        "ViT-B-16":[0.9191625,	0.9164375],
        # "ViT-L-14": [0.9133, 0.8309, 0.9893, 0.9895, 0.9976, 0.9721, 0.9794, 0.8197]
        "ViT-L-14":[0.936475,	0.9389625]
    }
}

linear_non_linear_data = {
    "ViT-B-32": {
        # "Linear": [0.7432, 0.7862, 0.9889, 0.984, 0.9968, 0.9506, 0.9672, 0.747],
        "Linear": [0.8895,	0.8885],
        # "Non-linear": [0.7839, 0.792, 0.9904, 0.9921, 0.9967, 0.9598, 0.9735, 0.7505]
        "Non-linear": [0.9048625,	0.9035]
    },
    "ViT-B-16": {
        # "Linear": [0.8188, 0.8165, 0.9896, 0.9864, 0.9957, 0.9592, 0.973, 0.7695],
        "Linear": [0.9135875,	0.9097375],
        # "Non-linear": [0.8712, 0.816, 0.9878, 0.989, 0.9967, 0.9679, 0.9791, 0.7866]
        "Non-linear":[0.9242875,	0.922675]
    },
    "ViT-L-14": {
        # "Linear": [0.896, 0.8324, 0.9896, 0.9899, 0.9978, 0.9692, 0.9771, 0.8152],
        "Linear": [0.9334,	0.9326],
        # "Non-linear": [0.923, 0.8431, 0.9919, 0.9918, 0.9977, 0.9741, 0.9816, 0.8216]
        "Non-linear":[0.9406,	0.941675]
    }
}
# Recreating the combined chart with doubled font sizes and without a title

# Set the font size for the plot
plt.rcParams.update({'font.size': 20})  # Default is around 10, so set to 20 for double size

# Creating the figure and axis for the revised chart
fig, ax = plt.subplots(figsize=(10, 8))

# Plotting Non-linear vs Ours comparison with Ours on the y-axis now
for model, color in zip(models, colors):
    ax.scatter(data_updated['Non-linear'][model], data_updated['Ours'][model], color=color, label=f'Ours vs. Non-linear ({model})', marker='o')

# Plotting Linear vs Non-linear comparison with Linear on the y-axis now
for model, color in zip(models, colors):
    ax.scatter(linear_non_linear_data[model]['Non-linear'], linear_non_linear_data[model]['Linear'], color=color, label=f'NTK Linearization vs. Non-linear ({model})', marker='^')

# Adding a diagonal line for reference
x_values = [0.7, 1.0]  # Extending the range a bit more for clarity
ax.plot(x_values, x_values, 'k--', label='Equal Accuracy Line')

# Adjusting labels with increased font size
ax.set_xlabel('Non-linear Finetuning Accuracy(%)')
ax.set_ylabel('Linear Finetuning Accuracy(%)')
ax.legend()

# Show plot
plt.savefig('single-task performance.pdf',dpi=300)
