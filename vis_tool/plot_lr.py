import matplotlib.pyplot as plt
import numpy as np
# from math import round

def learning_rate_schedule(t, ti, alpha_i_0):
    max_iterations_i = total_iterations*ti
    lr_values = np.zeros_like(t)

    # Warmup phase for the first 10 iterations
    warmup_iterations = 10
    warmup_rate = alpha_i_0 / warmup_iterations
    for i in range(warmup_iterations):
        lr_values[i] = warmup_rate * (i )

    # Adjusted cosine schedule starting from iteration 10
    adjusted_t = t - warmup_iterations
    adjusted_t[adjusted_t < 0] = 0  # Ensure no negative values
    lr_values[warmup_iterations:] += 0.5 * alpha_i_0 * (1 + np.cos(np.pi * adjusted_t[warmup_iterations:] / (max_iterations_i - warmup_iterations)))
    lr_values[round(max_iterations_i):] = 0

    return lr_values

# Parameters
num_layers = 13
t0 = 0.8
total_iterations = 100  # Assumed total number of iterations for plotting

# Generate layer-wise learning rate schedules
t_values = np.linspace(0, total_iterations, 100)  # 500 points for smooth curve
layer_schedules = []

t0_cubed = t0**3 # cubic scale.
alpha_0_0 = 1 / t0_cubed  # Initial learning rate scaled
lr_values = learning_rate_schedule(t_values, t0_cubed, alpha_0_0)
layer_schedules.append(lr_values)

for i in range(1,num_layers):
    ti = t0 + (i / (num_layers - 1)) * (1 - t0)  # Linear spacing for ti
    ti_cubed = ti**3 # cubic scale.
    alpha_i_0 = 1 / ti_cubed  # Initial learning rate scaled
    lr_values = learning_rate_schedule(t_values, ti_cubed, alpha_i_0)
    layer_schedules.append(lr_values)

# Plotting
plt.figure(figsize=(10, 6))
for i, lr_values in enumerate(layer_schedules):
    plt.plot(t_values, lr_values / layer_schedules[-1][10] * 100, label=f'Layer {i+1}')

plt.xlabel('% Into Training')
plt.ylabel('% Of Initial Learning Rate')
plt.title('Layer-wise Learning Rate Schedules with Cosine Annealing')
plt.legend()
plt.grid(True)
plt.show()

# Function to calculate the learning rate according to the cosine schedule with warmup