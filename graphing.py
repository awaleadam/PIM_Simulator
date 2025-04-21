import numpy as np
import matplotlib.pyplot as plt

# Data
samsung_costs = [9833448380.0, 15276218382.0, 1275242300, 20770952316, 1507360028, 1171114918398.0, 36307479376.0]
model_samsung_costs = [8679049315.38889, 3927628206.0, 303158768.0, 5069068104.0, 571486008.0, 297594251250.0, 1298433680.5]
sk_costs = [11536395680.64865, 4303087308.0, 315368920.0, 5726719160.0, 556789896.0, 309459234111.60394, 10871790104.0]
model_sk_costs = [9977858163.135136, 4279107852.0, 314744760.0, 5324345974.0, 434604936.0, 308775239583.5999, 1324763224.0]

# X-axis labels
models = ['DORN', 'GPT2', 'LSTM', 'RNN', 'StarGAN', 'ViT', 'ResNet']
x = np.arange(len(models))  # X-axis positions

# Bar width
bar_width = 0.2

# Plotting
plt.figure(figsize=(12, 6))
plt.bar(x - 1.5 * bar_width, samsung_costs, width=bar_width, label='SAMSUNG COSTS')
plt.bar(x - 0.5 * bar_width, model_samsung_costs, width=bar_width, label='MODEL Samsung COSTS')
plt.bar(x + 0.5 * bar_width, sk_costs, width=bar_width, label='SK COSTS')
plt.bar(x + 1.5 * bar_width, model_sk_costs, width=bar_width, label='MODEL SK COSTS')

# Adding labels and title
plt.xlabel('Models', fontsize=14)
plt.ylabel('Costs (Log Scale)', fontsize=14)
plt.title('Comparison of Costs', fontsize=16)
plt.xticks(x, models, fontsize=12)
plt.yscale('log')  # Set y-axis to log scale
plt.legend(fontsize=12)
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()