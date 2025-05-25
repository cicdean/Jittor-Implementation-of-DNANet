import matplotlib.pyplot as plt
save_dir="NUDT-SIRST_DNANet_24_05_2025_20_16_48_wDS"

loss_list = []
with open(f"result_jt\\{save_dir}\\loss.txt", 'r') as file:
    for line in file:
        loss_list.append(float(line.strip()))

epochs = list(range(1, len(loss_list) + 1))
plt.plot(epochs, loss_list, color='red', marker='.', label='Training Loss')

plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.show()
