import matplotlib.pyplot as plt

epoch_list = []
loss_list = []
accuracy_list = []

with open("pytorch_results.txt", "r") as f:
    lines = f.readlines()

for line in lines[3:]:
    parts = line.strip().split('\t')
    if len(parts) == 3:
        epoch_list.append(int(parts[0]))
        loss_list.append(float(parts[1]))
        accuracy_list.append(float(parts[2]))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epoch_list, loss_list, marker='o', linestyle='-')
plt.title("Training Loss per  Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(epoch_list, accuracy_list, marker='o', linestyle='-')
plt.title("Test Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.tight_layout()
plt.savefig("visualization.png")
plt.show()
