import matplotlib.pyplot as plt
import pickle
plt.style.use("seaborn")
# loss
# Import loss and val_loss history from train.py
with open("token_history", "rb") as f:
    _, loss, val_loss = pickle.load(f)

print("Train loss: ", loss[len(loss) - 1])
print("Validation loss: ", val_loss[len(val_loss) - 1])

plt.plot(loss, color='blue')
plt.plot(val_loss, color='red')
plt.title('Loss Curves')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()