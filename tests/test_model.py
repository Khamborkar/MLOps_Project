from src.model import model.py

test_loss, test_accuracy = model.evaluate(X_val,val_labels)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
