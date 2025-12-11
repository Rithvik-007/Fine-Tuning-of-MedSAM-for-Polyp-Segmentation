# src/plot_training_curve.py

import matplotlib.pyplot as plt

def main():
    # Manually enter or load from log file
    val_dice_values = [
        0.70, 0.74, 0.77, 0.79, 0.81, 0.83, 0.84, 0.864  # example
    ]
    
    epochs = list(range(1, len(val_dice_values)+1))

    plt.figure(figsize=(7,5))
    plt.plot(epochs, val_dice_values, marker="o", color="#4C72B0", linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Validation Dice")
    plt.title("Training Progress – Scratch Model")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    plt.savefig("work_dir/training_curve.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
