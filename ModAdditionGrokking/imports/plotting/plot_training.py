import os

import matplotlib.pyplot as plt


def initialise_plot():
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    acc_train_line, = ax1.plot([], [], color='blue', label='Train Accuracy')
    acc_test_line, = ax1.plot([], [], color='red', label='Test Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.set_title('Accuracy over Epochs')
    ax1.set_ylim([0, 1])  # Fix accuracy ylim

    loss_train_line, = ax2.plot([], [], color='blue', label='Train Log Loss')
    loss_test_line, = ax2.plot([], [], color='red', label='Test Log Loss')
    ax2.set_ylabel('Log Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.set_title('Log Loss over Epochs')

    plt.tight_layout()
    fig.show()  # <-- Use fig.show() instead of plt.show()
    return fig, (ax1, ax2), (acc_train_line, acc_test_line, loss_train_line, loss_test_line)

def update_plot(acc_train_history, acc_test_history, loss_train_history, loss_test_history, lines):
    acc_train_line, acc_test_line, loss_train_line, loss_test_line = lines
    epochs = range(len(acc_train_history))
    acc_train_line.set_data(epochs, acc_train_history)
    acc_test_line.set_data(epochs, acc_test_history)
    loss_train_line.set_data(epochs, loss_train_history)
    loss_test_line.set_data(epochs, loss_test_history)

    # Update xlim for both axes
    for ax in [acc_train_line.axes, loss_train_line.axes]:
        ax.set_xlim([0, max(1, len(acc_train_history))])

    # Update ylim for loss dynamically
    if loss_train_history and loss_test_history:
        all_loss = loss_train_history + loss_test_history
        min_loss = min(all_loss)
        max_loss = max(all_loss)
        loss_train_line.axes.set_ylim([min_loss - 0.1 * abs(min_loss), max_loss + 0.1 * abs(max_loss)])

    acc_train_line.figure.canvas.draw()
    acc_train_line.figure.canvas.flush_events()


def plot_final_histories(acc_train_history, acc_test_history, loss_train_history, loss_test_history):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    epochs = range(len(acc_train_history))

    ax1.plot(epochs, acc_train_history, color='blue', label='Train Accuracy')
    ax1.plot(epochs, acc_test_history, color='red', label='Test Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.set_title('Accuracy over Epochs')
    ax1.set_ylim([0, 1])
    ax1.set_xlim([0, max(1, len(acc_train_history))])

    ax2.plot(epochs, loss_train_history, color='blue', label='Train Log Loss')
    ax2.plot(epochs, loss_test_history, color='red', label='Test Log Loss')
    ax2.set_ylabel('Log Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.set_title('Log Loss over Epochs')
    if loss_train_history and loss_test_history:
        all_loss = loss_train_history + loss_test_history
        min_loss = min(all_loss)
        max_loss = max(all_loss)
        ax2.set_ylim([min_loss - 0.1 * abs(min_loss), max_loss + 0.1 * abs(max_loss)])
    ax2.set_xlim([0, max(1, len(loss_train_history))])
    
    plt.tight_layout()
    plt.show()

    return fig