import matplotlib.pyplot as plt

# plot acc and loss history
def plot_history(history):
    'plots training history'
    fig,ax = plt.subplots(1,2,figsize=(15,5))
    ax[0].plot(history['train_acc'], label='train_acc')
    ax[0].plot(history['val_acc'], label='val_acc')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_title('Training History')
    ax[0].legend()
    ax[1].plot(history['train_loss'], label='train_loss')
    ax[1].plot(history['val_loss'], label='val_loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].set_title('Training History')
    ax[1].legend()
    plt.show()


if __name__ == '__main__':
    history = {'train_loss': [0.5, 0.4, 0.3, 0.2, 0.1],
                'val_loss': [0.6, 0.5, 0.4, 0.3, 0.2],
                'train_acc': [0.1, 0.2, 0.3, 0.4, 0.5],
                'val_acc': [0.2, 0.3, 0.4, 0.5, 0.6]}
    plot_history(history)
