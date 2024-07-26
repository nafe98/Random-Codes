%matplotlib inline
import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class PCAPlotter(tf.keras.callbacks.Callback):
    
    def __init__(self, embedding_model, x_test, y_test):
        super(PCAPlotter, self).__init__()
        self.embedding_model = embedding_model
        self.x_test = x_test
        self.y_test = y_test
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(9, 4))
        plt.ion()
        self.losses = []
    
    def plot(self, epoch=None, plot_loss=False):
        x_test_embeddings = self.embedding_model.predict(self.x_test)
        pca_out = PCA(n_components=2).fit_transform(x_test_embeddings)
        self.ax1.clear()
        scatter = self.ax1.scatter(pca_out[:, 0], pca_out[:, 1], c=self.y_test, cmap='seismic')
        self.ax1.set_title('PCA of Test Set Embeddings')
        legend1 = self.ax1.legend(*scatter.legend_elements(), title="Classes")
        self.ax1.add_artist(legend1)
        if plot_loss:
            self.ax2.clear()
            self.ax2.plot(range(1, epoch + 1), self.losses)
            self.ax2.set_xlabel('Epochs')
            self.ax2.set_ylabel('Loss')
            self.ax2.set_title('Training Loss Over Epochs')
        self.fig.canvas.draw()
    
    def on_train_begin(self, logs=None):
        self.losses = []
        self.fig.show()
        self.plot()
        
    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.plot(epoch + 1, plot_loss=True)