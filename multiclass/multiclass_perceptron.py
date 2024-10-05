import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class PerceptronMulticlass:
    def __init__(self, learning_rate=0.01, n_iter=1000, n_classes=3):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_classes = n_classes
        self.W = None
        self.b = None

    def activation(self, x):
        """Función de activación escalón."""
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        """Entrenamiento del perceptrón multiclase."""
        n_samples, n_features = X.shape
        self.W = np.zeros((self.n_classes, n_features))
        self.b = np.zeros(self.n_classes)

        for _ in range(self.n_iter):
            for idx, x_i in enumerate(X):
                y_true = np.zeros(self.n_classes)
                y_true[y[idx]] = 1

                for j in range(self.n_classes):
                    linear_output = np.dot(x_i, self.W[j]) + self.b[j]
                    y_pred = self.activation(linear_output)

                    if y_pred != y_true[j]:
                        self.W[j] += self.learning_rate * (y_true[j] - y_pred) * x_i
                        self.b[j] += self.learning_rate * (y_true[j] - y_pred)

    def predict(self, X):
        """Predicción de nuevas muestras."""
        y_pred = []
        for x_i in X:
            linear_output = np.dot(x_i, self.W.T) + self.b
            y_pred.append(np.argmax(linear_output))
        return np.array(y_pred)

    def plot_decision_boundary(self, X, y):
        """Visualización de las fronteras de decisión."""
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        grid = np.c_[xx.ravel(), yy.ravel()]
        predictions = self.predict(grid).reshape(xx.shape)

        plt.contourf(xx, yy, predictions, alpha=0.6, cmap='viridis')
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
        plt.show()

# Función para generar los datos de muestra
def generate_data(n_samples=300, n_classes=3):
    X, y = make_blobs(n_samples=n_samples, centers=n_classes, n_features=2, random_state=42)
    return X, y

# Función para ejecutar el modelo y visualizar la frontera
def run_model(n_samples=300, n_classes=3, learning_rate=0.01, n_iter=1000):
    X, y = generate_data(n_samples=n_samples, n_classes=n_classes)
    
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
    plt.title("Datos Generados")
    plt.show()

    model = PerceptronMulticlass(learning_rate=learning_rate, n_iter=n_iter, n_classes=n_classes)
    model.fit(X, y)
    
    print("Modelo entrenado. Visualizando las fronteras de decisión...")
    model.plot_decision_boundary(X, y)
