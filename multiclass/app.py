import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from multiclass_perceptron import PerceptronMulticlass, generate_data

# Utilizamos st.session_state para almacenar los datos generados y persistirlos entre las interacciones
def main():
    st.title("Perceptrón Multiclase con Python")

    # Contenedor izquierdo para la generación de datos
    st.sidebar.header("Generación de Datos")
    n_samples = st.sidebar.slider("Número de muestras", 100, 1000, step=100)
    n_classes = st.sidebar.slider("Número de clases", 2, 5)
    generate_data_btn = st.sidebar.button("Generar Datos")

    # Si el botón para generar los datos se presiona, generamos los datos y los guardamos en la sesión
    if generate_data_btn:
        X, y = generate_data(n_samples=n_samples, n_classes=n_classes)
        st.session_state['X'] = X
        st.session_state['y'] = y
        st.session_state['n_classes'] = n_classes
        st.write("Datos Generados")
        fig, ax = plt.subplots()
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
        st.pyplot(fig)

    # Solo mostramos la opción de entrenar el modelo si ya se han generado los datos
    if 'X' in st.session_state and 'y' in st.session_state:
        st.sidebar.header("Entrenamiento del Perceptrón")
        learning_rate = st.sidebar.slider("Tasa de aprendizaje", 0.01, 1.0, step=0.01)
        n_iter = st.sidebar.slider("Número de iteraciones", 100, 1000, step=100)
        train_model_btn = st.sidebar.button("Entrenar Perceptrón")

        if train_model_btn:
            X = st.session_state['X']
            y = st.session_state['y']
            n_classes = st.session_state['n_classes']

            # Entrenamos el modelo
            model = PerceptronMulticlass(learning_rate=learning_rate, n_iter=n_iter, n_classes=n_classes)
            model.fit(X, y)
            st.write("Modelo entrenado. Visualizando las fronteras de decisión...")
            
            # Visualizamos las fronteras de decisión
            fig, ax = plt.subplots()
            model.plot_decision_boundary(X, y)
            st.pyplot(fig)

if __name__ == "__main__":
    main()
