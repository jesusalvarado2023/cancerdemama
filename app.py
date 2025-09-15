import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# -------------------------------
# 1. Configuración de la página
# -------------------------------
st.title("Uso de la IA para detectar cáncer de mama")

# Problemática
st.subheader("Problemática")
st.write("""
El cáncer de mama es una de las principales causas de mortalidad en mujeres a nivel mundial.
La detección temprana mediante imágenes médicas, como ecografías, permite iniciar tratamientos oportunos que mejoran la supervivencia. 
Sin embargo, el diagnóstico puede ser complejo y requiere experiencia especializada. 
La Inteligencia Artificial (IA) aplicada en imágenes médicas emerge como una herramienta prometedora para apoyar a los profesionales de la salud, reduciendo errores y tiempos de diagnóstico. 
Esta investigación busca entrenar modelos de IA con el dataset **BreastMNIST**, con el fin de desarrollar sistemas automáticos de apoyo a la decisión clínica en el tamizaje del cáncer de mama.
""")

# Objetivo
st.subheader("Objetivo")
st.write("""
Desarrollar y evaluar un modelo de inteligencia artificial basado en imágenes médicas que permita detectar lesiones asociadas al cáncer de mama en ecografías, 
como herramienta complementaria para la toma de decisiones clínicas.
""")

# Metodología
st.subheader("Metodología")
st.write("""
1. Recolección y preprocesamiento del dataset **BreastMNIST**.  
2. Entrenamiento de un modelo de deep learning con redes convolucionales (CNN).  
3. Evaluación del desempeño mediante métricas (precisión, sensibilidad, especificidad).  
4. Implementación en una aplicación web interactiva con **Streamlit** para pruebas con imágenes nuevas.  
""")

# -------------------------------
# 2. Cargar el modelo entrenado
# -------------------------------
#@st.cache_resource
def load_model():
    filename = "./model_carol/breastcancer.pickle"
    model = pickle.load(open(filename, "rb"))
    #model = tf.keras.models.load_model("breast_cancer_model.h5")  # Asegúrate de que el archivo esté en la carpeta
    return model

model = load_model()

# -------------------------------
# 3. Subir imagen y preprocesar
# -------------------------------
st.subheader("Prueba tu ecografía")
uploaded_file = st.file_uploader("Sube tu ecografía de mama en formato JPG/PNG", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar imagen subida
    image = Image.open(uploaded_file).convert("L")  # convertir a escala de grises
    st.image(image, caption="Ecografía subida", use_column_width=True)

    # Preprocesamiento según BreastMNIST (28x28 en escala de grises)
    img_resized = image.resize((28, 28))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))  # shape: (1, 28, 28, 1)

    # -------------------------------
    # 4. Predicción con el modelo
    # -------------------------------
    prediction = model.predict(img_array)
    pred_class = (prediction > 0.5).astype("int")[0][0]

    # Mostrar resultado
    if pred_class == 1:
        st.error("⚠️ Posible hallazgo compatible con cáncer de mama.")
    else:
        st.success("✅ No se detectaron hallazgos compatibles con cáncer de mama.")
