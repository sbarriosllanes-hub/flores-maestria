
import streamlit as st
import joblib
import pandas as pd
import os

# --- Page Configuration (UX Improvement) ---
st.set_page_config(layout='wide', page_title='Iris Species Predictor', page_icon='🌸')

# --- Load Models (Error Handling & Cache for performance) ---
@st.cache_resource
def load_models():
    try:
        best_knn_model = joblib.load('best_knn_model.pkl')
        label_encoder_species = joblib.load('label_encoder_species.pkl')
        return best_knn_model, label_encoder_species
    except FileNotFoundError:
        st.error("Error: Model files 'best_knn_model.pkl' or 'label_encoder_species.pkl' not found.")
        st.stop()

best_knn_model, label_encoder_species = load_models()

# --- Header and Introduction (UX Improvement) ---
st.title('🌸 Clasificador de Especies de Iris')
st.markdown('### Introduce las medidas de la flor de Iris para predecir su especie.')
st.info('Utiliza los controles deslizantes para ajustar los valores de los sépalos y pétalos.')

# --- Input Section (UX Improvement) ---
with st.container():
    st.subheader('Medidas de la Flor (cm)')
    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.slider('Longitud del Sépalo', 4.0, 8.0, 5.0, 0.1, help='Longitud del sépalo en centímetros.')
        sepal_width = st.slider('Ancho del Sépalo', 2.0, 4.5, 3.0, 0.1, help='Ancho del sépalo en centímetros.')
    with col2:
        petal_length = st.slider('Longitud del Pétalo', 1.0, 7.0, 4.0, 0.1, help='Longitud del pétalo en centímetros.')
        petal_width = st.slider('Ancho del Pétalo', 0.1, 2.5, 1.0, 0.1, help='Ancho del pétalo en centímetros.')

# --- Prediction Button (UX Improvement) ---
if st.button('🚀 Predecir Especie', help='Haz clic para obtener la predicción de la especie de Iris.'):
    # Create a DataFrame for the new flower's characteristics
    # Ensure feature names and order match the model's training data
    try:
        expected_feature_names = best_knn_model.feature_names_in_
    except AttributeError:
        # Fallback if model doesn't have this attribute (older scikit-learn versions)
        expected_feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

    # Create a dictionary with user inputs, ensuring correct keys
    user_input_values_map = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }

    # Create a list of values in the correct order based on the model's expected_feature_names
    new_flower_values_ordered = [user_input_values_map[name] for name in expected_feature_names]

    # Create the DataFrame with the correct feature names and order
    new_flower_df = pd.DataFrame([new_flower_values_ordered], columns=expected_feature_names)

    # Display input data (for verification/debugging - UX Improvement)
    st.markdown('---')
    st.subheader('Valores de Entrada:')
    st.dataframe(new_flower_df.style.highlight_max(axis=0), use_container_width=True)

    # Make a prediction
    predicted_species_encoded = best_knn_model.predict(new_flower_df)
    predicted_species = label_encoder_species.inverse_transform(predicted_species_encoded)

    # Display prediction result (UX Improvement)
    st.success(f'✨ La especie de Iris predicha es: **{predicted_species[0].upper()}**')
    st.balloons()

# --- Model Info (UX Improvement) ---
with st.expander('🔍 Información del Modelo (Para desarrolladores)'):
    st.write('Este clasificador utiliza un modelo K-Nearest Neighbors (KNN).')
    st.write('**Características que el modelo espera (y su orden):**')
    try:
        st.write(best_knn_model.feature_names_in_)
    except AttributeError:
        st.write("['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'] (orden asumido)")
    st.write('El modelo y el codificador de etiquetas fueron cargados usando `joblib`.')

