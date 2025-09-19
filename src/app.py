import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="Predictor de Cáncer de Hígado",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar la apariencia
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 3rem;
    }
    
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        text-align: center;
    }
    
    .positive-prediction {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
    }
    
    .negative-prediction {
        background: linear-gradient(135deg, #00d2d3, #54a0ff);
        color: white;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 0.5rem;
    }
    
    .stSelectbox > div > div {
        background-color: #f8f9ff;
    }
    
    .stSlider > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Función para cargar el modelo (aquí debes cargar tu modelo real)
@st.cache_resource
def load_model():
    # Aquí cargarías tu modelo entrenado
    # return joblib.load('tu_modelo.pkl')
    # Por ahora uso un modelo simulado
    return None

# Función para hacer predicción (simulada)
def predict_cancer(features):
    """
    Esta función debe ser reemplazada por tu modelo real
    """
    # Simulación de predicción basada en algunos factores de riesgo
    risk_score = 0
    
    # Factores de riesgo principales
    if features['age'] > 60:
        risk_score += 0.3
    if features['alcohol_consumption'] > 7:
        risk_score += 0.2
    if features['smoking_status'] in [1, 2]:  # Fumador actual o ex-fumador
        risk_score += 0.25
    if features['hepatitis_b'] == 1:
        risk_score += 0.4
    if features['hepatitis_c'] == 1:
        risk_score += 0.45
    if features['cirrhosis_history'] == 1:
        risk_score += 0.5
    if features['family_history_cancer'] == 1:
        risk_score += 0.15
    if features['alpha_fetoprotein_level'] > 10:
        risk_score += 0.3
    if features['liver_function_score'] < 7:
        risk_score += 0.2
        
    # Añadir algo de aleatoriedad para simular un modelo real
    risk_score += np.random.normal(0, 0.1)
    probability = min(max(risk_score, 0), 1)
    
    return int(probability > 0.5), probability

# Encabezado principal
st.markdown('<h1 class="main-header">🏥 Predictor de Cáncer de Hígado</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Sistema inteligente para evaluación de riesgo oncológico</p>', unsafe_allow_html=True)

# Sidebar para entrada de datos
st.sidebar.header("📋 Datos del Paciente")

# Crear dos columnas para organizar mejor los inputs
col1, col2 = st.columns(2)

with st.sidebar:
    st.markdown("### 👤 Información Demográfica")
    
    age = st.slider("Edad", 18, 90, 45, help="Edad del paciente en años")
    gender = st.selectbox("Género", 
                         options=[0, 1], 
                         format_func=lambda x: "Masculino" if x == 1 else "Femenino",
                         help="Género del paciente")
    bmi = st.slider("IMC (Índice de Masa Corporal)", 15.0, 45.0, 25.0, 0.1,
                    help="Peso en kg / (altura en metros)²")
    
    st.markdown("### 🚬 Hábitos de Vida")
    
    alcohol_consumption = st.slider("Consumo de Alcohol (bebidas/semana)", 0, 20, 2,
                                   help="Número de bebidas alcohólicas por semana")
    
    smoking_status = st.selectbox("Estado de Fumador",
                                 options=[0, 1, 2],
                                 format_func=lambda x: ["No fumador", "Fumador actual", "Ex-fumador"][x],
                                 help="Historial de tabaquismo del paciente")
    
    physical_activity_level = st.slider("Nivel de Actividad Física", 0, 10, 5,
                                       help="Escala de 0-10, donde 10 es muy activo")
    
    st.markdown("### 🧬 Historial Médico")
    
    hepatitis_b = st.selectbox("Hepatitis B",
                              options=[0, 1],
                              format_func=lambda x: "No" if x == 0 else "Sí",
                              help="¿El paciente tiene hepatitis B?")
    
    hepatitis_c = st.selectbox("Hepatitis C",
                              options=[0, 1],
                              format_func=lambda x: "No" if x == 0 else "Sí",
                              help="¿El paciente tiene hepatitis C?")
    
    cirrhosis_history = st.selectbox("Historial de Cirrosis",
                                    options=[0, 1],
                                    format_func=lambda x: "No" if x == 0 else "Sí",
                                    help="¿El paciente tiene historial de cirrosis?")
    
    family_history_cancer = st.selectbox("Historial Familiar de Cáncer",
                                        options=[0, 1],
                                        format_func=lambda x: "No" if x == 0 else "Sí",
                                        help="¿Hay historial familiar de cáncer?")
    
    diabetes = st.selectbox("Diabetes",
                           options=[0, 1],
                           format_func=lambda x: "No" if x == 0 else "Sí",
                           help="¿El paciente tiene diabetes?")
    
    st.markdown("### 🔬 Análisis Clínicos")
    
    liver_function_score = st.slider("Puntuación de Función Hepática", 0.0, 15.0, 10.0, 0.1,
                                    help="Puntuación de función hepática (0-15)")
    
    alpha_fetoprotein_level = st.slider("Nivel de Alfa-fetoproteína (ng/mL)", 0.0, 100.0, 5.0, 0.1,
                                       help="Nivel de AFP en sangre")

# Crear el diccionario de características
features = {
    'age': age,
    'gender': gender,
    'bmi': bmi,
    'alcohol_consumption': alcohol_consumption,
    'smoking_status': smoking_status,
    'hepatitis_b': hepatitis_b,
    'hepatitis_c': hepatitis_c,
    'liver_function_score': liver_function_score,
    'alpha_fetoprotein_level': alpha_fetoprotein_level,
    'cirrhosis_history': cirrhosis_history,
    'family_history_cancer': family_history_cancer,
    'physical_activity_level': physical_activity_level,
    'diabetes': diabetes
}

# Botón de predicción
if st.sidebar.button("🔍 Realizar Predicción", type="primary", use_container_width=True):
    with st.spinner('Analizando datos del paciente...'):
        # Hacer predicción
        prediction, probability = predict_cancer(features)
        
        # Mostrar resultado
        if prediction == 1:
            st.markdown(f"""
            <div class="prediction-box positive-prediction">
                <h2>⚠️ RIESGO ALTO</h2>
                <h3>Probabilidad de cáncer: {probability:.1%}</h3>
                <p>Se recomienda consulta médica inmediata y estudios adicionales</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-box negative-prediction">
                <h2>✅ RIESGO BAJO</h2>
                <h3>Probabilidad de cáncer: {probability:.1%}</h3>
                <p>Continúe con controles médicos regulares</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Mostrar métricas adicionales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Edad", f"{age} años")
        with col2:
            st.metric("IMC", f"{bmi:.1f}")
        with col3:
            st.metric("Función Hepática", f"{liver_function_score:.1f}/15")
        with col4:
            st.metric("AFP", f"{alpha_fetoprotein_level:.1f} ng/mL")
        
        # Gráfico de factores de riesgo
        st.subheader("📊 Análisis de Factores de Riesgo")
        
        risk_factors = {
            'Edad': min(age/90, 1),
            'Alcohol': min(alcohol_consumption/20, 1),
            'Tabaquismo': smoking_status/2,
            'Hepatitis B': hepatitis_b,
            'Hepatitis C': hepatitis_c,
            'Cirrosis': cirrhosis_history,
            'Historial Familiar': family_history_cancer,
            'AFP': min(alpha_fetoprotein_level/100, 1),
            'Función Hepática': 1 - (liver_function_score/15)
        }
        
        fig = px.bar(
            x=list(risk_factors.keys()),
            y=list(risk_factors.values()),
            title="Nivel de Riesgo por Factor",
            color=list(risk_factors.values()),
            color_continuous_scale="RdYlBu_r"
        )
        fig.update_layout(
            xaxis_title="Factores de Riesgo",
            yaxis_title="Nivel de Riesgo (0-1)",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Gráfico de probabilidad
        fig2 = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Probabilidad de Cáncer (%)"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred" if probability > 0.5 else "darkgreen"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgreen"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        st.plotly_chart(fig2, use_container_width=True)

# Información adicional
st.markdown("---")
st.subheader("ℹ️ Información Importante")

col1, col2 = st.columns(2)

with col1:
    st.info("""
    **Factores de Riesgo Principales:**
    - Hepatitis B y C
    - Cirrosis hepática
    - Consumo excesivo de alcohol
    - Edad avanzada
    - Historial familiar de cáncer
    """)

with col2:
    st.warning("""
    **Limitaciones del Modelo:**
    - Este es un sistema de apoyo al diagnóstico
    - No reemplaza la evaluación médica profesional
    - Los resultados deben interpretarse junto con otros estudios clínicos
    - Siempre consulte con un especialista
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🏥 Sistema de Predicción de Cáncer de Hígado | Desarrollado con Streamlit</p>
    <p><small>Última actualización: {}</small></p>
</div>
""".format(datetime.now().strftime("%d/%m/%Y")), unsafe_allow_html=True)