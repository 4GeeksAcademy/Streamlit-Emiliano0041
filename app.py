import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

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
        font-size: 1.2rem;
    }
    
    .positive-prediction {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        border: 3px solid #ff4757;
    }
    
    .negative-prediction {
        background: linear-gradient(135deg, #00d2d3, #54a0ff);
        color: white;
        border: 3px solid #0fbcf9;
    }
    
    .risk-factor-high {
        background-color: #ffebee;
        color: #c62828;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem;
        border-left: 4px solid #f44336;
    }
    
    .risk-factor-medium {
        background-color: #fff3e0;
        color: #ef6c00;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem;
        border-left: 4px solid #ff9800;
    }
    
    .risk-factor-low {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem;
        border-left: 4px solid #4caf50;
    }
    
    .probability-meter {
        background: #f0f0f0;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        text-align: center;
    }
    
    .progress-bar {
        background: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
        height: 30px;
        margin: 10px 0;
    }
    
    .progress-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def predict_cancer(features):
    model = load_model()
    # Convierte a DataFrame con el orden correcto de columnas
    feature_order = ['age', 'gender', 'bmi', 'alcohol_consumption', 'smoking_status', 
                    'hepatitis_b', 'hepatitis_c', 'liver_function_score', 
                    'alpha_fetoprotein_level', 'cirrhosis_history', 
                    'family_history_cancer', 'physical_activity_level', 'diabetes']
    
    df = pd.DataFrame([features])[feature_order]
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    
    # Mantén los risk_weights para el análisis visual
    risk_weights = {...}  # Tu lógica de factores de riesgo
    
    return prediction, probability, risk_weights

def create_progress_bar(value, max_value=1, color=""):
    """Crear una barra de progreso HTML"""
    percentage = (value / max_value) * 100
    
    if percentage < 25:
        bar_color = "#4caf50"  # Verde
    elif percentage < 50:
        bar_color = "#ff9800"  # Naranja
    elif percentage < 75:
        bar_color = "#ff5722"  # Rojo naranja
    else:
        bar_color = "#f44336"  # Rojo
    
    return f"""
    <div class="progress-bar">
        <div class="progress-fill" style="width: {percentage}%; background-color: {bar_color};">
            {percentage:.1f}%
        </div>
    </div>
    """

# Encabezado principal
st.markdown('<h1 class="main-header">🏥 Predictor de Cáncer de Hígado</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Sistema inteligente para evaluación de riesgo oncológico</p>', unsafe_allow_html=True)

# Sidebar para entrada de datos
st.sidebar.header("📋 Datos del Paciente")

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
        prediction, probability, risk_weights = predict_cancer(features)
        
        # Mostrar resultado principal
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
        
        # Medidor de probabilidad
        st.markdown("### 📊 Medidor de Probabilidad")
        st.markdown(f"""
        <div class="probability-meter">
            <h4>Probabilidad de Cáncer: {probability:.1%}</h4>
            {create_progress_bar(probability)}
        </div>
        """, unsafe_allow_html=True)
        
        # Mostrar métricas clave
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Edad", f"{age} años", 
                     delta="Factor de riesgo" if age > 60 else "Normal")
        with col2:
            st.metric("IMC", f"{bmi:.1f}", 
                     delta="Sobrepeso" if bmi > 25 else "Normal")
        with col3:
            st.metric("Función Hepática", f"{liver_function_score:.1f}/15",
                     delta="Bajo" if liver_function_score < 8 else "Normal")
        with col4:
            st.metric("AFP", f"{alpha_fetoprotein_level:.1f} ng/mL",
                     delta="Elevado" if alpha_fetoprotein_level > 10 else "Normal")
        
        # Análisis detallado de factores de riesgo
        st.markdown("### 🎯 Análisis de Factores de Riesgo")
        
        # Crear dos columnas para mostrar los factores
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Factores de Alto Riesgo")
            high_risk_factors = []
            
            if risk_weights['hepatitis_b_risk'] > 0:
                high_risk_factors.append("🦠 Hepatitis B positiva")
            if risk_weights['hepatitis_c_risk'] > 0:
                high_risk_factors.append("🦠 Hepatitis C positiva")
            if risk_weights['cirrhosis_risk'] > 0:
                high_risk_factors.append("🔴 Historial de cirrosis")
            if risk_weights['afp_risk'] > 0.3:
                high_risk_factors.append("📈 AFP muy elevada")
            if risk_weights['age_risk'] > 0.2:
                high_risk_factors.append("📅 Edad avanzada")
            if risk_weights['smoking_risk'] > 0.25:
                high_risk_factors.append("🚬 Fumador activo")
                
            if high_risk_factors:
                for factor in high_risk_factors:
                    st.markdown(f'<div class="risk-factor-high">{factor}</div>', unsafe_allow_html=True)
            else:
                st.success("✅ No se detectaron factores de alto riesgo")
        
        with col2:
            st.markdown("#### Factores de Riesgo Moderado")
            medium_risk_factors = []
            
            if risk_weights['alcohol_risk'] > 0:
                medium_risk_factors.append("🍺 Consumo de alcohol")
            if risk_weights['family_risk'] > 0:
                medium_risk_factors.append("👨‍👩‍👧‍👦 Historial familiar")
            if risk_weights['liver_function_risk'] > 0:
                medium_risk_factors.append("🧪 Función hepática alterada")
            if risk_weights['diabetes_risk'] > 0:
                medium_risk_factors.append("🩺 Diabetes")
            if risk_weights['bmi_risk'] > 0:
                medium_risk_factors.append("⚖️ Sobrepeso")
            if risk_weights['activity_risk'] > 0:
                medium_risk_factors.append("🏃‍♂️ Poca actividad física")
                
            if medium_risk_factors:
                for factor in medium_risk_factors:
                    st.markdown(f'<div class="risk-factor-medium">{factor}</div>', unsafe_allow_html=True)
            else:
                st.success("✅ No se detectaron factores de riesgo moderado")
        
        # Gráfico de barras simple con matplotlib
        st.markdown("### 📈 Distribución de Factores de Riesgo")
        
        # Preparar datos para el gráfico
        factor_names = [
            'Edad', 'Alcohol', 'Tabaco', 'Hep. B', 'Hep. C', 
            'Cirrosis', 'Fam. Cancer', 'AFP', 'Func. Hígado', 'Diabetes'
        ]
        factor_values = [
            risk_weights['age_risk'],
            risk_weights['alcohol_risk'],
            risk_weights['smoking_risk'],
            risk_weights['hepatitis_b_risk'],
            risk_weights['hepatitis_c_risk'],
            risk_weights['cirrhosis_risk'],
            risk_weights['family_risk'],
            risk_weights['afp_risk'],
            risk_weights['liver_function_risk'],
            risk_weights['diabetes_risk']
        ]
        
        # Crear gráfico con matplotlib
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = ['#ff4757' if v > 0.3 else '#ff6348' if v > 0.15 else '#26de81' for v in factor_values]
        bars = ax.bar(factor_names, factor_values, color=colors, alpha=0.8)
        
        ax.set_ylabel('Nivel de Riesgo')
        ax.set_title('Análisis Individual de Factores de Riesgo')
        ax.set_ylim(0, max(0.6, max(factor_values) + 0.1))
        
        # Añadir líneas de referencia
        ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='Riesgo Alto')
        ax.axhline(y=0.15, color='orange', linestyle='--', alpha=0.7, label='Riesgo Moderado')
        
        # Rotar etiquetas del eje x
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Recomendaciones personalizadas
        st.markdown("### 💡 Recomendaciones Personalizadas")
        
        recommendations = []
        
        if risk_weights['alcohol_risk'] > 0:
            recommendations.append("🍺 Considere reducir el consumo de alcohol")
        if risk_weights['smoking_risk'] > 0:
            recommendations.append("🚭 Se recomienda encarecidamente dejar de fumar")
        if risk_weights['activity_risk'] > 0:
            recommendations.append("🏃‍♂️ Aumentar la actividad física regular")
        if risk_weights['bmi_risk'] > 0:
            recommendations.append("⚖️ Mantener un peso saludable")
        if any([risk_weights['hepatitis_b_risk'], risk_weights['hepatitis_c_risk'], risk_weights['cirrhosis_risk']]):
            recommendations.append("🏥 Control médico especializado frecuente")
        if risk_weights['afp_risk'] > 0:
            recommendations.append("🔬 Monitoreo regular de marcadores tumorales")
        
        recommendations.append("📅 Controles médicos regulares cada 6-12 meses")
        recommendations.append("🥗 Mantener una dieta saludable y equilibrada")
        
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")

# Información adicional sin predicción
else:
    # Mostrar información general
    st.markdown("## 👈 Complete los datos del paciente en la barra lateral")
    st.markdown("Una vez completados todos los campos, presione **'Realizar Predicción'** para obtener el análisis.")
    
    # Información educativa
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **📊 Factores de Riesgo Principales:**
        - 🦠 Hepatitis B y C crónicas
        - 🔴 Cirrosis hepática
        - 🍺 Consumo excesivo de alcohol
        - 🚬 Tabaquismo
        - 📅 Edad avanzada (>60 años)
        - 👨‍👩‍👧‍👦 Historial familiar de cáncer
        - 📈 Niveles elevados de AFP
        - 🩺 Diabetes mellitus
        """)
    
    with col2:
        st.warning("""
        **⚠️ Limitaciones Importantes:**
        - 🤖 Sistema de apoyo al diagnóstico únicamente
        - 👨‍⚕️ No reemplaza la evaluación médica profesional
        - 🔬 Los resultados requieren interpretación clínica
        - 📋 Debe combinarse con otros estudios diagnósticos
        - 🏥 Siempre consulte con un especialista en hepatología
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🏥 Sistema de Predicción de Cáncer de Hígado | Desarrollado con Streamlit</p>
    <p><small>Última actualización: {}</small></p>
    <p><small>⚠️ Solo para uso educativo y de apoyo clínico</small></p>
</div>
""".format(datetime.now().strftime("%d/%m/%Y")), unsafe_allow_html=True)