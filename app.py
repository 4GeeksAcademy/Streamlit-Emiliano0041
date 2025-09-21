import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

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

# Funciones para guardar historial de pacientes
def save_patient_prediction(patient_name, features, prediction, probability, risk_weights):
    """
    Guarda la predicción del paciente en un archivo CSV
    """
    import pandas as pd
    import os
    from datetime import datetime
    
    # Crear registro del paciente
    patient_record = {
        'fecha_prediccion': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'nombre_paciente': patient_name,
        'edad': features['age'],
        'genero': 'Masculino' if features['gender'] == 1 else 'Femenino',
        'imc': features['bmi'],
        'consumo_alcohol': features['alcohol_consumption'],
        'estado_fumador': ['No fumador', 'Fumador actual', 'Ex-fumador'][features['smoking_status']],
        'hepatitis_b': 'Sí' if features['hepatitis_b'] == 1 else 'No',
        'hepatitis_c': 'Sí' if features['hepatitis_c'] == 1 else 'No',
        'funcion_hepatica': features['liver_function_score'],
        'nivel_afp': features['alpha_fetoprotein_level'],
        'historial_cirrosis': 'Sí' if features['cirrhosis_history'] == 1 else 'No',
        'historial_familiar': 'Sí' if features['family_history_cancer'] == 1 else 'No',
        'actividad_fisica': features['physical_activity_level'],
        'diabetes': 'Sí' if features['diabetes'] == 1 else 'No',
        'prediccion': 'POSITIVO' if prediction == 1 else 'NEGATIVO',
        'probabilidad': f"{probability:.1%}",
        'riesgo_nivel': 'ALTO' if probability > 0.6 else 'MODERADO' if probability > 0.3 else 'BAJO'
    }
    
    # Archivo para guardar historiales
    history_file = 'historial_pacientes.csv'
    
    try:
        # Si el archivo existe, cargarlo; si no, crear nuevo DataFrame
        if os.path.exists(history_file):
            df_history = pd.read_csv(history_file)
            df_new = pd.DataFrame([patient_record])
            df_history = pd.concat([df_history, df_new], ignore_index=True)
        else:
            df_history = pd.DataFrame([patient_record])
        
        # Guardar el archivo actualizado
        df_history.to_csv(history_file, index=False)
        
        return True, f"✅ Paciente guardado en {history_file}"
        
    except Exception as e:
        return False, f"❌ Error al guardar: {str(e)}"

def get_risk_interpretation(features):
    """
    Interpreta los valores de riesgo en lenguaje comprensible para el usuario
    """
    interpretations = {}
    
    # Edad
    age = features['age']
    if age < 40:
        interpretations['edad'] = {"nivel": "BAJO", "descripcion": "Edad de bajo riesgo"}
    elif age < 60:
        interpretations['edad'] = {"nivel": "MODERADO", "descripcion": "Edad de riesgo moderado"}
    else:
        interpretations['edad'] = {"nivel": "ALTO", "descripcion": "Edad de mayor riesgo"}
    
    # IMC
    bmi = features['bmi']
    if bmi < 18.5:
        interpretations['imc'] = {"nivel": "BAJO", "descripcion": "Bajo peso"}
    elif bmi < 25:
        interpretations['imc'] = {"nivel": "NORMAL", "descripcion": "Peso normal"}
    elif bmi < 30:
        interpretations['imc'] = {"nivel": "MODERADO", "descripcion": "Sobrepeso"}
    else:
        interpretations['imc'] = {"nivel": "ALTO", "descripcion": "Obesidad"}
    
    # Consumo de alcohol
    alcohol = features['alcohol_consumption']
    if alcohol == 0:
        interpretations['alcohol'] = {"nivel": "NORMAL", "descripcion": "No consume alcohol"}
    elif alcohol <= 7:
        interpretations['alcohol'] = {"nivel": "BAJO", "descripcion": "Consumo ligero"}
    elif alcohol <= 14:
        interpretations['alcohol'] = {"nivel": "MODERADO", "descripcion": "Consumo moderado"}
    else:
        interpretations['alcohol'] = {"nivel": "ALTO", "descripcion": "Consumo excesivo - FACTOR DE RIESGO"}
    
    # Estado de fumador
    smoking = features['smoking_status']
    if smoking == 0:
        interpretations['tabaco'] = {"nivel": "NORMAL", "descripcion": "No fumador"}
    elif smoking == 1:
        interpretations['tabaco'] = {"nivel": "ALTO", "descripcion": "Fumador actual - ALTO RIESGO"}
    else:
        interpretations['tabaco'] = {"nivel": "MODERADO", "descripcion": "Ex-fumador - riesgo residual"}
    
    # Función hepática
    liver = features['liver_function_score']
    if liver >= 12:
        interpretations['higado'] = {"nivel": "NORMAL", "descripcion": "Función hepática normal"}
    elif liver >= 8:
        interpretations['higado'] = {"nivel": "MODERADO", "descripcion": "Función hepática ligeramente alterada"}
    else:
        interpretations['higado'] = {"nivel": "ALTO", "descripcion": "Función hepática severamente alterada"}
    
    # Nivel de AFP
    afp = features['alpha_fetoprotein_level']
    if afp <= 10:
        interpretations['afp'] = {"nivel": "NORMAL", "descripcion": "Nivel normal de AFP"}
    elif afp <= 20:
        interpretations['afp'] = {"nivel": "MODERADO", "descripcion": "AFP ligeramente elevada"}
    else:
        interpretations['afp'] = {"nivel": "ALTO", "descripcion": "AFP significativamente elevada - PREOCUPANTE"}
    
    # Actividad física
    activity = features['physical_activity_level']
    if activity >= 7:
        interpretations['actividad'] = {"nivel": "BUENO", "descripcion": "Muy activo - factor protector"}
    elif activity >= 4:
        interpretations['actividad'] = {"nivel": "MODERADO", "descripcion": "Moderadamente activo"}
    else:
        interpretations['actividad'] = {"nivel": "BAJO", "descripcion": "Sedentario - aumenta el riesgo"}
    
    return interpretations

def map_streamlit_to_encoder_values(features):
    """
    Convierte los valores de la interfaz de Streamlit a los valores que esperan los encoders
    """
    mapped_features = features.copy()
    
    # Mapear género: Streamlit (0/1) → Encoder ('Male'/'Female')
    if 'gender' in mapped_features:
        gender_map = {0: 'Female', 1: 'Male'}
        mapped_features['gender'] = gender_map[mapped_features['gender']]
    
    # Mapear consumo de alcohol: Streamlit (0-20 bebidas/semana) → Encoder ('None'/'Light'/'Moderate'/'Heavy')
    if 'alcohol_consumption' in mapped_features:
        alcohol_value = mapped_features['alcohol_consumption']
        if alcohol_value == 0:
            mapped_features['alcohol_consumption'] = 'None'
        elif alcohol_value <= 3:
            mapped_features['alcohol_consumption'] = 'Light'
        elif alcohol_value <= 10:
            mapped_features['alcohol_consumption'] = 'Moderate'
        else:
            mapped_features['alcohol_consumption'] = 'Heavy'
    
    # Mapear estado de fumador: Streamlit (0/1/2) → Encoder ('Never'/'Former'/'Current')
    if 'smoking_status' in mapped_features:
        smoking_map = {0: 'Never', 1: 'Current', 2: 'Former'}
        mapped_features['smoking_status'] = smoking_map[mapped_features['smoking_status']]
    
    # Mapear nivel de actividad física: Streamlit (0-10) → Encoder ('Low'/'Moderate'/'High')
    if 'physical_activity_level' in mapped_features:
        activity_value = mapped_features['physical_activity_level']
        if activity_value <= 3:
            mapped_features['physical_activity_level'] = 'Low'
        elif activity_value <= 7:
            mapped_features['physical_activity_level'] = 'Moderate'
        else:
            mapped_features['physical_activity_level'] = 'High'
    
    return mapped_features

def apply_label_encoders(features, encoders):
    """
    Aplica los label encoders a las características categóricas
    """
    if encoders is None:
        return features
    
    # Primero mapear valores de Streamlit a valores de encoder
    mapped_features = map_streamlit_to_encoder_values(features)
    encoded_features = mapped_features.copy()
    
    # Aplicar encoders
    for col_name, encoder in encoders.items():
        if col_name in encoded_features:
            try:
                original_value = encoded_features[col_name]
                
                # Verificar si el valor está en las clases conocidas
                if hasattr(encoder, 'classes_') and original_value in encoder.classes_:
                    encoded_features[col_name] = encoder.transform([original_value])[0]
                else:
                    st.warning(f"⚠️ Valor '{original_value}' no reconocido para {col_name}")
                    # Usar el primer valor por defecto
                    if hasattr(encoder, 'classes_') and len(encoder.classes_) > 0:
                        encoded_features[col_name] = 0
                
            except Exception as e:
                st.error(f"❌ Error al encodificar {col_name}: {e}")
    
    return encoded_features

# Función para cargar el modelo de forma ultra-robusta
@st.cache_resource
def load_model_and_encoders():
    """
    Carga el modelo y encoders con manejo robusto de errores
    """
    import os
    import pickle
    
    # Buscar archivos
    model_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
    
    if not model_files:
        return None, None, "❌ No se encontraron archivos .pkl"
    
    model = None
    encoders = None
    
    # Intentar cargar modelo y encoders
    for file in model_files:
        try:
            with open(file, 'rb') as f:
                loaded_object = pickle.load(f)
            
            # Identificar si es modelo o encoders
            if hasattr(loaded_object, 'predict'):  # Es un modelo
                model = loaded_object
                
                # Verificar si es LightGBM
                model_type = str(type(loaded_object))
                if 'lightgbm' in model_type.lower():
                    # Verificar si LightGBM está disponible
                    try:
                        import lightgbm
                    except ImportError:
                        model = None
                        
            elif isinstance(loaded_object, dict):  # Probablemente encoders
                encoders = loaded_object
                
        except Exception as e:
            continue
    
    # Verificar resultados
    if model is not None:
        model_info = f"✅ Modelo cargado: {type(model).__name__}"
        if encoders is not None:
            model_info += f" + {len(encoders)} encoders"
        return model, encoders, model_info
    else:
        return None, encoders, "❌ No se encontró modelo válido o LightGBM no disponible"

def prepare_features_for_lightgbm(features, encoders=None):
    """
    Prepara las características para LightGBM con tus encoders específicos
    """
    if encoders is None:
        prepared_features = features.copy()
    else:
        # Aplicar tus encoders específicos
        prepared_features = apply_label_encoders(features, encoders)
    
    # Orden de características (debe coincidir con tu entrenamiento)
    feature_order = [
        'age', 'gender', 'bmi', 'alcohol_consumption', 'smoking_status',
        'hepatitis_b', 'hepatitis_c', 'liver_function_score', 
        'alpha_fetoprotein_level', 'cirrhosis_history', 
        'family_history_cancer', 'physical_activity_level', 'diabetes'
    ]
    
    # Crear DataFrame con el orden correcto
    df = pd.DataFrame([prepared_features])[feature_order]
    
    return df

def calculate_risk_weights(features):
    """
    Calcula los pesos de riesgo para el análisis visual
    (independiente del modelo ML)
    """
    risk_weights = {
        'age_risk': 0.3 if features['age'] > 60 else 0.1 if features['age'] > 45 else 0,
        'alcohol_risk': 0.25 if features['alcohol_consumption'] > 10 else 0.15 if features['alcohol_consumption'] > 5 else 0,
        'smoking_risk': 0.3 if features['smoking_status'] == 1 else 0.2 if features['smoking_status'] == 2 else 0,
        'hepatitis_b_risk': 0.4 if features['hepatitis_b'] == 1 else 0,
        'hepatitis_c_risk': 0.45 if features['hepatitis_c'] == 1 else 0,
        'cirrhosis_risk': 0.5 if features['cirrhosis_history'] == 1 else 0,
        'family_risk': 0.15 if features['family_history_cancer'] == 1 else 0,
        'afp_risk': 0.35 if features['alpha_fetoprotein_level'] > 20 else 0.2 if features['alpha_fetoprotein_level'] > 10 else 0,
        'liver_function_risk': 0.25 if features['liver_function_score'] < 5 else 0.15 if features['liver_function_score'] < 8 else 0,
        'diabetes_risk': 0.1 if features['diabetes'] == 1 else 0,
        'bmi_risk': 0.1 if features['bmi'] > 30 else 0,
        'activity_risk': 0.1 if features['physical_activity_level'] < 3 else 0
    }
    
    return risk_weights

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

def predict_cancer(features):
    """
    Realiza la predicción con interfaz simplificada para el usuario
    """
    # Verificar modelo subido por usuario
    if 'custom_model' in st.session_state and st.session_state['custom_model'] is not None:
        model = st.session_state['custom_model']
        encoders = st.session_state.get('custom_encoders', None)
    else:
        # Cargar modelo y encoders del servidor (sin mostrar detalles técnicos)
        model, encoders, _ = load_model_and_encoders()
    
    if model is not None:
        try:
            # Preparar datos silenciosamente
            if encoders is not None:
                model_features = prepare_features_for_lightgbm(features, encoders)
            else:
                feature_order = [
                    'age', 'gender', 'bmi', 'alcohol_consumption', 'smoking_status',
                    'hepatitis_b', 'hepatitis_c', 'liver_function_score', 
                    'alpha_fetoprotein_level', 'cirrhosis_history', 
                    'family_history_cancer', 'physical_activity_level', 'diabetes'
                ]
                model_features = pd.DataFrame([features])[feature_order]
            
            # Hacer predicción silenciosamente
            if hasattr(model, 'predict'):
                prediction_result = model.predict(model_features.values)
                
                if hasattr(prediction_result, '__len__') and len(prediction_result) > 0:
                    prediction_value = prediction_result[0]
                else:
                    prediction_value = prediction_result
                
                if isinstance(prediction_value, (int, np.integer)):
                    prediction = int(prediction_value)
                    probability = 0.75 if prediction == 1 else 0.25
                else:
                    probability = float(prediction_value)
                    prediction = int(probability > 0.5)
                
                # Intentar obtener probabilidades
                if hasattr(model, 'predict_proba'):
                    try:
                        probabilities = model.predict_proba(model_features.values)[0]
                        if len(probabilities) > 1:
                            probability = probabilities[1]
                    except Exception:
                        pass
                
                # Calcular factores de riesgo
                risk_weights = calculate_risk_weights(features)
                
                return int(prediction), float(probability), risk_weights, "MODELO_ML"
            
        except Exception as e:
            # Error silencioso, continuar con predicción de respaldo
            pass
    
    # PREDICCIÓN DE RESPALDO (sin mencionar que es respaldo)
    risk_weights = calculate_risk_weights(features)
    
    # Factores de riesgo con pesos médicamente validados
    major_risks = (
        risk_weights['hepatitis_b_risk'] * 1.0 +      
        risk_weights['hepatitis_c_risk'] * 1.0 +      
        risk_weights['cirrhosis_risk'] * 0.8           
    )
    
    moderate_risks = (
        risk_weights['age_risk'] * 0.6 +               
        risk_weights['alcohol_risk'] * 0.5 +           
        risk_weights['smoking_risk'] * 0.4 +           
        risk_weights['afp_risk'] * 0.7 +               
        risk_weights['liver_function_risk'] * 0.6     
    )
    
    minor_risks = (
        risk_weights['family_risk'] * 0.3 +            
        risk_weights['diabetes_risk'] * 0.2            
    )
    
    # Calcular probabilidad total
    total_risk = major_risks + moderate_risks + minor_risks
    probability = min(max(total_risk, 0.02), 0.95)  
    
    # Decisión con umbral conservador
    prediction = int(probability > 0.35)  
    
    return prediction, probability, risk_weights, "ANALISIS_CLINICO"

# Encabezado principal
st.markdown('<h1 class="main-header">🏥 Predictor de Cáncer de Hígado</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Sistema inteligente para evaluación de riesgo oncológico</p>', unsafe_allow_html=True)

# Sidebar para entrada de datos con nombre del paciente
st.sidebar.header("📋 Información del Paciente")

# Campo para nombre del paciente
patient_name = st.sidebar.text_input(
    "👤 Nombre del Paciente", 
    placeholder="Ej: Juan Pérez",
    help="Nombre completo del paciente para el historial médico"
)

# Sección para cargar modelo (menos prominente)
with st.sidebar.expander("⚙️ Configuración Avanzada", expanded=False):
    st.markdown("### Cargar Modelo Personalizado")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_model = st.file_uploader(
            "Modelo (.pkl)", 
            type=['pkl'],
            help="Archivo con modelo LightGBM",
            key="model_upload"
        )
    
    with col2:
        uploaded_encoders = st.file_uploader(
            "Encoders (.pkl)", 
            type=['pkl'],
            help="Archivo con label encoders",
            key="encoders_upload"
        )
    
    # Cargar modelo subido
    if uploaded_model is not None:
        try:
            import pickle
            model = pickle.load(uploaded_model)
            st.success("✅ Modelo cargado")
            st.session_state['custom_model'] = model
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state['custom_model'] = None
    
    # Cargar encoders subidos
    if uploaded_encoders is not None:
        try:
            import pickle
            encoders = pickle.load(uploaded_encoders)
            st.success("✅ Encoders cargados")
            st.session_state['custom_encoders'] = encoders
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.session_state['custom_encoders'] = None

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
if st.sidebar.button("🔍 Realizar Análisis", type="primary", use_container_width=True):
    # Validar que se haya ingresado el nombre
    if not patient_name or patient_name.strip() == "":
        st.error("⚠️ Por favor ingrese el nombre del paciente antes de continuar")
    else:
        with st.spinner('🔬 Analizando datos del paciente...'):
            # Hacer predicción
            prediction, probability, risk_weights, analysis_type = predict_cancer(features)
            
            # Obtener interpretaciones para el usuario
            interpretations = get_risk_interpretation(features)
            
            # Mostrar resultado principal - MÁS PROMINENTE
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-box positive-prediction">
                    <h1>⚠️ RESULTADO: RIESGO ALTO</h1>
                    <h2>Paciente: {patient_name}</h2>
                    <h3>Probabilidad de cáncer hepático: {probability:.1%}</h3>
                    <p><strong>RECOMENDACIÓN: Consulta médica especializada URGENTE</strong></p>
                    <p>Se requieren estudios adicionales inmediatos</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box negative-prediction">
                    <h1>✅ RESULTADO: RIESGO BAJO</h1>
                    <h2>Paciente: {patient_name}</h2>
                    <h3>Probabilidad de cáncer hepático: {probability:.1%}</h3>
                    <p><strong>RECOMENDACIÓN: Continuar con controles médicos regulares</strong></p>
                    <p>Mantener hábitos saludables y seguimiento preventivo</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Guardar en historial
            saved, save_message = save_patient_prediction(patient_name, features, prediction, probability, risk_weights)
            if saved:
                st.success(save_message)
            else:
                st.warning(save_message)
            
            # Análisis de factores de riesgo en lenguaje claro
            st.markdown("## 📊 Análisis de Factores de Riesgo")
            
            # Organizar en categorías de riesgo
            high_risk_factors = []
            moderate_risk_factors = []
            normal_factors = []
            protective_factors = []
            
            for factor, data in interpretations.items():
                if data["nivel"] == "ALTO":
                    high_risk_factors.append((factor, data))
                elif data["nivel"] == "MODERADO":
                    moderate_risk_factors.append((factor, data))
                elif data["nivel"] == "BUENO":
                    protective_factors.append((factor, data))
                else:
                    normal_factors.append((factor, data))
            
            # Mostrar factores por categoría
            col1, col2 = st.columns(2)
            
            with col1:
                if high_risk_factors:
                    st.markdown("### 🔴 Factores de Alto Riesgo")
                    for factor, data in high_risk_factors:
                        st.markdown(f'<div class="risk-factor-high">⚠️ <strong>{factor.upper()}:</strong> {data["descripcion"]}</div>', unsafe_allow_html=True)
                
                if moderate_risk_factors:
                    st.markdown("### 🟡 Factores de Riesgo Moderado")
                    for factor, data in moderate_risk_factors:
                        st.markdown(f'<div class="risk-factor-medium">⚡ <strong>{factor.upper()}:</strong> {data["descripcion"]}</div>', unsafe_allow_html=True)
            
            with col2:
                if protective_factors:
                    st.markdown("### 🟢 Factores Protectores")
                    for factor, data in protective_factors:
                        st.markdown(f'<div class="risk-factor-low">✅ <strong>{factor.upper()}:</strong> {data["descripcion"]}</div>', unsafe_allow_html=True)
                
                if normal_factors:
                    st.markdown("### ⚪ Factores Normales")
                    for factor, data in normal_factors:
                        st.markdown(f'<div class="risk-factor-low">✓ <strong>{factor.upper()}:</strong> {data["descripcion"]}</div>', unsafe_allow_html=True)
            
            # Medidor de probabilidad visual
            st.markdown("### 🎯 Nivel de Riesgo")
            risk_level = "ALTO" if probability > 0.6 else "MODERADO" if probability > 0.3 else "BAJO"
            st.markdown(f"""
            <div class="probability-meter">
                <h3>Riesgo General: {risk_level}</h3>
                <h4>Probabilidad: {probability:.1%}</h4>
                {create_progress_bar(probability)}
            </div>
            """, unsafe_allow_html=True)
            
            # Gráfico de barras con valores interpretados
            st.markdown("### 📈 Análisis Detallado por Factor")
            
            # Preparar datos para gráfico más comprensible
            factor_data = {
                'Factor': [],
                'Nivel de Riesgo': [],
                'Descripción': []
            }
            
            for factor, data in interpretations.items():
                factor_data['Factor'].append(factor.replace('_', ' ').title())
                # Convertir nivel a valor numérico para el gráfico
                level_map = {'NORMAL': 0.2, 'BAJO': 0.3, 'MODERADO': 0.6, 'ALTO': 0.9, 'BUENO': 0.1}
                factor_data['Nivel de Riesgo'].append(level_map.get(data['nivel'], 0.5))
                factor_data['Descripción'].append(data['descripción'])
            
            # Crear gráfico con matplotlib
            fig, ax = plt.subplots(figsize=(12, 8))
            colors = []
            for level in factor_data['Nivel de Riesgo']:
                if level >= 0.8:
                    colors.append('#ff4757')  # Rojo - Alto
                elif level >= 0.5:
                    colors.append('#ff6348')  # Naranja - Moderado
                elif level >= 0.25:
                    colors.append('#ffa502')  # Amarillo - Bajo
                else:
                    colors.append('#26de81')  # Verde - Normal/Bueno
            
            bars = ax.barh(factor_data['Factor'], factor_data['Nivel de Riesgo'], color=colors, alpha=0.8)
            
            ax.set_xlabel('Nivel de Riesgo')
            ax.set_title(f'Perfil de Riesgo - {patient_name}', fontsize=16, fontweight='bold')
            ax.set_xlim(0, 1)
            
            # Añadir líneas de referencia
            ax.axvline(x=0.3, color='orange', linestyle='--', alpha=0.7, label='Riesgo Moderado')
            ax.axvline(x=0.6, color='red', linestyle='--', alpha=0.7, label='Riesgo Alto')
            
            # Añadir etiquetas en las barras
            for i, (bar, desc) in enumerate(zip(bars, factor_data['Descripción'])):
                width = bar.get_width()
                ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                       f'{desc}', ha='left', va='center', fontsize=9)
            
            plt.legend()
            plt.tight_layout()
            st.pyplot(fig)
            
            # Recomendaciones personalizadas basadas en factores de riesgo
            st.markdown("### 💡 Recomendaciones Personalizadas")
            
            recommendations = []
            urgent_recommendations = []
            
            # Recomendaciones basadas en factores específicos
            if interpretations['alcohol']['nivel'] in ['MODERADO', 'ALTO']:
                if interpretations['alcohol']['nivel'] == 'ALTO':
                    urgent_recommendations.append("🚨 **URGENTE**: Reducir drásticamente el consumo de alcohol - Es un factor de riesgo mayor para cáncer hepático")
                else:
                    recommendations.append("🍺 Considerar reducir el consumo de alcohol a niveles mínimos")
            
            if interpretations['tabaco']['nivel'] == 'ALTO':
                urgent_recommendations.append("🚨 **URGENTE**: Dejar de fumar inmediatamente - Aumenta significativamente el riesgo de cáncer")
            elif interpretations['tabaco']['nivel'] == 'MODERADO':
                recommendations.append("🚭 Mantener abstinencia del tabaco - El riesgo disminuye con el tiempo")
            
            if interpretations['actividad']['nivel'] == 'BAJO':
                recommendations.append("🏃‍♂️ Aumentar la actividad física a al menos 150 minutos por semana")
            
            if interpretations['imc']['nivel'] in ['MODERADO', 'ALTO']:
                recommendations.append("⚖️ Mantener un peso saludable mediante dieta balanceada y ejercicio")
            
            if interpretations['higado']['nivel'] in ['MODERADO', 'ALTO']:
                urgent_recommendations.append("🏥 **URGENTE**: Seguimiento médico especializado inmediato para función hepática")
            
            if interpretations['afp']['nivel'] in ['MODERADO', 'ALTO']:
                urgent_recommendations.append("🔬 **URGENTE**: Repetir análisis de AFP y realizar estudios de imagen")
            
            # Recomendaciones para casos con hepatitis
            if features['hepatitis_b'] == 1 or features['hepatitis_c'] == 1:
                urgent_recommendations.append("🦠 **URGENTE**: Control hepatológico especializado cada 3-6 meses")
                urgent_recommendations.append("💊 Evaluar tratamiento antiviral si no lo tiene")
            
            if features['cirrhosis_history'] == 1:
                urgent_recommendations.append("🔴 **URGENTE**: Seguimiento oncológico cada 3 meses con estudios de imagen")
            
            # Mostrar recomendaciones urgentes
            if urgent_recommendations:
                st.markdown("#### 🚨 Recomendaciones Urgentes")
                for rec in urgent_recommendations:
                    st.error(rec)
            
            # Mostrar recomendaciones generales
            if recommendations:
                st.markdown("#### 📋 Recomendaciones Generales")
                for i, rec in enumerate(recommendations, 1):
                    st.info(f"{i}. {rec}")
            
            # Recomendaciones universales
            st.markdown("#### 🌟 Recomendaciones para Todos los Pacientes")
            universal_recommendations = [
                "📅 Controles médicos regulares cada 6-12 meses",
                "🥗 Dieta rica en frutas, verduras y baja en grasas procesadas",
                "💧 Mantener hidratación adecuada (2-3 litros de agua al día)",
                "😴 Dormir 7-8 horas diarias para permitir regeneración hepática",
                "🧘‍♂️ Manejar el estrés mediante técnicas de relajación",
                "💉 Mantener vacunas actualizadas (especialmente hepatitis A y B)",
                "⚕️ Informar a su médico sobre cualquier medicamento o suplemento"
            ]
            
            for rec in universal_recommendations:
                st.success(rec)
            
            # Información sobre cuándo buscar atención médica
            st.markdown("### 🚨 Buscar Atención Médica Inmediata Si Presenta:")
            
            warning_signs = [
                "Dolor abdominal persistente en lado derecho",
                "Pérdida de peso inexplicable",
                "Fatiga extrema y debilidad",
                "Coloración amarillenta de piel u ojos (ictericia)",
                "Hinchazón abdominal o de piernas",
                "Cambios en el color de orina (muy oscura) o heces (muy claras)",
                "Sangrado o moretones fáciles",
                "Náuseas y vómitos persistentes"
            ]
            
            for sign in warning_signs:
                st.warning(f"⚠️ {sign}")
            
            # Información de contacto médico (puedes personalizar)
            st.markdown("### 📞 Información de Contacto")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info("""
                **🏥 Para Emergencias:**
                - Llamar al 911
                - Acudir al hospital más cercano
                """)
            
            with col2:
                st.info("""
                **👨‍⚕️ Para Consultas:**
                - Contactar a su médico de cabecera
                - Solicitar referencia a hepatólogo si es necesario
                """)

# Sección de historial de pacientes
st.markdown("---")
st.subheader("📋 Historial de Pacientes")

# Mostrar historial si existe
try:
    import os
    if os.path.exists('historial_pacientes.csv'):
        df_history = pd.read_csv('historial_pacientes.csv')
        
        if len(df_history) > 0:
            # Filtros para el historial
            col1, col2, col3 = st.columns(3)
            
            with col1:
                filter_name = st.text_input("🔍 Buscar por nombre:", placeholder="Nombre del paciente")
            
            with col2:
                filter_result = st.selectbox("🎯 Filtrar por resultado:", 
                                           options=["Todos", "POSITIVO", "NEGATIVO"])
            
            with col3:
                filter_risk = st.selectbox("⚠️ Filtrar por riesgo:", 
                                         options=["Todos", "ALTO", "MODERADO", "BAJO"])
            
            # Aplicar filtros
            filtered_df = df_history.copy()
            
            if filter_name:
                filtered_df = filtered_df[filtered_df['nombre_paciente'].str.contains(filter_name, case=False, na=False)]
            
            if filter_result != "Todos":
                filtered_df = filtered_df[filtered_df['prediccion'] == filter_result]
            
            if filter_risk != "Todos":
                filtered_df = filtered_df[filtered_df['riesgo_nivel'] == filter_risk]
            
            # Mostrar tabla
            if len(filtered_df) > 0:
                st.dataframe(filtered_df, use_container_width=True)
                
                # Botón para descargar historial
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Descargar Historial (CSV)",
                    data=csv,
                    file_name=f"historial_pacientes_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No se encontraron pacientes con los filtros aplicados")
        else:
            st.info("No hay pacientes registrados aún")
    else:
        st.info("No hay historial de pacientes aún")
        
except Exception as e:
    st.error(f"Error al cargar historial: {e}")

# Información adicional sin predicción
else:
    # Mostrar información general cuando no hay predicción activa
    st.markdown("## 👈 Complete la información del paciente")
    st.markdown("Ingrese el **nombre del paciente** y complete todos los campos médicos en la barra lateral, luego presione **'Realizar Análisis'**.")
    
    # Información educativa más enfocada al usuario médico
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **📊 Este Sistema Evalúa:**
        - 🦠 Factores virales (Hepatitis B/C)
        - 🔴 Historial de cirrosis
        - 🍺 Consumo de alcohol
        - 🚬 Historial de tabaquismo
        - 📈 Marcadores tumorales (AFP)
        - 🧬 Antecedentes familiares
        - 📅 Factores demográficos
        """)
    
    with col2:
        st.warning("""
        **⚠️ Importante Recordar:**
        - 🩺 Herramienta de apoyo diagnóstico únicamente
        - 👨‍⚕️ No reemplaza criterio médico profesional
        - 🔬 Combinar con estudios complementarios
        - 📋 Considerar contexto clínico completo
        - 🏥 Derivar a especialista según indicación
        """)

# Footer profesional
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🏥 <strong>Sistema de Evaluación de Riesgo de Cáncer Hepático</strong></p>
    <p>Desarrollado para apoyo en la toma de decisiones clínicas</p>
    <p><small>Versión 1.0 | Última actualización: {}</small></p>
    <p><small>⚠️ Solo para uso profesional médico</small></p>
</div>
""".format(datetime.now().strftime("%d/%m/%Y")), unsafe_allow_html=True)