"""
Aplicación Streamlit para Detección de Fraude con SMOTE
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from imblearn.over_sampling import SMOTE

# Configuración de la página
st.set_page_config(
    page_title="Detección de Fraude con SMOTE",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Título principal
st.title("💳 Sistema de Detección de Fraude con SMOTE")
st.markdown("### Aplicación Interactiva para Balanceo de Datos")
st.markdown("---")

# Sidebar para navegación
st.sidebar.title("🎯 Navegación")
pagina = st.sidebar.radio(
    "Selecciona una sección:",
    ["📊 Exploración de Datos", 
     "⚙️ Entrenamiento del Modelo", 
     "🔮 Hacer Predicciones",
     "📈 Análisis de Resultados"]
)

# Inicializar variables en session_state
if 'modelo' not in st.session_state:
    st.session_state.modelo = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'columnas' not in st.session_state:
    st.session_state.columnas = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'metricas_baseline' not in st.session_state:
    st.session_state.metricas_baseline = None
if 'metricas_smote' not in st.session_state:
    st.session_state.metricas_smote = None

# ========================================
# PÁGINA 1: EXPLORACIÓN DE DATOS
# ========================================
if pagina == "📊 Exploración de Datos":
    st.header("📊 Exploración y Análisis del Dataset")
    
    # Cargar datos
    uploaded_file = st.file_uploader(
        "Sube tu archivo CSV de transacciones:",
        type=['csv'],
        help="Debe contener las columnas V1-V28, Time, Amount y Class"
    )
    
    if uploaded_file is not None:
        try:
            # Intentar cargar normalmente primero
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
            except:
                # Si falla, reintentar con encoding diferente
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='latin-1')
            
            # Limpiar nombres de columnas (quitar espacios, comillas, caracteres especiales)
            df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", '').str.replace('\\', '')
            
            # Si la primera columna se llama 'ï»¿Time' o similar (BOM), limpiarla
            if df.columns[0].startswith('ï»¿') or df.columns[0].startswith('\ufeff'):
                df.columns = [col.replace('ï»¿', '').replace('\ufeff', '') for col in df.columns]
            
            # Verificar que exista la columna Class
            if 'Class' not in df.columns:
                st.error(f"""
                ❌ **Error: Columna 'Class' no encontrada**
                
                **Columnas disponibles en el archivo:**
                {', '.join(df.columns.tolist())}
                
                **Soluciones:**
                1. Verifica que tu CSV tenga una columna llamada 'Class'
                2. Si el CSV tiene formato incorrecto, usa el script `limpiar_csv.py`
                3. Asegúrate de que el archivo sea el correcto
                """)
                
                # Mostrar las primeras filas para debug
                with st.expander("🔍 Ver primeras filas del archivo (debug)"):
                    st.dataframe(df.head())
                
                st.stop()
            
            # Verificar que Class tenga valores 0 y 1
            valores_class = df['Class'].unique()
            if not all(v in [0, 1, '0', '1'] for v in valores_class):
                st.warning(f"""
                ⚠️ **Advertencia: Valores inesperados en 'Class'**
                
                Valores encontrados: {valores_class}
                
                La columna 'Class' debería contener solo 0 (No Fraude) y 1 (Fraude).
                """)
            
            # Convertir Class a numérico si es necesario
            df['Class'] = pd.to_numeric(df['Class'], errors='coerce')
            
            # Eliminar filas con valores nulos en Class
            if df['Class'].isna().any():
                df = df.dropna(subset=['Class'])
                st.info(f"ℹ️ Se eliminaron {df['Class'].isna().sum()} filas con valores nulos en 'Class'")
            
            st.session_state.df = df
            st.session_state.columnas = df.drop('Class', axis=1).columns.tolist()
            
            st.success(f"✅ Archivo cargado exitosamente: {df.shape[0]:,} filas, {df.shape[1]} columnas")
            
        except Exception as e:
            st.error(f"""
            ❌ **Error al cargar el archivo:**
            
            ```
            {str(e)}
            ```
            
            **Soluciones:**
            
            1. **CSV con formato incorrecto:**
               - Ejecuta el script `limpiar_csv.py` primero
               - Comando: `python limpiar_csv.py`
               - Luego carga el archivo limpio
            
            2. **Verifica el formato:**
               - El archivo debe ser un CSV válido
               - Debe tener las columnas: Time, V1-V28, Amount, Class
               - La primera fila debe ser el encabezado
            
            3. **Encoding:**
               - Guarda el CSV con encoding UTF-8
               - Evita caracteres especiales en los nombres de columnas
            """)
            st.stop()
        
        # Tabs para organizar la información
        tab1, tab2, tab3 = st.tabs(["📋 Vista General", "📊 Análisis de Desbalanceo", "📈 Estadísticas"])
        
        with tab1:
            st.subheader("Vista General del Dataset")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total de Transacciones", f"{df.shape[0]:,}")
            with col2:
                st.metric("Total de Features", df.shape[1] - 1)
            with col3:
                st.metric("Valores Nulos", df.isnull().sum().sum())
            
            st.write("**Primeras 10 filas del dataset:**")
            st.dataframe(df.head(10), use_container_width=True)
            
        with tab2:
            st.subheader("Análisis del Desbalanceo de Clases")
            
            # Calcular distribución
            class_counts = df['Class'].value_counts()
            class_percentages = df['Class'].value_counts(normalize=True) * 100
            
            # Mostrar métricas de desbalanceo
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("No Fraude (Clase 0)", f"{class_counts[0]:,}", 
                         f"{class_percentages[0]:.2f}%")
            with col2:
                st.metric("Fraude (Clase 1)", f"{class_counts[1]:,}", 
                         f"{class_percentages[1]:.2f}%")
            with col3:
                ratio = class_counts[0] // class_counts[1]
                st.metric("Ratio de Desbalanceo", f"1:{ratio}")
            
            # Visualizaciones
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.bar(['No Fraude', 'Fraude'], class_counts.values, 
                       color=['#2ecc71', '#e74c3c'])
                ax.set_ylabel('Cantidad de Transacciones')
                ax.set_title('Distribución de Clases')
                ax.set_yscale('log')
                for i, v in enumerate(class_counts.values):
                    ax.text(i, v, f'{v:,}', ha='center', va='bottom')
                st.pyplot(fig)
                
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = ['#2ecc71', '#e74c3c']
                ax.pie(class_counts.values, labels=['No Fraude', 'Fraude'],
                      autopct='%1.2f%%', colors=colors, startangle=90)
                ax.set_title('Porcentaje de Clases')
                st.pyplot(fig)
            
            # Advertencia sobre desbalanceo
            if class_percentages[1] < 5:
                st.warning(f"""
                ⚠️ **Dataset Severamente Desbalanceado**
                
                La clase minoritaria (Fraude) representa solo el {class_percentages[1]:.2f}% de los datos.
                Esto puede causar que el modelo:
                - Tenga un bias hacia la clase mayoritaria
                - Presente baja sensibilidad para detectar fraudes
                - Muestre accuracy engañosamente alto
                
                **Solución**: Aplicar SMOTE en la siguiente sección.
                """)
        
        with tab3:
            st.subheader("Estadísticas Descriptivas")
            
            st.write("**Estadísticas de las Variables Numéricas:**")
            st.dataframe(df.describe(), use_container_width=True)
            
            # Distribución de Amount
            st.write("**Distribución del Monto de Transacciones:**")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.hist(df['Amount'], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Monto (USD)')
            ax.set_ylabel('Frecuencia')
            ax.set_title('Distribución de Montos de Transacciones')
            ax.set_yscale('log')
            st.pyplot(fig)
    
    else:
        st.info("👆 Por favor, sube un archivo CSV para comenzar el análisis.")
        st.markdown("""
        **El archivo debe contener las siguientes columnas:**
        - `Time`: Tiempo transcurrido desde la primera transacción
        - `V1` a `V28`: Variables transformadas por PCA
        - `Amount`: Monto de la transacción
        - `Class`: Variable objetivo (0 = No fraude, 1 = Fraude)
        """)

# ========================================
# PÁGINA 2: ENTRENAMIENTO DEL MODELO
# ========================================
elif pagina == "⚙️ Entrenamiento del Modelo":
    st.header("⚙️ Entrenamiento del Modelo con SMOTE")
    
    if st.session_state.df is None:
        st.warning("⚠️ Por favor, carga primero el dataset en la sección de Exploración de Datos.")
    else:
        df = st.session_state.df
        
        st.markdown("""
        ### ¿Qué es SMOTE?
        
        **SMOTE (Synthetic Minority Over-sampling Technique)** es una técnica que genera 
        muestras sintéticas de la clase minoritaria mediante interpolación entre instancias cercanas.
        
        **Proceso:**
        1. Selecciona una muestra de la clase minoritaria
        2. Encuentra sus k vecinos más cercanos
        3. Crea nuevas muestras interpolando entre ellos
        """)
        
        # Configuración de parámetros
        st.subheader("🎛️ Configuración de Parámetros")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sampling_strategy = st.slider(
                "Sampling Strategy",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.1,
                help="Ratio deseado: 0.5 significa que la clase minoritaria será 50% de la mayoritaria"
            )
        
        with col2:
            k_neighbors = st.number_input(
                "K Neighbors",
                min_value=1,
                max_value=10,
                value=5,
                help="Número de vecinos cercanos a considerar"
            )
        
        with col3:
            test_size = st.slider(
                "Tamaño del Test Set",
                min_value=0.1,
                max_value=0.4,
                value=0.3,
                step=0.05,
                help="Proporción de datos para el conjunto de prueba"
            )
        
        # Botón para entrenar
        if st.button("🚀 Entrenar Modelo", type="primary", use_container_width=True):
            with st.spinner("Entrenando modelo... Esto puede tomar unos segundos."):
                
                # Preparar datos
                X = df.drop('Class', axis=1)
                y = df['Class']
                
                # Split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                
                # Guardar en session_state
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                
                # Mostrar distribución original
                st.write("### 📊 Distribución de Datos")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Conjunto de Entrenamiento:**")
                    train_dist = Counter(y_train)
                    st.write(f"- Clase 0 (No Fraude): {train_dist[0]:,}")
                    st.write(f"- Clase 1 (Fraude): {train_dist[1]:,}")
                
                with col2:
                    st.write("**Conjunto de Prueba:**")
                    test_dist = Counter(y_test)
                    st.write(f"- Clase 0 (No Fraude): {test_dist[0]:,}")
                    st.write(f"- Clase 1 (Fraude): {test_dist[1]:,}")
                
                # Paso 1: Modelo Baseline
                st.write("### 🔵 Paso 1: Modelo Baseline (Sin Balanceo)")
                progress_bar = st.progress(0)
                
                modelo_baseline = LogisticRegression(random_state=42, max_iter=1000)
                modelo_baseline.fit(X_train, y_train)
                progress_bar.progress(25)
                
                y_pred_baseline = modelo_baseline.predict(X_test)
                y_pred_proba_baseline = modelo_baseline.predict_proba(X_test)[:, 1]
                
                # Métricas baseline
                metricas_baseline = {
                    'accuracy': accuracy_score(y_test, y_pred_baseline),
                    'precision': precision_score(y_test, y_pred_baseline),
                    'recall': recall_score(y_test, y_pred_baseline),
                    'f1': f1_score(y_test, y_pred_baseline),
                    'auc': roc_auc_score(y_test, y_pred_proba_baseline)
                }
                st.session_state.metricas_baseline = metricas_baseline
                progress_bar.progress(50)
                
                # Paso 2: Aplicar SMOTE
                st.write("### 🟢 Paso 2: Aplicar SMOTE y Entrenar")
                
                smote = SMOTE(sampling_strategy=sampling_strategy, 
                             random_state=42, 
                             k_neighbors=k_neighbors)
                X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
                progress_bar.progress(75)
                
                st.write("**Distribución después de SMOTE:**")
                smote_dist = Counter(y_train_smote)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Clase 0 (No Fraude)", f"{smote_dist[0]:,}")
                with col2:
                    st.metric("Clase 1 (Fraude)", f"{smote_dist[1]:,}", 
                             f"+{smote_dist[1] - train_dist[1]:,}")
                
                # Entrenar con SMOTE
                modelo_smote = LogisticRegression(random_state=42, max_iter=1000)
                modelo_smote.fit(X_train_smote, y_train_smote)
                
                y_pred_smote = modelo_smote.predict(X_test)
                y_pred_proba_smote = modelo_smote.predict_proba(X_test)[:, 1]
                
                # Métricas SMOTE
                metricas_smote = {
                    'accuracy': accuracy_score(y_test, y_pred_smote),
                    'precision': precision_score(y_test, y_pred_smote),
                    'recall': recall_score(y_test, y_pred_smote),
                    'f1': f1_score(y_test, y_pred_smote),
                    'auc': roc_auc_score(y_test, y_pred_proba_smote)
                }
                st.session_state.metricas_smote = metricas_smote
                st.session_state.modelo = modelo_smote
                progress_bar.progress(100)
                
                st.success("✅ ¡Modelo entrenado exitosamente!")
                
                # Comparación de métricas
                st.write("### 📊 Comparación de Resultados")
                
                comparacion = pd.DataFrame({
                    'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                    'Baseline': [
                        f"{metricas_baseline['accuracy']:.3f}",
                        f"{metricas_baseline['precision']:.3f}",
                        f"{metricas_baseline['recall']:.3f}",
                        f"{metricas_baseline['f1']:.3f}",
                        f"{metricas_baseline['auc']:.3f}"
                    ],
                    'SMOTE': [
                        f"{metricas_smote['accuracy']:.3f}",
                        f"{metricas_smote['precision']:.3f}",
                        f"{metricas_smote['recall']:.3f}",
                        f"{metricas_smote['f1']:.3f}",
                        f"{metricas_smote['auc']:.3f}"
                    ],
                    'Mejora': [
                        f"{(metricas_smote['accuracy'] - metricas_baseline['accuracy']):.3f}",
                        f"{(metricas_smote['precision'] - metricas_baseline['precision']):.3f}",
                        f"{(metricas_smote['recall'] - metricas_baseline['recall']):.3f}",
                        f"{(metricas_smote['f1'] - metricas_baseline['f1']):.3f}",
                        f"{(metricas_smote['auc'] - metricas_baseline['auc']):.3f}"
                    ]
                })
                
                st.dataframe(comparacion, use_container_width=True)
                
                # Visualización de matrices de confusión
                st.write("### 🔍 Matrices de Confusión")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Baseline**")
                    cm_baseline = confusion_matrix(y_test, y_pred_baseline)
                    fig, ax = plt.subplots(figsize=(6, 6))
                    sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues', ax=ax,
                               xticklabels=['No Fraude', 'Fraude'],
                               yticklabels=['No Fraude', 'Fraude'])
                    ax.set_ylabel('Real')
                    ax.set_xlabel('Predicho')
                    ax.set_title('Matriz de Confusión - Baseline')
                    st.pyplot(fig)
                
                with col2:
                    st.write("**SMOTE**")
                    cm_smote = confusion_matrix(y_test, y_pred_smote)
                    fig, ax = plt.subplots(figsize=(6, 6))
                    sns.heatmap(cm_smote, annot=True, fmt='d', cmap='Greens', ax=ax,
                               xticklabels=['No Fraude', 'Fraude'],
                               yticklabels=['No Fraude', 'Fraude'])
                    ax.set_ylabel('Real')
                    ax.set_xlabel('Predicho')
                    ax.set_title('Matriz de Confusión - SMOTE')
                    st.pyplot(fig)

# ========================================
# PÁGINA 3: HACER PREDICCIONES
# ========================================
elif pagina == "🔮 Hacer Predicciones":
    st.header("🔮 Hacer Predicciones de Fraude")
    
    if st.session_state.modelo is None:
        st.warning("⚠️ Por favor, entrena primero el modelo en la sección de Entrenamiento.")
    else:
        st.markdown("""
        ### Ingresa los datos de una transacción para predecir si es fraudulenta
        
        Puedes ingresar los valores manualmente o usar valores aleatorios del dataset.
        """)
        
        # Opción para usar datos del test set
        col1, col2 = st.columns([3, 1])
        
        with col1:
            usar_random = st.checkbox("Usar transacción aleatoria del conjunto de prueba")
        
        with col2:
            if usar_random and st.button("🎲 Generar Aleatorio"):
                idx_random = np.random.randint(0, len(st.session_state.X_test))
                st.session_state.idx_random = idx_random
        
        # Crear formulario para ingresar datos
        st.subheader("📝 Datos de la Transacción")
        
        if usar_random and 'idx_random' in st.session_state:
            idx = st.session_state.idx_random
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
            
            muestra = X_test.iloc[idx]
            real_class = y_test.iloc[idx]
            
            st.info(f"**Transacción #{idx}** - Clase Real: {'🔴 FRAUDE' if real_class == 1 else '🟢 NO FRAUDE'}")
        
        # Formulario con tabs para mejor organización
        tab1, tab2, tab3 = st.tabs(["⏰ Time & Amount", "📊 Variables V1-V14", "📊 Variables V15-V28"])
        
        input_data = {}
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                if usar_random and 'idx_random' in st.session_state:
                    input_data['Time'] = st.number_input("Time", value=float(muestra['Time']), format="%.2f")
                else:
                    input_data['Time'] = st.number_input("Time", value=0.0, format="%.2f")
            
            with col2:
                if usar_random and 'idx_random' in st.session_state:
                    input_data['Amount'] = st.number_input("Amount", value=float(muestra['Amount']), format="%.2f")
                else:
                    input_data['Amount'] = st.number_input("Amount", value=0.0, format="%.2f")
        
        with tab2:
            cols = st.columns(3)
            for i in range(1, 15):
                col_idx = (i-1) % 3
                with cols[col_idx]:
                    col_name = f'V{i}'
                    if usar_random and 'idx_random' in st.session_state:
                        input_data[col_name] = st.number_input(col_name, value=float(muestra[col_name]), format="%.6f")
                    else:
                        input_data[col_name] = st.number_input(col_name, value=0.0, format="%.6f")
        
        with tab3:
            cols = st.columns(3)
            for i in range(15, 29):
                col_idx = (i-15) % 3
                with cols[col_idx]:
                    col_name = f'V{i}'
                    if usar_random and 'idx_random' in st.session_state:
                        input_data[col_name] = st.number_input(col_name, value=float(muestra[col_name]), format="%.6f")
                    else:
                        input_data[col_name] = st.number_input(col_name, value=0.0, format="%.6f")
        
        # Botón para predecir
        st.markdown("---")
        if st.button("🔮 Predecir Fraude", type="primary", use_container_width=True):
            # Preparar datos para predicción
            input_df = pd.DataFrame([input_data])
            
            # Reordenar columnas para que coincidan con el modelo
            input_df = input_df[st.session_state.columnas]
            
            # Hacer predicción
            prediccion = st.session_state.modelo.predict(input_df)[0]
            probabilidad = st.session_state.modelo.predict_proba(input_df)[0]
            
            # Mostrar resultados
            st.markdown("---")
            st.subheader("📊 Resultado de la Predicción")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediccion == 1:
                    st.error("### 🔴 FRAUDE DETECTADO")
                else:
                    st.success("### 🟢 TRANSACCIÓN LEGÍTIMA")
            
            with col2:
                st.metric("Probabilidad de Fraude", f"{probabilidad[1]:.2%}")
            
            with col3:
                st.metric("Probabilidad de No Fraude", f"{probabilidad[0]:.2%}")
            
            # Gráfico de probabilidades
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(['No Fraude', 'Fraude'], probabilidad, color=['#2ecc71', '#e74c3c'])
            ax.set_xlabel('Probabilidad')
            ax.set_xlim([0, 1])
            ax.set_title('Distribución de Probabilidades')
            for i, v in enumerate(probabilidad):
                ax.text(v, i, f' {v:.2%}', va='center')
            st.pyplot(fig)
            
            # Si es una muestra del test set, mostrar si fue correcto
            if usar_random and 'idx_random' in st.session_state:
                st.markdown("---")
                st.subheader("🎯 Validación")
                
                if prediccion == real_class:
                    st.success(f"✅ **Predicción Correcta**: El modelo predijo correctamente que la transacción {'ES' if real_class == 1 else 'NO ES'} fraude.")
                else:
                    st.error(f"❌ **Predicción Incorrecta**: El modelo predijo {'FRAUDE' if prediccion == 1 else 'NO FRAUDE'}, pero la clase real es {'FRAUDE' if real_class == 1 else 'NO FRAUDE'}.")

# ========================================
# PÁGINA 4: ANÁLISIS DE RESULTADOS
# ========================================
elif pagina == "📈 Análisis de Resultados":
    st.header("📈 Análisis Detallado de Resultados")
    
    if st.session_state.metricas_baseline is None or st.session_state.metricas_smote is None:
        st.warning("⚠️ Por favor, entrena primero el modelo en la sección de Entrenamiento.")
    else:
        # Métricas comparativas
        st.subheader("📊 Comparación de Métricas")
        
        metricas_baseline = st.session_state.metricas_baseline
        metricas_smote = st.session_state.metricas_smote
        
        # Tarjetas de métricas
        col1, col2, col3, col4, col5 = st.columns(5)
        
        metricas_nombres = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        metricas_keys = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        cols = [col1, col2, col3, col4, col5]
        
        for col, nombre, key in zip(cols, metricas_nombres, metricas_keys):
            with col:
                val_baseline = metricas_baseline[key]
                val_smote = metricas_smote[key]
                delta = val_smote - val_baseline
                
                st.metric(
                    nombre,
                    f"{val_smote:.3f}",
                    f"{delta:+.3f}",
                    delta_color="normal"
                )
        
        # Gráfico comparativo
        st.subheader("📊 Visualización Comparativa")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(metricas_nombres))
        width = 0.35
        
        baseline_vals = [metricas_baseline[k] for k in metricas_keys]
        smote_vals = [metricas_smote[k] for k in metricas_keys]
        
        ax.bar(x - width/2, baseline_vals, width, label='Baseline', color='#e74c3c', alpha=0.8)
        ax.bar(x + width/2, smote_vals, width, label='SMOTE', color='#2ecc71', alpha=0.8)
        
        ax.set_ylabel('Valor')
        ax.set_title('Comparación de Métricas: Baseline vs SMOTE')
        ax.set_xticks(x)
        ax.set_xticklabels(metricas_nombres)
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        # Añadir valores sobre las barras
        for i, (b_val, s_val) in enumerate(zip(baseline_vals, smote_vals)):
            ax.text(i - width/2, b_val, f'{b_val:.3f}', ha='center', va='bottom', fontsize=9)
            ax.text(i + width/2, s_val, f'{s_val:.3f}', ha='center', va='bottom', fontsize=9)
        
        st.pyplot(fig)
        
        # Interpretación de resultados
        st.markdown("---")
        st.subheader("💡 Interpretación de Resultados")
        
        st.markdown(f"""
        ### Análisis de las Métricas:
        
        **Recall (Sensibilidad):**
        - Baseline: {metricas_baseline['recall']:.1%}
        - SMOTE: {metricas_smote['recall']:.1%}
        - {'✅ Mejora' if metricas_smote['recall'] > metricas_baseline['recall'] else '⚠️ Disminución'}: 
          {(metricas_smote['recall'] - metricas_baseline['recall']):.1%}
        
        > El recall indica qué porcentaje de fraudes reales detectamos. 
        {'Un recall mayor con SMOTE significa que estamos detectando más fraudes.' if metricas_smote['recall'] > metricas_baseline['recall'] else ''}
        
        **Precision:**
        - Baseline: {metricas_baseline['precision']:.1%}
        - SMOTE: {metricas_smote['precision']:.1%}
        - {'✅ Mejora' if metricas_smote['precision'] > metricas_baseline['precision'] else '⚠️ Disminución'}: 
          {(metricas_smote['precision'] - metricas_baseline['precision']):.1%}
        
        > La precision indica de todas las transacciones que marcamos como fraude, cuántas realmente lo son.
        
        **F1-Score:**
        - Baseline: {metricas_baseline['f1']:.1%}
        - SMOTE: {metricas_smote['f1']:.1%}
        - {'✅ Mejora' if metricas_smote['f1'] > metricas_baseline['f1'] else '⚠️ Disminución'}: 
          {(metricas_smote['f1'] - metricas_baseline['f1']):.1%}
        
        > El F1-Score es el balance entre precision y recall, y es una métrica clave para datos desbalanceados.
        """)
        
        # Recomendaciones
        st.markdown("---")
        st.subheader("🎯 Recomendaciones")
        
        if metricas_smote['recall'] > metricas_baseline['recall']:
            st.success("""
            ✅ **SMOTE ha mejorado significativamente la detección de fraudes.**
            
            El modelo ahora detecta más casos de fraude, lo cual es crítico para este problema.
            Aunque pueda haber un ligero aumento en falsos positivos, el beneficio de detectar
            más fraudes reales supera este costo.
            """)
        else:
            st.warning("""
            ⚠️ **Los resultados con SMOTE no muestran una mejora clara.**
            
            Considera:
            - Ajustar el `sampling_strategy`
            - Probar otros valores de `k_neighbors`
            - Usar técnicas de ensemble
            - Ajustar el umbral de decisión
            """)
        
        # Exportar modelo
        st.markdown("---")
        st.subheader("💾 Exportar Modelo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Descargar Modelo Entrenado", use_container_width=True):
                # Serializar el modelo
                modelo_bytes = pickle.dumps(st.session_state.modelo)
                st.download_button(
                    label="Descargar modelo.pkl",
                    data=modelo_bytes,
                    file_name="modelo_fraude_smote.pkl",
                    mime="application/octet-stream"
                )
        
        with col2:
            if st.button("📥 Descargar Reporte de Métricas", use_container_width=True):
                # Crear reporte
                reporte = pd.DataFrame({
                    'Métrica': metricas_nombres,
                    'Baseline': [metricas_baseline[k] for k in metricas_keys],
                    'SMOTE': [metricas_smote[k] for k in metricas_keys],
                    'Mejora': [(metricas_smote[k] - metricas_baseline[k]) for k in metricas_keys]
                })
                
                csv = reporte.to_csv(index=False)
                st.download_button(
                    label="Descargar reporte.csv",
                    data=csv,
                    file_name="reporte_metricas.csv",
                    mime="text/csv"
                )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Sistema de Detección de Fraude con SMOTE</strong></p>
</div>
""", unsafe_allow_html=True)
