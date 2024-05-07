import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, url_for
import matplotlib.pyplot as plt
import pickle
from sklearn.tree import DecisionTreeClassifier
import streamlit as st

# Path del modelo preentrenado
MODEL_PATH = 'models/pickle_model1.pkl'

# Se recibe la imagen y el modelo, devuelve la predicción
def model_prediction(x_in, model):
    x = np.asarray(x_in).reshape(1,-1)
    preds = model.predict(x)
    return preds

def main():
    
    model = ''

    # Se carga el modelo
    if model == '':
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
    
    # Título y descripción
    st.title("Sistema de Evaluación del Estrés Académico y Visualización de Factores Relevantes")
    st.write("Ingrese los valores para cada variable y obtenga una predicción sobre su nivel de estrés académico, ademas mediante una grafica visualizara los factores relevantes que influyen en su analisis.")

    # Factores Psicológicos
    st.header("Factores Psicológicos")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Nivel de Ansiedad (0 a 4: ausencia de sintomas, 5 a 9: sintomas leves, 10 a 14: sintomas moderados, 15 a 21: sintomas severos)")
        anxiety_level = st.number_input("Nivel de Ansiedad:", step=1, min_value=0, max_value=21)
        st.write("Autoestima (0 a 14: autoestima baja, 15 a 25: autoestima normal, 26 a 30: autoestima alta)")
        self_esteem = st.number_input("Autoestima:", step=1, min_value=0, max_value=30)
    with col2:    
        st.write("Historia de Salud Mental (0: Ausencia, 1: Presencia)")
        mental_health_history = st.selectbox("Historia de Salud Mental:", options=[0, 1])
        st.write("Depresión (1 a 4: depresión mínima, 5 a 9: depresión leve, 10 a 14: depresión moderada, 15 a 19: depresión moderadamente severa, 20 a 27: depresión severa)")
        depression = st.number_input("Depresión:", step=1, min_value=0, max_value=27)

    # Factores Fisiológicos
    st.header("Factores Fisiológicos")
    col3, col4 = st.columns(2)
    with col3:
        st.write("Dolor de Cabeza (0 a 1: Bajo, 2 a 3: Moderado, 4 a 5: Alto)")
        headache = st.selectbox("Dolor de Cabeza:", options=[0, 1, 2, 3, 4, 5])
        st.write("Presión Sanguínea (1: Ideal, 2: Normal, 3: Hipertensión)")
        blood_pressure = st.selectbox("Presión Sanguínea:", options=[1, 2, 3])
    with col4:    
        st.write("Calidad del Sueño (0 a 1: Buena, 2 a 3: Intermedia, 4 a 5: Mala)")
        sleep_quality = st.selectbox("Calidad del Sueño:", options=[0, 1, 2, 3, 4, 5])
        st.write("Problema Respiratorio (0 a 1: Bajo, 2 a 3: Moderado, 4 a 5: Alto)")
        breathing_problem = st.selectbox("Problema Respiratorio:", options=[0, 1, 2, 3, 4, 5])

    # Factores Ambientales
    st.header("Factores Ambientales")
    col5, col6 = st.columns(2)
    with col5:
        st.write("Nivel de Ruido (0 a 1: Bajo, 2 a 3: Moderado, 4 a 5: Alto)")
        noise_level = st.selectbox("Nivel de Ruido:", options=[0, 1, 2, 3, 4, 5])
        st.write("Condiciones de Vida (0 a 1: Buena, 2 a 3: Media, 4 a 5: Mala)")
        living_conditions = st.selectbox("Condiciones de Vida:", options=[0, 1, 2, 3, 4, 5])
    with col6:
        st.write("Nivel Seguridad (0 a 1: nula, 2 a 3: Moderada, 4 a 5: Alta)")
        safety = st.selectbox("Seguridad:", options=[0, 1, 2, 3, 4, 5])
        st.write("Necesidades Básicas (0 a 1: Cubiertas, 2 a 3: Parcialmente cubiertas, 4 a 5: No cubiertas)")
        basic_needs = st.selectbox("Necesidades Básicas:", options=[0, 1, 2, 3, 4, 5])

    # Factores Académicos
    st.header("Factores Académicos")
    col7, col8 = st.columns(2)
    with col7:
        st.write("Rendimiento Académico (0 a 1: Bueno, 2 a 3: Medio, 4 a 5: Bajo)")
        academic_performance = st.selectbox("Rendimiento Académico:", options=[0, 1, 2, 3, 4, 5])
        st.write("Carga de Estudio (0 a 1: Baja, 2 a 3: Moderada, 4 a 5: Alta)")
        study_load = st.selectbox("Carga de Estudio:", options=[0, 1, 2, 3, 4, 5])
    with col8:
        st.write("Relación Profesor-Estudiante (0 a 1: Buena, 2 a 3: Moderada, 4 a 5: Deficiente)")
        teacher_student_relationship = st.selectbox("Relación Profesor-Estudiante:", options=[0, 1, 2, 3, 4, 5])
        st.write("Preocupaciones Futuras de Carrera (0 a 1: Baja, 2 a 3: Moderada, 4 a 5: Alta)")
        future_career_concerns = st.selectbox("Preocupaciones Futuras de Carrera:", options=[0, 1, 2, 3, 4, 5])

    # Factores Sociales
    st.header("Factores Sociales")
    col9, col10 = st.columns(2)
    with col9:
        st.write("Apoyo Social (0: Ausencia, 1: Bajo, 2: Moderado, 3: Alto)")
        social_support = st.selectbox("Apoyo Social:", options=[0, 1, 2, 3])
        st.write("Presión de Pares (0: Nada, 1: Poca, 2: Moderada, 3: Medio, 4: Alto, 5: Extremo)")
        peer_pressure = st.selectbox("Presión de Pares:", options=[0, 1, 2, 3, 4, 5])
    with col10:
        st.write("Actividades Extracurriculares (1: Mínimo, 2: Bajo, 3: Medio, 4: Alto, 5: Máximo)")
        extracurricular_activities = st.selectbox("Actividades Extracurriculares:", options=[1, 2, 3, 4, 5])
        st.write("Acoso (1: Nada, 2: Poco, 3: Moderado, 4: Mucho, 5: Extremo)")
        bullying = st.selectbox("Acoso:", options=[1, 2, 3, 4, 5])
    
    # El botón de predicción inicia el procesamiento
    if st.button("Predicción", key="predict_button", help="Haz clic aquí para realizar la predicción", on_click=None):
        x_in = [anxiety_level, self_esteem, mental_health_history, depression,
                headache, blood_pressure, sleep_quality, breathing_problem,
                noise_level, living_conditions, safety, basic_needs,
                academic_performance, study_load, teacher_student_relationship,
                future_career_concerns, social_support, peer_pressure,
                extracurricular_activities, bullying]

    # Realizar predicción y obtener la importancia de los factores            
        prediction= model_prediction(x_in, model)

        #st.success(f"Su nivel de estrés es: {prediction[0]}")
        if prediction == 0:
            st.success(f"Su nivel de estrés es: Bajo")
        elif prediction == 1:
            st.success(f"Su nivel de estrés es: Medio")
        else: 
            st.success(f"Su nivel de estrés es: Alto")
        
        #Parte grafica
        Nivel_Ansiedad = anxiety_level * 100/21
        Autoestima = self_esteem * 100/30
        Historia_Salud_Mental = mental_health_history * 100/1
        Depresión = depression * 100/27
        Dolor_Cabeza = headache * 100/2
        Presion_Sanguinea = blood_pressure * 100/3
        Calidad_Sueño = sleep_quality * 100/5
        Problema_Respiratorio = breathing_problem * 100/5
        Nivel_Ruido = noise_level * 100/5
        Condiciones_Vida = living_conditions * 100/5
        Nivel_Seguridad = safety * 100/5  # Elimina la coma al final de esta línea
        Necesidades_Basicas = basic_needs * 100/5  # Elimina la coma al final de esta línea
        Rendimiento_Academico = academic_performance * 100/5
        Carga_de_Estudio = study_load * 100/5
        Relacion_Profesor_Estudiante = teacher_student_relationship * 100/5
        Preocupaciones_Futuras_Carrera = future_career_concerns * 100/5
        Apoyo_Social = social_support * 100/3
        Presión_Pares = peer_pressure * 100/5
        Actividades_Extracurriculares = extracurricular_activities * 100/5
        Acoso = bullying * 100/5

        #Define la importancia de cada variable (ejemplo)
        importancia_variables = {
            'Factores Sociales': Apoyo_Social + Presión_Pares + Actividades_Extracurriculares + Acoso /4,
            'Factores Académicos': Rendimiento_Academico + Carga_de_Estudio + Relacion_Profesor_Estudiante + Preocupaciones_Futuras_Carrera / 4,
            'Factores Ambientales': Nivel_Ruido + Condiciones_Vida + Nivel_Seguridad + Necesidades_Basicas / 4,
            'Factores Fisiológicos': Dolor_Cabeza + Presion_Sanguinea + Calidad_Sueño + Problema_Respiratorio / 4,
            'Factores Psicológicos': Nivel_Ansiedad + Autoestima + Historia_Salud_Mental + Depresión / 4
        }

        # Título y descripción
        st.title("Histograma Comparativo  de Factores con mayor Influencia al estrés")
        st.write("Visualización comparativa de la importancia de cada factor en el sistema de evaluación del estrés académico.")

        # Obtén los nombres de las variables y sus importancias
        nombres_variables = list(importancia_variables.keys())
        importancias = list(importancia_variables.values())

        # Crear el histograma comparativo
        fig, ax = plt.subplots()
        ax.barh(nombres_variables, importancias, color='skyblue')
        ax.set_xlabel('Importancia')
        ax.set_title('Importancia de los factores')
        st.pyplot(fig)

               
if __name__ == '__main__':
    main()

