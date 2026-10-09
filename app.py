import os
import traceback
import streamlit as st
import joblib  # Importa o joblib diretamente

# Importa apenas a função 'recomendar' (ou o que precisar) do seu arquivo .py
from recomendador_revenimento import recomendar

MODELO_PATH = "recomendador_rv.joblib"

@st.cache_resource
def carregar_modelo_cache():
    if not os.path.exists(MODELO_PATH):
        st.error(f"❌ Ficheiro não encontrado: `{MODELO_PATH}`")
        return None
    
    try:
        # Carrega diretamente via joblib
        return joblib.load(MODELO_PATH)
    except Exception as e:
        st.error(f"❌ Erro ao carregar `{MODELO_PATH}`:")
        st.exception(e)
        return None

modelo = carregar_modelo_cache()
