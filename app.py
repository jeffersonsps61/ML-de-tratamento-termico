import os
import traceback
import streamlit as st

# Nome do ficheiro do modelo
MODELO_PATH = "recomendador_rv.joblib"

@st.cache_resource
def carregar_modelo_cache():
    # Verifica a diretoria atual para garantir o caminho relativo correto
    caminho_absoluto = os.path.abspath(MODELO_PATH)
    
    if not os.path.exists(MODELO_PATH):
        st.error(f"❌ Ficheiro não encontrado: `{MODELO_PATH}`")
        st.write(f"**Caminho procurado:** `{caminho_absoluto}`")
        st.write("**Ficheiros na diretoria atual:**", os.listdir("."))
        return None
    
    try:
        # Tenta carregar o modelo utilizando a função importada
        return carregar(MODELO_PATH)
    except Exception as e:
        st.error(f"❌ Erro ao executar `carregar('{MODELO_PATH}')`:")
        st.exception(e)  # Mostra o erro detalhado no ecrã
        return None

modelo = carregar_modelo_cache()






