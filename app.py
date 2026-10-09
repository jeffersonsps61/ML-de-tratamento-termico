import os
import streamlit as st
import pandas as pd
import numpy as np

# Importa as funções do teu módulo existente
from recomendador_revenimento_5 import carregar, recomendar

st.set_page_config(
    page_title="Recomendador de Revenimento",
    page_icon="🔥",
    layout="wide"
)

st.title("🔥 Recomendador de Temperatura de Revenimento (v2.0)")
st.markdown("---")

# Carregamento do Modelo com Cache para alta performance
MODELO_PATH = "recomendador_rv.joblib"

@st.cache_resource
def carregar_modelo_cache():
    if not os.path.exists(MODELO_PATH):
        st.error(f"Ficheiro de modelo '{MODELO_PATH}' não encontrado no repositório.")
        return None
    return carregar(MODELO_PATH)

modelo = carregar_modelo_cache()

if modelo is not None:
    # Sidebar com Parâmetros de Entrada
    st.sidebar.header("📋 Parâmetros da Peça / Ordem")
    
    materiais_disponiveis = modelo.get("materiais", ["4140", "4340", "8630", "18CRNIMO7-6", "OUTRO"])
    material = st.sidebar.selectbox("Material (Aço)", materiais_disponiveis)
    usina = st.sidebar.text_input("Usina", value="SHI")
    corrida = st.sidebar.text_input("Corrida", value="-")
    ce = st.sidebar.number_input("Carbono Equivalente (CE)", min_value=0.20, max_value=2.00, value=0.772, step=0.001, format="%.3f")
    
    col_hb1, col_hb2 = st.sidebar.columns(2)
    with col_hb1:
        hb_min = st.sidebar.number_input("HBW Mínima", min_value=100, max_value=600, value=235)
    with col_hb2:
        hb_max = st.sidebar.number_input("HBW Máxima", min_value=100, max_value=600, value=262)
        
    dimensoes = st.sidebar.text_input("Dimensões (ex: 32x3510, 250x750)", value="32x3510")
    peca = st.sidebar.selectbox("Grupo / Tipo de Peça", ["BARRA", "DISCO", "TARUGO", "EIXO", "CORPO", "ANEL", "TUBO", "OUTRO"])
    
    # Opções Avançadas (Expandable)
    with st.sidebar.expander("⚙️ Opções Avançadas (Têmpera / Forno)"):
        centro_trabalho = st.text_input("Centro de Trabalho / Forno", value="")
        centro_trabalho = centro_trabalho if centro_trabalho.strip() else None
        
        tempo_rv = st.number_input("Tempo de RV (min)", min_value=0, max_value=1800, value=0)
        tempo_rv_min = tempo_rv if tempo_rv > 0 else None
        
        te_temp = st.number_input("Temp. Têmpera (°C)", min_value=0, max_value=1200, value=0)
        te_temp = te_temp if te_temp > 0 else None
        
        te_tempo = st.number_input("Tempo Têmpera (min)", min_value=0, max_value=1800, value=0)
        te_tempo_min = te_tempo if te_tempo > 0 else None
        
        te_meio = st.selectbox("Meio de Resfriamento (Têmpera)", ["Histórico / Padrão", "ÁGUA", "ÓLEO", "AGUA_OLEO", "AR"])
        te_meio = None if te_meio == "Histórico / Padrão" else te_meio
        
        normalizado = st.checkbox("Normalização Prévia", value=False)

    # Botão para Executar Recomendação
    if st.sidebar.button("🚀 Gerar Recomendação", type="primary"):
        try:
            res = recomendar(
                modelo=modelo,
                material=material,
                usina=usina,
                ce=ce,
                hb_min=hb_min,
                hb_max=hb_max,
                dimensoes=dimensoes,
                corrida=corrida,
                peca=peca,
                centro_trabalho=centro_trabalho,
                tempo_rv_min=tempo_rv_min,
                te_temp=te_temp,
                te_tempo_min=te_tempo_min,
                te_meio=te_meio,
                normalizado_antes=normalizado,
                imprimir=False
            )
            
            # Exibição dos Resultados Principais
            st.subheader("🎯 Resultado da Recomendação")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Temperatura Recomendada", f"{res['temperatura_recomendada']} °C")
            c2.metric("Tempo de Encharque", f"{res['tempo_rv_min']} min")
            lo, hi = res['intervalo_80']
            c3.metric("Intervalo 80%", f"{lo} a {hi} °C")
            c4.metric("Nível de Confiança", res['confianca'])
            
            st.markdown("---")
            
            # Detalhes do Modelo e Avisos
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.subheader("📊 Componentes do Modelo")
                st.write(f"- **GBM (HistGradientBoosting):** {res['gbm']} °C")
                st.write(f"- **kNN (Mediana Ponderada):** {res['knn'] if res['knn'] is not None else 'N/A'} °C")
                st.write(f"- **Distância Média Vizinhos:** {res['distancia_vizinhos']}")
                
                if res['assumidos']:
                    st.info("**Valores Assumidos / Padrões:**\n\n" + "\n".join([f"- {a}" for a in res['assumidos']]))
                    
                if res['avisos']:
                    for av in res['avisos']:
                        st.warning(f"⚠️ {av}")

            with col_right:
                st.subheader("📈 Sensibilidade à Dureza-Alvo")
                hbs, ts = res['sensibilidade']
                df_sens = pd.DataFrame({"HBW": hbs, "Temp_Recomendada_C": ts})
                st.line_chart(df_sens.set_index("HBW"))
                st.write(f"- **Inclinação:** {res['inclinacao_C_por_HB']:.2f} °C / HBW")
                if not np.isnan(res['janela_especificacao_C']):
                    st.write(f"- **Janela Equivalente de Especificação:** ~{res['janela_especificacao_C']:.0f} °C")
            
            st.markdown("---")
            st.subheader("🔍 Vizinhos Históricos Próximos (kNN)")
            if len(res['vizinhos']) > 0:
                cols_viz = ['Ordem', 'Usina', 'Corrida', 'CE', 'HB', 'Dimensoes', 'T1', 't1', 'aprov_1a', 'Teq', 'dist']
                df_viz_view = res['vizinhos'][cols_viz].rename(columns={
                    'HB': 'HB_alvo', 'T1': 'T_1a_tent', 't1': 'min',
                    'aprov_1a': 'Aprov_1a', 'Teq': 'T_efetiva'
                }).round(2)
                st.dataframe(df_viz_view, use_container_width=True)

        except Exception as e:
            st.error(f"Erro ao gerar recomendação: {e}")