import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ARQ_MODELO = 'modelo_revenimento.pkl'

# Dados base para treino por omissão (Aços e parâmetros de revenimento)
DADOS_TREINO_PADRAO = [
    # Aço, C, Mn, Si, Cr, Ni, Mo, Temp_Temp, Temp_Rev, Tempo_h, HB
    ['42CrMo4', 0.42, 0.75, 0.25, 1.05, 0.15, 0.22, 850, 520, 2.0, 340],
    ['42CrMo4', 0.42, 0.75, 0.25, 1.05, 0.15, 0.22, 850, 560, 2.0, 300],
    ['42CrMo4', 0.42, 0.75, 0.25, 1.05, 0.15, 0.22, 850, 600, 2.0, 260],
    ['42CrMo4', 0.42, 0.75, 0.25, 1.05, 0.15, 0.22, 850, 640, 2.0, 220],
    ['SAE 4340', 0.40, 0.70, 0.25, 0.80, 1.83, 0.25, 840, 500, 2.0, 380],
    ['SAE 4340', 0.40, 0.70, 0.25, 0.80, 1.83, 0.25, 840, 550, 2.0, 340],
    ['SAE 4340', 0.40, 0.70, 0.25, 0.80, 1.83, 0.25, 840, 600, 2.0, 290],
    ['SAE 1045', 0.45, 0.75, 0.25, 0.05, 0.05, 0.02, 840, 500, 1.5, 250],
    ['SAE 1045', 0.45, 0.75, 0.25, 0.05, 0.05, 0.02, 840, 550, 1.5, 220],
    ['SAE 1045', 0.45, 0.75, 0.25, 0.05, 0.05, 0.02, 840, 600, 1.5, 195],
]

COLUNAS_FEATURES = ['C', 'Mn', 'Si', 'Cr', 'Ni', 'Mo', 'Temp_Temp', 'Temp_Rev', 'Tempo_h']

def treinar_modelo():
    df = pd.DataFrame(DADOS_TREINO_PADRAO, columns=['Aco'] + COLUNAS_FEATURES + ['HB'])
    X = df[COLUNAS_FEATURES]
    y = df['HB']
    
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X, y)
    
    with open(ARQ_MODELO, 'wb') as f:
        pickle.dump(modelo, f)
    return modelo

def carregar():
    if os.path.exists(ARQ_MODELO):
        try:
            with open(ARQ_MODELO, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return treinar_modelo()
    else:
        return treinar_modelo()

def recomendar(modelo, c, mn, si, cr, ni, mo, temp_tempera, hb_min, hb_max, tempo_h=2.0):
    hb_alvo = (hb_min + hb_max) / 2.0
    melhor_temp = None
    menor_dif = float('inf')
    
    # Busca da melhor temperatura de revenimento entre 400°C e 680°C
    for temp_rev in range(400, 685, 5):
        entrada = pd.DataFrame([[c, mn, si, cr, ni, mo, temp_tempera, temp_rev, tempo_h]], 
                               columns=COLUNAS_FEATURES)
        hb_pred = modelo.predict(entrada)[0]
        dif = abs(hb_pred - hb_alvo)
        
        if dif < menor_dif:
            menor_dif = dif
            melhor_temp = temp_rev
            hb_estimada = hb_pred

    # Validação de confiança
    confianca = "ALTA"
    if hb_estimada < hb_min or hb_estimada > hb_max:
        confianca = "BAIXA (extrapolação: validar com engenharia)"
    elif menor_dif > 15:
        confianca = "MEDIA"

    return {
        'temp_rev': melhor_temp,
        'hb_estimada': round(float(hb_estimada), 1),
        'confianca': confianca
    }

def executar(retreinar=False):
    if retreinar:
        treinar_modelo()
    print("Modelo carregado e pronto para operação.")

if __name__ == '__main__':
    executar(retreinar=True)