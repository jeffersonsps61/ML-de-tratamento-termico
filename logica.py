# -*- coding: utf-8 -*-
"""Camada entre a interface e o modelo: valida entradas e formata a saída."""
import os
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
from recomendador_revenimento import carregar, recomendar  # noqa: E402

AQUI = os.path.dirname(os.path.abspath(__file__))
MODELO = carregar(os.path.join(AQUI, 'recomendador_rv.joblib'))

# Se False, esconde Ordem/Corrida na tabela de vizinhos (dados internos).
MOSTRAR_IDENTIFICADORES = False

CATS = MODELO['enc'].cats
MATERIAIS = [m for m in MODELO['materiais'] if m != 'OUTRO']
USINAS = [u for u in CATS['Usina'] if u not in ('OUTRO', 'nan')]
PECAS = [p for p in CATS['Peca'] if p != 'OUTRO']
FORNOS = [c for c in CATS['CT'] if c != 'OUTRO']
NAO_INFORMADO = '(não informado)'


def _num(v):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)


def calcular(material, usina, ce, hb_min, hb_max, dimensoes, peca, forno,
             tempo_rv, te_temp, te_tempo, te_meio, normalizado):
    """Retorna (markdown_resultado, figura, tabela_vizinhos)."""
    erros = []
    if not material:
        erros.append('informe o **material**')
    if not usina:
        erros.append('informe a **usina**')
    if ce is None:
        erros.append('informe o **CE**')
    if hb_min is None or hb_max is None:
        erros.append('informe a **faixa de dureza** (mín e máx)')
    elif hb_min >= hb_max:
        erros.append('a dureza **mínima** deve ser menor que a **máxima**')
    if not dimensoes or not str(dimensoes).strip():
        erros.append('informe as **dimensões** (ex.: 32x3510)')
    if erros:
        return '### ⚠️ Corrija os campos\n- ' + '\n- '.join(erros), None, None

    try:
        r = recomendar(
            MODELO, material=str(material), usina=str(usina), ce=float(ce),
            hb_min=float(hb_min), hb_max=float(hb_max), dimensoes=str(dimensoes).strip(),
            peca=peca or 'BARRA',
            centro_trabalho=None if forno in (None, '', NAO_INFORMADO) else forno,
            tempo_rv_min=_num(tempo_rv), te_temp=_num(te_temp), te_tempo_min=_num(te_tempo),
            te_meio=None if te_meio in (None, '', NAO_INFORMADO) else te_meio,
            normalizado_antes={'Sim': 1, 'Não': 0}.get(normalizado),
            imprimir=False)
    except ValueError as e:
        return f'### ⚠️ Entrada inválida\n{e}', None, None
    except Exception as e:  # noqa: BLE001
        return f'### ❌ Erro inesperado\n`{type(e).__name__}: {e}`', None, None

    lo, hi = r['intervalo_80']
    conf = r['confianca']
    icone = {'A': '🟢', 'M': '🟡'}.get(conf[0], '🔴')
    md = [
        '## Temperatura recomendada (1ª tentativa)',
        f"# {r['temperatura_recomendada']} °C por {r['tempo_rv_min']} min",
        f'**Intervalo calibrado de 80%:** {lo} a {hi} °C  ',
        f"**Confiança:** {icone} {conf}  ",
        f"<sub>GBM = {r['gbm']} °C · kNN = {r['knn']} °C · "
        f"distância média aos vizinhos = {r['distancia_vizinhos']}</sub>",
    ]
    if r['assumidos']:
        md.append('\n**Valores assumidos (não informados):** ' + '; '.join(r['assumidos']))
    for a in r['avisos']:
        md.append(f'\n⚠️ {a}')

    jan = r['janela_especificacao_C']
    inc = float(r['inclinacao_C_por_HB'])
    if not np.isnan(jan):
        md.append(f'\n**Sensibilidade:** {inc:.2f} °C por HBW. A faixa de dureza '
                  f'({hb_max - hb_min:.0f} HBW) equivale a ~{jan:.0f} °C de janela.')
        if jan < (hi - lo) / 2:
            md.append('\n⚠️ Janela de especificação estreita frente à incerteza: '
                      'risco de retrabalho elevado.')

    hbs, ts = r['sensibilidade']
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(hbs, ts, marker='o', ms=3, color='#1f77b4')
    ax.axvspan(hb_min, hb_max, color='#1f77b4', alpha=0.12, label='faixa especificada')
    ax.axhline(r['temperatura_recomendada'], color='gray', ls='--', lw=0.8)
    ax.set_xlabel('Dureza-alvo (HBW)')
    ax.set_ylabel('Temperatura recomendada (°C)')
    ax.set_title('Sensibilidade à dureza-alvo', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    v = r['vizinhos']
    tabela = None
    if len(v):
        cols = ['Ordem', 'Corrida', 'Usina', 'CE', 'HB', 'Dimensoes', 'T1', 't1', 'aprov_1a', 'Teq', 'dist']
        if not MOSTRAR_IDENTIFICADORES:
            cols = [c for c in cols if c not in ('Ordem', 'Corrida')]
        tabela = v[cols].rename(columns={
            'HB': 'Dureza alvo', 'Dimensoes': 'Dimensões', 'T1': 'T 1ª tent. (°C)',
            't1': 'Tempo (min)', 'aprov_1a': 'Aprovou de 1ª', 'Teq': 'T efetiva (°C)',
            'dist': 'Distância'}).round(2).reset_index(drop=True)
        tabela['Aprovou de 1ª'] = tabela['Aprovou de 1ª'].map({True: 'Sim', False: 'Não'})
    return '\n'.join(md), fig, tabela
