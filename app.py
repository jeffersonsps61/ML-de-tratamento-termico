# -*- coding: utf-8 -*-
import gradio as gr

import logica as L

DESCRICAO = f"""
# 🔥 Recomendador de Temperatura de Revenimento
Sugere a temperatura da **1ª tentativa de revenimento (RV)** a partir de material, usina, CE,
dureza-alvo e dimensões, com intervalo calibrado de 80% e os casos históricos mais parecidos.
Modelo treinado com **{L.MODELO['n_treino']} ordens** aprovadas.
> Ferramenta de apoio à decisão: valide sempre com a engenharia de processo.
"""

with gr.Blocks(title='Recomendador de Revenimento') as demo:
    gr.Markdown(DESCRICAO)
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown('### Dados da ordem')
            material = gr.Dropdown(L.MATERIAIS, value='4140', label='Material (SAE)',
                                   allow_custom_value=True)
            usina = gr.Dropdown(L.USINAS, value=L.USINAS[0], label='Usina',
                                allow_custom_value=True)
            ce = gr.Number(label='CE (carbono equivalente)', value=0.772, precision=3,
                           minimum=0.3, maximum=1.3)
            with gr.Row():
                hb_min = gr.Number(label='Dureza mín (HBW)', value=235)
                hb_max = gr.Number(label='Dureza máx (HBW)', value=262)
            dimensoes = gr.Textbox(label='Dimensões', value='32x3510',
                                   info="Mesmo formato da base: 32x3510, 250x750, 160x3007/3010")
            peca = gr.Dropdown(L.PECAS, value='BARRA', label='Tipo de peça',
                               allow_custom_value=True)

            with gr.Accordion('Opcionais (se vazio, usa o padrão histórico do material)', open=False):
                forno = gr.Dropdown([L.NAO_INFORMADO] + L.FORNOS, value=L.NAO_INFORMADO,
                                    label='Forno (Centro de Trabalho)')
                tempo_rv = gr.Number(label='Tempo de RV (min)', value=None)
                te_temp = gr.Number(label='Temperatura de têmpera (°C)', value=None)
                te_tempo = gr.Number(label='Tempo de têmpera (min)', value=None)
                te_meio = gr.Dropdown([L.NAO_INFORMADO, 'Água', 'Óleo'], value=L.NAO_INFORMADO,
                                      label='Meio de têmpera')
                normalizado = gr.Dropdown([L.NAO_INFORMADO, 'Sim', 'Não'], value=L.NAO_INFORMADO,
                                          label='Normalização prévia')
            btn = gr.Button('Calcular recomendação', variant='primary')

        with gr.Column(scale=1):
            saida_md = gr.Markdown()
            saida_fig = gr.Plot(label='Sensibilidade')

    gr.Markdown('### Vizinhos históricos (1 por corrida, mesmo material)')
    saida_tab = gr.Dataframe(interactive=False)

    entradas = [material, usina, ce, hb_min, hb_max, dimensoes, peca, forno,
                tempo_rv, te_temp, te_tempo, te_meio, normalizado]
    saidas = [saida_md, saida_fig, saida_tab]
    btn.click(L.calcular, entradas, saidas)
    demo.load(L.calcular, entradas, saidas)   # já abre com o exemplo calculado

if __name__ == '__main__':
    demo.launch()
