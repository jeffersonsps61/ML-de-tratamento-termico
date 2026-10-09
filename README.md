# Recomendador de Temperatura de Revenimento (RV) - v2.0

Sistema preditivo e interface web local para recomendação e calibração de temperaturas de revenimento no tratamento térmico de aços forjados/estruturais.

---

## 📌 Principais Funcionalidades

- **Predição Híbrida (GBM + kNN):** Combina *HistGradientBoostingRegressor* (com Conformal Quantile Regression - CQR) e busca por vizinhos históricos mais próximos deduplicados por corrida.
- **Normalização Física via Hollomon-Jaffe (HJP):** Converte ciclos de revenimento acumulados (incluindo retrabalhos) em uma temperatura equivalente ($T_{eq}$) de passe único.
- **Ajuste Fisiológico de Têpera:** Considera variáveis da etapa prévia de têmpera (temperatura, tempo, meio de resfriamento e normalização).
- **Interface Web Integrada:** Aplicação HTTP nativa em Python (sem dependência de frameworks externos como Flask ou Streamlit).
- **Modo Sombra e Registro (CSV):** Grava automaticamente todas as recomendações em `registro_recomendacoes.csv` para auditoria e retreinamento contínuo do modelo.

---

## 🛠️ Estrutura de Arquivos Recomendada

Certifique-se de que todos os arquivos estejam no mesmo diretório:

```text
.
├── df_completo.xlsx              # Base de dados de entrada para treino
├── recomendador_revenimento.py   # Script do modelo (módulo principal)
├── app_web.py                    # Aplicação do servidor web local
├── recomendador_rv.joblib        # Arquivo do modelo treinado (gerado automaticamente)
├── registro_recomendacoes.csv    # Histórico de recomendações (gerado automaticamente)
├── requirements.txt              # Lista de dependências Python
└── README.md                     # Documentação do projeto
```

> **Nota sobre o nome dos scripts:** Se os seus scripts estiverem salvos com sufixos (ex: `recomendador_revenimento_3.py` ou `app_2.py`), renomeie-os para `recomendador_revenimento.py` e `app_web.py` para garantir o funcionamento correto das importações.

---

## 🚀 Como Configurar e Executar

### 1. Pré-requisitos e Instalação
Certifique-se de ter o **Python 3.9+** instalado. Instale as dependências via `pip`:

```bash
pip install -r requirements.txt
```

### 2. Treinar o Modelo
Antes de rodar a interface web, você deve treinar o modelo para gerar o arquivo `.joblib`:

```bash
python recomendador_revenimento.py
```
Esse comando lerá a base `df_completo.xlsx`, executará a validação via *GroupKFold* por corrida e criará o arquivo `recomendador_rv.joblib`.

### 3. Iniciar a Interface Web
Para abrir a interface no seu navegador local:

```bash
python app_web.py
```
O navegador abrirá automaticamente no endereço: [http://127.0.0.1:8000](http://127.0.0.1:8000).

---

## 📊 Estrutura do Registro de Recomendações (CSV)

Cada recomendação efetuada via interface web é registrada em `registro_recomendacoes.csv` (formato com separador `;`, compatível com Microsoft Excel).

O arquivo contém os seguintes campos:
`id`, `data_hora`, `material`, `usina`, `corrida`, `ce`, `hb_min`, `hb_max`, `dimensoes`, `peca`, `forno`, `tempo_rv_min`, `T_recomendada`, `T_min_80`, `T_max_80`, `confianca`, `T_usada`, `resultado`, `observacao`.

---

## 🔒 Licença e Uso Interno
Uso restrito a análises internas e engenharia de processos.