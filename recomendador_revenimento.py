# -*- coding: utf-8 -*-
"""
RECOMENDADOR DE TEMPERATURA DE REVENIMENTO (RV) - v2.0 (refatoração completa)

Dependências: pandas, numpy, scikit-learn (>=1.3), openpyxl, joblib.
(sem XGBoost/CatBoost: o HistGradientBoosting do scikit-learn cobre quantis,
 categóricas e NaN nativamente, e reduz problemas de instalação.)

O QUE MUDOU EM RELAÇÃO À Rev.17/18  (cada item foi medido em GroupKFold por corrida)
  1. RÓTULO FÍSICO: o ciclo aprovado de ordens com retrabalho é convertido numa
     temperatura equivalente de passe único via parâmetro de Hollomon-Jaffe
     (efeito cumulativo dos revenimentos), em vez de usar a temperatura da
     última etapa (que era enviesada para cima).
  2. TÊMPERA ENTRA COMO ATRIBUTO (temperatura, tempo, meio, normalização prévia),
     além do forno (Centro Trabalho) e do tempo de RV.
  3. CORREÇÃO DE BUGS DE TREINO x INFERÊNCIA:
       - Delta_Dureza_Tolerancia = (máx - mín), não (máx - mín)/2.
       - Dimensões com faixas ("160x3007/3010") são lidas corretamente.
       - Categorias desconhecidas geram AVISO (antes viravam NaN em silêncio).
  4. SEM PESO DE AMOSTRA INVERTIDO e com limpeza de outliers (faixas físicas).
  5. INCERTEZA CALIBRADA: conformal quantile regression (CQR) com cobertura
     verificada em validação, no lugar de "janela segura" não calibrada.
  6. KNN ÚNICO E COERENTE: escala global, pesos calibrados por validação,
     vizinhos deduplicados por corrida, confiança calibrada com dados
     (leave-one-corrida-out) em vez de cortes arbitrários.
  7. SENSIBILIDADE VÁLIDA: curva "dureza-alvo -> temperatura recomendada"
     (efeito real da especificação), no lugar de sobrescrever a dureza-alvo
     para simular a temperatura.
  8. MODELO PERSISTIDO (joblib): treina uma vez, recomenda em milissegundos.

COMO USAR
  - Treinar + validar + salvar:  python recomendador_revenimento.py
  - Em outro script:  from recomendador_revenimento import carregar, recomendar
"""
import os
import re
import warnings
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HGB
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

warnings.filterwarnings('once')

# ==============================================================================
# 0. CONFIGURAÇÃO
# ==============================================================================
CFG = dict(
    ARQUIVO='df_completo.xlsx',
    MODELO='recomendador_rv.joblib',
    C_HJ=20.0,                 # constante de Hollomon-Jaffe p/ aços de baixa liga
    FAIXA_T_RV=(450, 720),     # °C plausíveis para revenimento
    FAIXA_HB=(150, 450),       # dureza-alvo plausível (HBW)
    FAIXA_CE=(0.40, 1.20),
    FAIXA_TEMPO=(20, 1800),    # min de encharque plausíveis
    MIN_FREQ_CAT=15,           # categorias raras -> 'OUTRO'
    MIN_FREQ_MATERIAL=25,
    N_FOLDS=5,
    ALPHA=0.20,                # intervalo de 80%
    SEED=42,
    K_VIZ=10,
    # hiperparâmetros (busca em grade feita em GroupKFold por corrida)
    HGB=dict(max_depth=6, learning_rate=0.06, max_iter=250,
             min_samples_leaf=10, l2_regularization=10.0),
    HGB_Q=dict(max_depth=4, learning_rate=0.06, max_iter=150,
               min_samples_leaf=20, l2_regularization=10.0),
)

CATS = ['Mat', 'Usina', 'Peca', 'CT', 'TE_m']
NUMS = ['CE', 'HB', 'TOL', 'lD1', 'lD2', 'n_dim', 'TE_T', 'TE_t', 'NO', 'lt']
FEATS = [c + '_c' for c in CATS] + NUMS
CAT_MASK = [True] * len(CATS) + [False] * len(NUMS)


# ==============================================================================
# 1. FÍSICA E NORMALIZAÇÕES
# ==============================================================================
def hjp(temp_c, tempo_min):
  """Parâmetro de Hollomon-Jaffe: P = T[K] * (C + log10(t[h]))."""
  return (np.asarray(temp_c, float) + 273.15) * (
      CFG['C_HJ'] + np.log10(np.asarray(tempo_min, float) / 60.0))


def temp_de_hjp(p, tempo_min):
  """Inverte o HJP: temperatura (°C) que produz P para um dado tempo (min)."""
  return np.asarray(p, float) / (
      CFG['C_HJ'] + np.log10(np.asarray(tempo_min, float) / 60.0)) - 273.15


def arredonda10(x):
  return int(np.round(x / 10.0) * 10)


def normaliza_material(x):
  x = str(x).upper().strip()
  if x.endswith('.0'):
    x = x[:-2]
  x = x.replace('MOD', '').replace('+H', '').strip()
  if x == '4130M':
    return '4130'
  if x == '8630M':
    return '8630'
  if x in ('42CRMOS4', '42CRMO4V'):
    return '42CRMO4'
  if x.startswith('18CRNI'):
    return '18CRNIMO7-6'
  return x


def normaliza_meio(x):
  s = str(x).upper()
  if 'ÁGUA' in s or 'AGUA' in s:
    return 'AGUA_OLEO' if ('ÓLEO' in s or 'OLEO' in s) else 'AGUA'
  if 'ÓLEO' in s or 'OLEO' in s:
    return 'OLEO'
  if s.startswith('AR'):
    return 'AR'
  return 'NA'


def grupo_peca(x):
  m = re.search(r'(BARRA|DISCO|TARUGO|EIXO|CORPO|ANEL|TUBO)', str(x).upper())
  return m.group(1) if m else 'OUTRO'


def geometria(dim):
  """
  Lê strings como '35x55x170', '898x275', '317x185/260', '160x3007/3010/3020'.
  Cada token separado por 'x' pode ter alternativas com '/': usa-se o maior.
  Retorna (menor dimensão, 2ª menor, nº de tokens). A menor dimensão é a
  proxy da seção crítica de têmpera.
  """
  if not isinstance(dim, str):
    return np.nan, np.nan, 0
  tokens = []
  for parte in re.split(r'[x×*]', dim.lower().replace(',', '.')):
    vals = [float(v) for v in re.findall(r'\d+(?:\.\d+)?', parte) if float(v) > 0]
    if vals:
      tokens.append(max(vals))
  if not tokens:
    return np.nan, np.nan, 0
  ts = sorted(tokens)
  return ts[0], (ts[1] if len(ts) > 1 else ts[0]), len(tokens)


# ==============================================================================
# 2. CONSTRUÇÃO DA TABELA POR ORDEM (1 linha = 1 ordem)
# ==============================================================================
def construir_tabela(df):
  """
  Para cada ordem aprovada em algum RV monta:
    - atributos conhecidos ANTES do tratamento (material, CE, dureza-alvo, dim.)
    - contexto da têmpera que antecede o 1º RV
    - 1ª tentativa de RV (T1, t1, aprovou?)
    - RÓTULO: temperatura equivalente de passe único (Teq) via HJP cumulativo.
  """
  df = df.sort_values(['Ordem', 'Seq_Operacao_Ordem']).reset_index(drop=True)
  linhas = []
  for ordem, g in df.groupby('Ordem', sort=False):
    rvs = g[(g['Tratamento Térmico'] == 'RV') & (g['Temperatura'] > 0)]
    if rvs.empty:
      continue
    aprov = rvs[rvs['Status Revenimento'].astype(str).str.contains('Aprovado')]
    if aprov.empty:
      continue  # ordens nunca aprovadas não têm rótulo confiável
    primeira = rvs.iloc[0]
    passos = rvs.loc[:aprov.index[0]]

    # efeito cumulativo dos revenimentos até a aprovação (HJP aditivo em tempo-equivalente)
    p_acum = 0.0
    for _, s in passos.iterrows():
      t = s['Tempo Encharque']
      if not (CFG['FAIXA_TEMPO'][0] <= t <= CFG['FAIXA_TEMPO'][1]):
        p_acum = np.nan
        break
      tk = s['Temperatura'] + 273.15
      t_eq_h = 10 ** (p_acum / tk - CFG['C_HJ']) if p_acum > 0 else 0.0
      p_acum = tk * (CFG['C_HJ'] + np.log10(t_eq_h + t / 60.0))

    antes = g[g.index < primeira.name]
    te = antes[antes['Tratamento Térmico'] == 'TE']
    te = te.iloc[-1] if len(te) else None
    d1, d2, n_dim = geometria(primeira['Dimensões'])

    linhas.append(dict(
        Ordem=ordem, Corrida=primeira['Corrida'],
        Mat=normaliza_material(primeira['Material']),
        Usina=str(primeira['Usina']), Peca=grupo_peca(primeira['Peça']),
        CT=str(primeira['Centro Trabalho']),
        CE=primeira['CE Material'], HB=primeira['Dureza_Alvo_Media_HBW'],
        TOL=primeira['Delta_Dureza_Tolerancia_HBW'],
        Dimensoes=primeira['Dimensões'], D1=d1, D2=d2, n_dim=n_dim,
        T1=primeira['Temperatura'], t1=primeira['Tempo Encharque'],
        aprov_1a=bool('Aprovado' in str(primeira['Status Revenimento'])),
        Tfin=passos.iloc[-1]['Temperatura'], P_eq=p_acum, n_rv=len(passos),
        TE_T=te['Temperatura'] if te is not None else np.nan,
        TE_t=te['Tempo Encharque'] if te is not None else np.nan,
        TE_m=normaliza_meio(te['Meio Resfriamento']) if te is not None else 'NA',
        NO=int((antes['Tratamento Térmico'] == 'NO').any()),
    ))
  return pd.DataFrame(linhas)


def limpar_tabela(t):
  n0 = len(t)
  ok = (
      t['T1'].between(*CFG['FAIXA_T_RV']) & t['Tfin'].between(*CFG['FAIXA_T_RV'])
      & t['HB'].between(*CFG['FAIXA_HB']) & t['CE'].between(*CFG['FAIXA_CE'])
      & t['D1'].notna() & t['t1'].between(*CFG['FAIXA_TEMPO'])
      & t['P_eq'].notna() & t['TOL'].notna() & (t['TOL'] > 0)
  )
  t = t[ok].copy().reset_index(drop=True)
  t['Teq'] = temp_de_hjp(t['P_eq'], t['t1'])       # rótulo (°C, para o tempo t1)
  t['lD1'], t['lD2'] = np.log(t['D1']), np.log(t['D2'])
  t['lt'] = np.log(t['t1'])
  t['Grupo'] = t['Corrida'].where(t['Corrida'].notna(), t['Ordem']).astype(str)
  # materiais/categorias raros -> OUTRO
  for col, minf in [('Mat', CFG['MIN_FREQ_MATERIAL']), ('Usina', CFG['MIN_FREQ_CAT']),
                    ('CT', CFG['MIN_FREQ_CAT']), ('TE_m', 5), ('Peca', CFG['MIN_FREQ_CAT'])]:
    vc = t[col].value_counts()
    t[col] = np.where(t[col].map(vc) < minf, 'OUTRO', t[col])
  print(f'[dados] ordens com rótulo: {n0} | após limpeza: {len(t)} '
        f'({len(t)/n0:.0%}) | aprovadas de 1ª: {t.aprov_1a.mean():.1%}')
  return t


# ==============================================================================
# 3. CODIFICAÇÃO DE CATEGORIAS (com aviso para categorias desconhecidas)
# ==============================================================================
@dataclass
class Codificador:
  cats: dict = field(default_factory=dict)

  def fit(self, t):
    for c in CATS:
      self.cats[c] = {v: i for i, v in enumerate(sorted(t[c].unique()))}
    return self

  def transform(self, t, avisos=None):
    t = t.copy()
    for c in CATS:
      mapa = self.cats[c]

      def cod(v, c=c, mapa=mapa):
        if v is None or (isinstance(v, float) and np.isnan(v)):
          return np.nan
        if v in mapa:
          return mapa[v]
        if avisos is not None:
          avisos.append(f"categoria nunca vista em '{c}': {v!r} -> tratada como "
                        f"{'OUTRO' if 'OUTRO' in mapa else 'desconhecida (NaN)'}")
        return mapa.get('OUTRO', np.nan)

      t[c + '_c'] = t[c].map(cod).astype(float)
    return t


# ==============================================================================
# 4. MODELOS: PONTUAL + QUANTIS COM CONFORMALIZAÇÃO (CQR)
# ==============================================================================
def novo_hgb(params, seed, **extra):
  return HGB(categorical_features=CAT_MASK, random_state=seed, **params, **extra)


def ajusta_cqr(X, y, grupos, seed):
  """Conformalized Quantile Regression (Romano et al., 2019), split por corrida."""
  a = CFG['ALPHA']
  gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
  tr, ca = next(gss.split(X, y, grupos))
  lo = novo_hgb(CFG['HGB_Q'], seed, loss='quantile', quantile=a / 2).fit(X.iloc[tr], y.iloc[tr])
  hi = novo_hgb(CFG['HGB_Q'], seed, loss='quantile', quantile=1 - a / 2).fit(X.iloc[tr], y.iloc[tr])
  lo_c, hi_c = lo.predict(X.iloc[ca]), hi.predict(X.iloc[ca])
  escore = np.maximum(np.minimum(lo_c, hi_c) - y.iloc[ca], y.iloc[ca] - np.maximum(lo_c, hi_c))
  n = len(escore)
  q = np.quantile(escore, min(1.0, (1 - a) * (1 + 1 / n)), method='higher')
  return lo, hi, float(q)


def preve_intervalo(lo, hi, q, X, ponto):
  l, h = lo.predict(X), hi.predict(X)
  inf = np.minimum(np.minimum(l, h) - q, ponto)
  sup = np.maximum(np.maximum(l, h) + q, ponto)
  return inf, sup


# ==============================================================================
# 5. KNN COERENTE (escala global, pesos calibrados, 1 vizinho por corrida)
# ==============================================================================
KNN_COLS = ['CE', 'HB', 'lD1']
PENAL_USINA = 0.5


class VizinhosHistoricos:

  def __init__(self, pesos=(1.5, 2.0, 1.0)):
    self.pesos = np.array(pesos, float)

  def fit(self, t):
    self.t = t.reset_index(drop=True).copy()
    self.mu = self.t[KNN_COLS].mean().values
    self.sd = self.t[KNN_COLS].std().replace(0, 1).values
    self.Z = self._z(self.t[KNN_COLS].values)
    self.mat = self.t['Mat'].values
    self.usina = self.t['Usina'].values
    self.grupo = pd.factorize(self.t['Grupo'])[0]
    self.idx_mat = {m: np.where(self.mat == m)[0] for m in np.unique(self.mat)}
    return self

  def _z(self, X):
    return ((X - self.mu) / self.sd) * self.pesos

  def _top(self, mat, usina, zq, k, grupo_excl=None):
    """Índices e distâncias dos k vizinhos (1 por corrida) do mesmo material."""
    idx = self.idx_mat.get(mat, np.array([], int))
    if grupo_excl is not None:
      idx = idx[self.grupo[idx] != grupo_excl]
    if len(idx) == 0:
      return idx, np.array([])
    d = np.sqrt(((self.Z[idx] - zq) ** 2).sum(1)) + PENAL_USINA * (self.usina[idx] != usina)
    ordem = np.argsort(d, kind='stable')
    _, primeiro = np.unique(self.grupo[idx][ordem], return_index=True)
    sel = ordem[np.sort(primeiro)][:k]
    return idx[sel], d[sel]

  def consulta(self, x, k=None):
    """x: dict com Mat, Usina, CE, HB, lD1. Retorna DataFrame ordenado por distância."""
    k = k or CFG['K_VIZ']
    zq = self._z(np.array([[x['CE'], x['HB'], x['lD1']]], float))[0]
    idx, d = self._top(x['Mat'], x['Usina'], zq, k)
    return self.t.iloc[idx].assign(dist=d)

  @staticmethod
  def estima(viz, col='Teq'):
    if len(viz) == 0:
      return np.nan
    w = 1.0 / (viz['dist'].values + 0.1)
    v = viz[col].values
    o = np.argsort(v)
    cw = np.cumsum(w[o]) / w.sum()
    return float(v[o][np.searchsorted(cw, 0.5)])   # mediana ponderada

  def loco(self, k=None):
    """Leave-one-corrida-out: predição kNN e distância média (top-5) p/ cada ordem."""
    k = k or CFG['K_VIZ']
    n = len(self.t)
    pred, dmean = np.full(n, np.nan), np.full(n, np.nan)
    teq = self.t['Teq'].values
    for i in range(n):
      idx, d = self._top(self.mat[i], self.usina[i], self.Z[i], k, self.grupo[i])
      if len(idx):
        w = 1.0 / (d + 0.1)
        o = np.argsort(teq[idx])
        cw = np.cumsum(w[o]) / w.sum()
        pred[i] = teq[idx][o][np.searchsorted(cw, 0.5)]
        dmean[i] = d[:5].mean()
    return pred, dmean


def calibra_pesos_knn(t):
  """Escolhe pesos (CE, HB, seção) que minimizam o MAE LOCO do kNN."""
  melhor = (np.inf, None)
  for wce in (1.0, 1.5, 2.5):
    for whb in (1.0, 2.0, 3.0):
      for wd in (0.5, 1.0, 1.5):
        v = VizinhosHistoricos((wce, whb, wd)).fit(t)
        pred, _ = v.loco()
        mae = np.nanmean(np.abs(pred - t['Teq'].values))
        if mae < melhor[0]:
          melhor = (mae, (wce, whb, wd))
  print(f'[kNN] pesos calibrados (CE, HB, seção) = {melhor[1]} | MAE LOCO = {melhor[0]:.1f} °C')
  return melhor[1]


# ==============================================================================
# 6. VALIDAÇÃO (GroupKFold por corrida) COM BASELINES E MÉTRICAS DE NEGÓCIO
# ==============================================================================
def relatorio_metricas(nome, pred, t, lo=None, hi=None):
  a = t['aprov_1a'].values
  e_ok = np.abs(pred - t['T1'].values)[a]                 # 1ª aprovadas: T1 funcionou
  rw = ~a
  err_rw = np.abs(pred - t['Teq'].values)[rw]
  err_op = np.abs(t['T1'].values - t['Teq'].values)[rw]   # erro do operador nos retrabalhos
  linha = dict(
      modelo=nome,
      MAE_aprov_1a=e_ok.mean(), dentro_10=np.mean(e_ok <= 10) * 100,
      dentro_20=np.mean(e_ok <= 20) * 100, dentro_30=np.mean(e_ok <= 30) * 100,
      MAE_retrab=err_rw.mean(), melhor_que_operador=np.mean(err_rw < err_op) * 100)
  if lo is not None:
    linha['cobertura_%'] = np.mean((t['Teq'].values >= lo) & (t['Teq'].values <= hi)) * 100
    linha['T1_aprov_dentro_%'] = np.mean((t['T1'].values[a] >= lo[a]) & (t['T1'].values[a] <= hi[a])) * 100
    linha['largura'] = np.mean(hi - lo)
  return linha


def validar(t, enc):
  tt = enc.transform(t)
  X, y, g = tt[FEATS], tt['Teq'], tt['Grupo']
  n = len(tt)
  oof = np.zeros(n)
  lo, hi = np.zeros(n), np.zeros(n)
  for tr, va in GroupKFold(CFG['N_FOLDS']).split(X, y, g):
    pt = novo_hgb(CFG['HGB'], CFG['SEED']).fit(X.iloc[tr], y.iloc[tr])
    ponto = pt.predict(X.iloc[va])
    qlo, qhi, q = ajusta_cqr(X.iloc[tr], y.iloc[tr], g.iloc[tr], CFG['SEED'])
    lo[va], hi[va] = preve_intervalo(qlo, qhi, q, X.iloc[va], ponto)
    oof[va] = ponto

  # baselines
  tab = np.zeros(n)
  tt['hb_bin'] = pd.cut(tt['HB'], [0, 230, 260, 290, 320, 360, 500]).astype(str)
  for tr, va in GroupKFold(CFG['N_FOLDS']).split(X, y, g):
    med = tt.iloc[tr].groupby(['Mat', 'hb_bin'])['Tfin'].median()
    gm = tt.iloc[tr]['Tfin'].median()
    tab[va] = [med.get((r.Mat, r.hb_bin), gm) for r in tt.iloc[va].itertuples()]

  knn = VizinhosHistoricos(CFG['PESOS_KNN']).fit(t)
  p_knn, d_knn = knn.loco()
  p_knn = np.where(np.isnan(p_knn), oof, p_knn)

  # blend GBM + kNN (peso escolhido no OOF)
  grade = np.linspace(0, 1, 11)
  maes = [np.abs((1 - w) * oof + w * p_knn - tt['Teq'].values).mean() for w in grade]
  w_knn = float(grade[int(np.argmin(maes))])
  final = (1 - w_knn) * oof + w_knn * p_knn
  CFG['PESO_KNN_BLEND'] = w_knn

  linhas = [
      relatorio_metricas('Mediana global', np.full(n, tt['Tfin'].median()), tt),
      relatorio_metricas('Tabela material x faixa HB', tab, tt),
      relatorio_metricas('kNN (LOCO)', p_knn, tt),
      relatorio_metricas('GBM', oof, tt, lo, hi),
      relatorio_metricas(f'FINAL (GBM + {w_knn:.0%} kNN)', final, tt, lo + (final - oof), hi + (final - oof)),
  ]
  res = pd.DataFrame(linhas).set_index('modelo')
  pd.set_option('display.width', 200)
  print('\n' + '=' * 100)
  print('VALIDAÇÃO - GroupKFold por corrida (nenhuma corrida aparece em treino e teste)')
  print('  MAE_aprov_1a  : erro vs. temperatura usada nas ordens aprovadas de 1ª (T1 comprovadamente boa)')
  print('  dentro_X      : % das recomendações a no máx. X °C dessa temperatura')
  print('  MAE_retrab    : erro vs. ciclo efetivo nas ordens que precisaram de retrabalho')
  print('  melhor_que_op : % dos retrabalhos em que a recomendação ficaria mais perto do ciclo efetivo')
  print('                  que a 1ª tentativa do operador')
  print('=' * 100)
  print(res.round(1).to_string())
  op = np.abs(tt['T1'].values - tt['Teq'].values)[~tt['aprov_1a'].values].mean()
  print(f'\nReferência: nos retrabalhos a 1ª tentativa do operador ficou em média {op:.1f} °C '
        f'do ciclo efetivo.')

  # confiança calibrada com dados (distância kNN LOCO x erro do modelo final)
  erro = np.abs(final - tt['Teq'].values)
  ok = ~np.isnan(d_knn)
  q_med, q_alt = np.nanquantile(d_knn, [0.50, 0.85])
  b = np.where(d_knn <= q_med, 'ALTA', np.where(d_knn <= q_alt, 'MÉDIA', 'BAIXA'))
  print('\nCONFIANÇA CALIBRADA (distância média aos 5 vizinhos de outras corridas):')
  for nome in ('ALTA', 'MÉDIA', 'BAIXA'):
    mk = (b == nome) & ok
    print(f'  {nome:6s} n={mk.sum():4d} | MAE do modelo final = {erro[mk].mean():5.1f} °C | '
          f'dentro de 20 °C = {np.mean(erro[mk] <= 20) * 100:4.1f}%')
  CFG['LIM_CONF'] = (float(q_med), float(q_alt))
  return res


# ==============================================================================
# 7. TREINAMENTO FINAL E PERSISTÊNCIA
# ==============================================================================
def treinar(validar_modelo=True, salvar=True):
  print('Lendo', CFG['ARQUIVO'], '...')
  bruto = pd.read_excel(CFG['ARQUIVO'])
  t = limpar_tabela(construir_tabela(bruto))
  enc = Codificador().fit(t)

  CFG['PESOS_KNN'] = calibra_pesos_knn(t)
  CFG['PESO_KNN_BLEND'], CFG['LIM_CONF'] = 0.0, (1.0, 1.5)
  if validar_modelo:
    validar(t, enc)

  tt = enc.transform(t)
  X, y, g = tt[FEATS], tt['Teq'], tt['Grupo']
  ponto = novo_hgb(CFG['HGB'], CFG['SEED']).fit(X, y)
  qlo, qhi, q = ajusta_cqr(X, y, g, CFG['SEED'])
  knn = VizinhosHistoricos(CFG['PESOS_KNN']).fit(t)

  # valores padrão (moda por material) para campos opcionais da têmpera
  padroes = {}
  for mat, gm in t.groupby('Mat'):
    padroes[mat] = dict(
        TE_T=float(gm['TE_T'].median()), TE_t=float(gm['TE_t'].median()),
        TE_m=gm['TE_m'].mode().iat[0], NO=int(round(gm['NO'].mean())),
        CT=gm['CT'].mode().iat[0], Peca=gm['Peca'].mode().iat[0])
  modelo = dict(ponto=ponto, qlo=qlo, qhi=qhi, q=q, enc=enc, knn=knn, padroes=padroes,
                cfg={k: CFG[k] for k in ('PESOS_KNN', 'PESO_KNN_BLEND', 'LIM_CONF')},
                n_treino=len(t), materiais=sorted(t['Mat'].unique()))
  if salvar:
    joblib.dump(modelo, CFG['MODELO'])
    print(f"\n[modelo] salvo em {CFG['MODELO']} ({len(t)} ordens)")
  return modelo


def carregar(caminho=None):
  m = joblib.load(caminho or CFG['MODELO'])
  CFG.update(m['cfg'])
  return m


# ==============================================================================
# 8. RECOMENDAÇÃO
# ==============================================================================
def recomendar(modelo, material, usina, ce, hb_min, hb_max, dimensoes,
               corrida='-', peca='BARRA', centro_trabalho=None, tempo_rv_min=None,
               te_temp=None, te_tempo_min=None, te_meio=None, normalizado_antes=None,
               imprimir=True):
  """
  dimensoes : texto no MESMO formato da base (ex.: '32x3510', '250x750', '160x3007/3010').
  Campos opcionais (têmpera, tempo de RV, forno): se omitidos, usa-se o padrão histórico
  do material e isso é informado no relatório.
  """
  avisos, assumidos = [], []
  mat = normaliza_material(material)
  pad = modelo['padroes'].get(mat) or modelo['padroes'].get('OUTRO')
  if pad is None:
    pad = next(iter(modelo['padroes'].values()))
  if mat not in modelo['materiais']:
    avisos.append(f"material '{material}' fora do histórico -> tratado como OUTRO "
                  f"(confiança reduzida)")

  def usar(valor, chave, rotulo):
    if valor is None:
      assumidos.append(f'{rotulo} = {pad[chave]}')
      return pad[chave]
    return valor

  d1, d2, n_dim = geometria(dimensoes)
  if np.isnan(d1):
    raise ValueError(f"não consegui ler as dimensões: {dimensoes!r}")
  hb = (hb_min + hb_max) / 2.0
  tol = hb_max - hb_min                         # MESMA definição do treino (máx - mín)

  x = dict(Mat=mat, Usina=str(usina), Peca=grupo_peca(peca) if peca else pad['Peca'],
           CT=centro_trabalho if centro_trabalho else None,
           CE=ce, HB=hb, TOL=tol, lD1=np.log(d1), lD2=np.log(d2), n_dim=n_dim,
           TE_T=usar(te_temp, 'TE_T', 'temp. têmpera (°C)'),
           TE_t=usar(te_tempo_min, 'TE_t', 'tempo têmpera (min)'),
           TE_m=normaliza_meio(te_meio) if te_meio else usar(None, 'TE_m', 'meio de têmpera'),
           NO=int(normalizado_antes) if normalizado_antes is not None else usar(None, 'NO', 'normalização prévia'))
  if centro_trabalho is None:
    assumidos.append('forno (Centro Trabalho) = não informado (o modelo trata como ausente)')

  viz = modelo['knn'].consulta(dict(Mat=x['Mat'], Usina=x['Usina'], CE=ce, HB=hb, lD1=x['lD1']))
  if tempo_rv_min is None:
    tempo_rv_min = float(viz['t1'].median()) if len(viz) else float(np.exp(
        modelo['knn'].t['lt'].median()))
    assumidos.append(f'tempo de RV = {tempo_rv_min:.0f} min (mediana dos vizinhos)')
  x['lt'] = np.log(tempo_rv_min)

  def prever(xd):
    df = modelo['enc'].transform(pd.DataFrame([xd]), avisos)
    ponto = float(modelo['ponto'].predict(df[FEATS])[0])
    return df, ponto

  df_in, p_gbm = prever(x)
  inf, sup = preve_intervalo(modelo['qlo'], modelo['qhi'], modelo['q'], df_in[FEATS], np.array([p_gbm]))
  p_knn = VizinhosHistoricos.estima(viz)
  w = modelo['cfg']['PESO_KNN_BLEND']
  central = (1 - w) * p_gbm + w * p_knn if not np.isnan(p_knn) else p_gbm
  inf, sup = float(inf[0]) + (central - p_gbm), float(sup[0]) + (central - p_gbm)

  # confiança calibrada
  dmean = float(viz['dist'].head(5).mean()) if len(viz) else np.inf
  l1, l2 = modelo['cfg']['LIM_CONF']
  conf = 'ALTA' if dmean <= l1 else ('MÉDIA' if dmean <= l2 else 'BAIXA (extrapolação: validar com engenharia)')
  if not np.isnan(p_knn) and abs(p_knn - p_gbm) > 30:
    avisos.append(f'GBM ({p_gbm:.0f}) e kNN ({p_knn:.0f}) divergem >30 °C: revisar manualmente')

  # sensibilidade à dureza-alvo (efeito real da especificação), monotonizada
  grade_hb = np.linspace(hb - 30, hb + 30, 13)
  curva = []
  for h in grade_hb:
    xs = dict(x, HB=float(h))
    curva.append(prever(xs)[1])
  iso = IsotonicRegression(increasing=False).fit(grade_hb, curva)
  curva_s = iso.predict(grade_hb)
  inclin = np.polyfit(grade_hb, curva_s, 1)[0]            # °C por HBW
  janela_t = (hb_max - hb_min) / abs(inclin) if inclin < -0.05 else np.nan

  res = dict(temperatura_recomendada=arredonda10(central), intervalo_80=(int(np.floor(inf / 10) * 10),
             int(np.ceil(sup / 10) * 10)), tempo_rv_min=round(tempo_rv_min), confianca=conf,
             gbm=round(p_gbm, 1), knn=None if np.isnan(p_knn) else round(p_knn, 1),
             distancia_vizinhos=round(dmean, 2), avisos=sorted(set(avisos)),
             assumidos=assumidos, vizinhos=viz, sensibilidade=(grade_hb, curva_s),
             inclinacao_C_por_HB=inclin, janela_especificacao_C=janela_t)
  if imprimir:
    imprimir_relatorio(res, material, usina, corrida, ce, hb_min, hb_max, dimensoes)
  return res


def imprimir_relatorio(r, material, usina, corrida, ce, hb_min, hb_max, dim):
  print('\n' + '=' * 84)
  print('  RECOMENDAÇÃO DE REVENIMENTO (1ª TENTATIVA)')
  print('=' * 84)
  print(f'Entrada : SAE {material} | usina {usina} | corrida {corrida} | CE {ce:.3f} | '
        f'dureza {hb_min:.0f}-{hb_max:.0f} HBW | dim. {dim}')
  print(f"\n>>> TEMPERATURA RECOMENDADA : {r['temperatura_recomendada']} °C por {r['tempo_rv_min']} min")
  lo, hi = r['intervalo_80']
  print(f'    Intervalo calibrado 80%  : {lo} a {hi} °C (faixa histórica de ciclos que aprovaram)')
  print(f"    Confiança                : {r['confianca']} (distância média aos 5 vizinhos = {r['distancia_vizinhos']})")
  print(f"    Componentes              : GBM = {r['gbm']} °C | kNN (mediana ponderada) = {r['knn']} °C")
  if r['assumidos']:
    print('\nValores ASSUMIDOS (não informados): ' + '; '.join(r['assumidos']))
  for a in r['avisos']:
    print(f'⚠ {a}')

  print('\nSENSIBILIDADE À DUREZA-ALVO (efeito da especificação na temperatura recomendada):')
  hbs, ts = r['sensibilidade']
  for h, tv in list(zip(hbs, ts))[::2]:
    print(f'   {h:6.1f} HBW  ->  {tv:6.1f} °C')
  print(f"   inclinação estimada pelo modelo: {r['inclinacao_C_por_HB']:.2f} °C por HBW", end='')
  if not np.isnan(r['janela_especificacao_C']):
    print(f"  | a faixa de dureza ({hb_max - hb_min:.0f} HBW) equivale a ~{r['janela_especificacao_C']:.0f} °C de janela")
    if r['janela_especificacao_C'] < (hi - lo) / 2:
      print('   ⚠ janela de especificação estreita frente à incerteza: risco de retrabalho elevado.')
  else:
    print('  (sem sensibilidade confiável)')

  v = r['vizinhos']
  if len(v):
    print('\nVIZINHOS HISTÓRICOS (1 por corrida, mesmo material):')
    cols = ['Ordem', 'Usina', 'Corrida', 'CE', 'HB', 'Dimensoes', 'T1', 't1', 'aprov_1a', 'Teq', 'dist']
    print(v[cols].rename(columns={'HB': 'HB_alvo', 'T1': 'T_1a_tent', 't1': 'min', 'aprov_1a': 'aprov_1a',
                                  'Teq': 'T_efetiva'}).round(2).to_string(index=False))
  print('=' * 84)


# ==============================================================================
# 9. EXECUÇÃO
# ==============================================================================
def executar(retreinar=True):
  """Treina (ou carrega) o modelo e roda dois exemplos."""
  if retreinar or not os.path.exists(CFG['MODELO']):
    modelo = treinar(validar_modelo=True, salvar=True)
  else:
    modelo = carregar()

  # exemplo 1 (Rev.18): SAE 4140, usina SHI
  recomendar(modelo, material='4140', usina='SHI', corrida='2520729', ce=0.772,
             hb_min=235, hb_max=262, dimensoes='32x3510', peca='BARRA')

  # exemplo 2 (Rev.17): SAE 4340, usina VMT
  recomendar(modelo, material='4340', usina='VMT', corrida='4870025', ce=0.852,
             hb_min=269, hb_max=302, dimensoes='250x750', peca='BARRA')
  return modelo


if __name__ == '__main__':
  # Importa o próprio arquivo como módulo: assim as classes salvas no .joblib
  # ficam registradas como 'recomendador_revenimento.X' (e não '__main__.X'),
  # o que permite carregar() em qualquer outro script.
  import sys
  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
  import recomendador_revenimento as _rr
  _rr.executar(retreinar=True)
