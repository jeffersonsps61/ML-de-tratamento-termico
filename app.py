# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 14:00:39 2026

@author: jefferson.santos
"""

# -*- coding: utf-8 -*-
"""
INTERFACE WEB LOCAL - Recomendador de Revenimento

Como usar:
    python app_web.py
O navegador abre em http://127.0.0.1:8000  (acessível só no seu computador).

Requisitos: os arquivos abaixo na MESMA pasta deste script
    - recomendador_revenimento.py
    - recomendador_rv.joblib   (gerado com: python recomendador_revenimento.py)
Não precisa instalar nada além do que o recomendador já usa (usa só a biblioteca
padrão do Python para o servidor web).

Cada recomendação é registrada em  registro_recomendacoes.csv  (abre no Excel).
Na aba "Histórico" você preenche a temperatura realmente usada e o resultado:
esse registro é a base do "modo sombra" para medir o ganho real do sistema.
"""
import csv
import datetime
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
os.chdir(AQUI)

PORTA = int(os.environ.get('PORTA', 8000))
ARQ_MODELO = os.path.join(AQUI, 'recomendador_rv.joblib')
ARQ_LOG = os.path.join(AQUI, 'registro_recomendacoes.csv')
CAMPOS_LOG = ['id', 'data_hora', 'material', 'usina', 'corrida', 'ce', 'hb_min', 'hb_max',
              'dimensoes', 'peca', 'forno', 'tempo_rv_min', 'T_recomendada', 'T_min_80',
              'T_max_80', 'confianca', 'T_usada', 'resultado', 'observacao']

try:
  from recomendador_revenimento import carregar, recomendar
except Exception as e:  # pragma: no cover
  sys.exit(f'Não consegui importar recomendador_revenimento.py: {e}')

if not os.path.exists(ARQ_MODELO):
  sys.exit('Modelo não encontrado. Rode primeiro:  python recomendador_revenimento.py')

MODELO = carregar(ARQ_MODELO)
TRAVA = threading.Lock()
_t = MODELO['knn'].t
FAIXAS = dict(CE=(float(_t.CE.min()), float(_t.CE.max())),
              HB=(float(_t.HB.min()), float(_t.HB.max())))


# ==============================================================================
# UTILITÁRIOS
# ==============================================================================
def limpa(o):
  """Converte tipos numpy/NaN em JSON válido."""
  if isinstance(o, dict):
    return {str(k): limpa(v) for k, v in o.items()}
  if isinstance(o, (list, tuple)):
    return [limpa(v) for v in o]
  if isinstance(o, np.ndarray):
    return [limpa(v) for v in o.tolist()]
  if isinstance(o, (np.integer,)):
    return int(o)
  if isinstance(o, (float, np.floating)):
    f = float(o)
    return None if (np.isnan(f) or np.isinf(f)) else f
  if isinstance(o, (np.bool_,)):
    return bool(o)
  return o


def num(v, nome, obrigatorio=True):
  if v is None or str(v).strip() == '':
    if obrigatorio:
      raise ValueError(f'Informe o campo "{nome}".')
    return None
  try:
    return float(str(v).strip().replace(',', '.'))
  except ValueError:
    raise ValueError(f'Valor inválido em "{nome}": {v!r}')


def opcoes():
  mats = [m for m in MODELO['materiais'] if m != 'OUTRO']
  enc = MODELO['enc'].cats
  usinas = [u for u in enc['Usina'] if u not in ('OUTRO', 'nan')]
  fornos = [c for c in enc['CT'] if c not in ('OUTRO', 'nan')]
  pecas = [p for p in enc['Peca'] if p != 'OUTRO']
  return dict(materiais=mats, usinas=usinas, fornos=fornos, pecas=pecas,
              faixas=FAIXAS, n_treino=MODELO['n_treino'])


def ler_log():
  if not os.path.exists(ARQ_LOG):
    return []
  with open(ARQ_LOG, newline='', encoding='utf-8-sig') as f:
    return list(csv.DictReader(f, delimiter=';'))


def gravar_log(linhas):
  with open(ARQ_LOG, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=CAMPOS_LOG, delimiter=';')
    w.writeheader()
    w.writerows(linhas)


def registrar(entrada, res):
  linhas = ler_log()
  novo_id = (max([int(r['id']) for r in linhas]) + 1) if linhas else 1
  linhas.append(dict(
      id=novo_id, data_hora=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
      material=entrada['material'], usina=entrada['usina'], corrida=entrada['corrida'],
      ce=entrada['ce'], hb_min=entrada['hb_min'], hb_max=entrada['hb_max'],
      dimensoes=entrada['dimensoes'], peca=entrada['peca'], forno=entrada['forno'] or '',
      tempo_rv_min=res['tempo_rv_min'], T_recomendada=res['temperatura_recomendada'],
      T_min_80=res['intervalo_80'][0], T_max_80=res['intervalo_80'][1],
      confianca=res['confianca'].split(' ')[0], T_usada='', resultado='', observacao=''))
  gravar_log(linhas)
  return novo_id


def processar(d):
  material = str(d.get('material', '')).strip()
  usina = str(d.get('usina', '')).strip()
  dimensoes = str(d.get('dimensoes', '')).strip()
  if not material:
    raise ValueError('Informe o material.')
  if not dimensoes:
    raise ValueError('Informe as dimensões (ex.: 32x3510).')
  ce = num(d.get('ce'), 'CE')
  hb_min, hb_max = num(d.get('hb_min'), 'Dureza mínima'), num(d.get('hb_max'), 'Dureza máxima')
  if hb_min >= hb_max:
    raise ValueError('A dureza mínima deve ser menor que a máxima.')
  tempo = num(d.get('tempo_rv_min'), 'Tempo de RV', False)
  te_temp = num(d.get('te_temp'), 'Temperatura de têmpera', False)
  te_tempo = num(d.get('te_tempo_min'), 'Tempo de têmpera', False)
  norm = {'': None, 'sim': 1, 'nao': 0}.get(str(d.get('normalizado', '')), None)
  entrada = dict(material=material, usina=usina or 'nan', corrida=str(d.get('corrida', '')).strip() or '-',
                 ce=ce, hb_min=hb_min, hb_max=hb_max, dimensoes=dimensoes,
                 peca=str(d.get('peca', 'BARRA')) or 'BARRA',
                 forno=str(d.get('forno', '')).strip() or None)
  with TRAVA:
    r = recomendar(MODELO, material=entrada['material'], usina=entrada['usina'],
                   corrida=entrada['corrida'], ce=ce, hb_min=hb_min, hb_max=hb_max,
                   dimensoes=dimensoes, peca=entrada['peca'], centro_trabalho=entrada['forno'],
                   tempo_rv_min=tempo, te_temp=te_temp, te_tempo_min=te_tempo,
                   te_meio=(str(d.get('te_meio', '')).strip() or None),
                   normalizado_antes=norm, imprimir=False)
  extras = []
  lo, hi = FAIXAS['CE']
  if not lo <= ce <= hi:
    extras.append(f'CE {ce:.3f} fora da faixa do histórico ({lo:.2f} a {hi:.2f}): extrapolação.')
  lo, hi = FAIXAS['HB']
  if hb_min < lo or hb_max > hi:
    extras.append(f'Dureza fora da faixa do histórico ({lo:.0f} a {hi:.0f} HBW): extrapolação.')
  r['avisos'] = list(r['avisos']) + extras

  viz = r['vizinhos']
  cols = ['Ordem', 'Usina', 'Corrida', 'CE', 'HB', 'Dimensoes', 'T1', 't1', 'aprov_1a', 'Teq', 'dist']
  r['vizinhos'] = viz[cols].round(2).to_dict('records') if len(viz) else []
  hbs, ts = r['sensibilidade']
  r['sensibilidade'] = dict(hb=list(hbs), t=list(ts))
  r['id_registro'] = registrar(entrada, r)
  return limpa(r)


# ==============================================================================
# SERVIDOR
# ==============================================================================
class Handler(BaseHTTPRequestHandler):

  def log_message(self, *a):   # silencia o log padrão
    pass

  def _enviar(self, codigo, corpo, tipo='application/json; charset=utf-8', extra=None):
    if isinstance(corpo, (dict, list)):
      corpo = json.dumps(corpo, ensure_ascii=False)
    corpo = corpo.encode('utf-8') if isinstance(corpo, str) else corpo
    self.send_response(codigo)
    self.send_header('Content-Type', tipo)
    self.send_header('Content-Length', str(len(corpo)))
    for k, v in (extra or {}).items():
      self.send_header(k, v)
    self.end_headers()
    self.wfile.write(corpo)

  def do_GET(self):
    if self.path in ('/', '/index.html'):
      self._enviar(200, PAGINA, 'text/html; charset=utf-8')
    elif self.path == '/api/opcoes':
      self._enviar(200, limpa(opcoes()))
    elif self.path == '/api/historico':
      self._enviar(200, ler_log())
    elif self.path == '/api/historico.csv':
      with open(ARQ_LOG, 'rb') if os.path.exists(ARQ_LOG) else open(os.devnull, 'rb') as f:
        dados = f.read()
      self._enviar(200, dados, 'text/csv; charset=utf-8',
                   {'Content-Disposition': 'attachment; filename=registro_recomendacoes.csv'})
    else:
      self._enviar(404, {'erro': 'não encontrado'})

  def do_POST(self):
    try:
      n = int(self.headers.get('Content-Length', 0))
      d = json.loads(self.rfile.read(n) or b'{}')
      if self.path == '/api/recomendar':
        self._enviar(200, processar(d))
      elif self.path == '/api/historico/atualizar':
        linhas = ler_log()
        for r in linhas:
          if r['id'] == str(d.get('id')):
            for campo in ('T_usada', 'resultado', 'observacao'):
              if campo in d:
                r[campo] = str(d[campo])
        gravar_log(linhas)
        self._enviar(200, {'ok': True})
      else:
        self._enviar(404, {'erro': 'não encontrado'})
    except ValueError as e:
      self._enviar(400, {'erro': str(e)})
    except Exception as e:
      self._enviar(500, {'erro': f'Erro interno: {e}'})


PAGINA = r'''<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Recomendador de Revenimento</title>
<style>
:root{--bg:#f4f6f9;--card:#fff;--tx:#1b2430;--mut:#667385;--bd:#dde3ec;--pri:#1f5fbf;--pri2:#e8f0fc;
--ok:#1a7f4b;--okbg:#e4f5ec;--md:#9a6700;--mdbg:#fff4d6;--bx:#b42318;--bxbg:#fde8e6;--sh:0 1px 3px rgba(20,30,50,.08)}
@media(prefers-color-scheme:dark){:root{--bg:#10151c;--card:#182029;--tx:#e6ebf2;--mut:#93a1b4;--bd:#2a3644;
--pri:#6aa3ff;--pri2:#1c2b44;--ok:#4cc38a;--okbg:#14301f;--md:#e3b341;--mdbg:#3a2f10;--bx:#ff8a80;--bxbg:#3d1c1a}}
*{box-sizing:border-box}
body{margin:0;font:15px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--tx)}
header{padding:18px 24px;background:var(--card);border-bottom:1px solid var(--bd);display:flex;align-items:center;gap:16px;flex-wrap:wrap}
header h1{font-size:18px;margin:0;font-weight:650}
header small{color:var(--mut)}
nav{margin-left:auto;display:flex;gap:6px}
nav button{border:1px solid var(--bd);background:transparent;color:var(--tx);padding:7px 14px;border-radius:8px;cursor:pointer;font:inherit}
nav button.on{background:var(--pri);border-color:var(--pri);color:#fff}
main{max-width:1180px;margin:0 auto;padding:20px;display:grid;grid-template-columns:360px 1fr;gap:20px}
@media(max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--bd);border-radius:12px;padding:18px;box-shadow:var(--sh)}
.card h2{font-size:14px;margin:0 0 12px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut);font-weight:600}
label{display:block;font-size:13px;color:var(--mut);margin:10px 0 4px}
input,select{width:100%;padding:9px 10px;border:1px solid var(--bd);border-radius:8px;background:var(--bg);color:var(--tx);font:inherit}
input:focus,select:focus{outline:2px solid var(--pri);outline-offset:-1px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
details{margin-top:14px;border-top:1px solid var(--bd);padding-top:10px}
summary{cursor:pointer;color:var(--pri);font-weight:600}
.btn{margin-top:16px;width:100%;padding:12px;border:0;border-radius:10px;background:var(--pri);color:#fff;font:inherit;font-weight:650;cursor:pointer}
.btn:disabled{opacity:.6;cursor:wait}
.btn2{border:1px solid var(--bd);background:transparent;color:var(--tx);padding:7px 12px;border-radius:8px;cursor:pointer;font:inherit}
.hero{display:flex;align-items:flex-end;gap:18px;flex-wrap:wrap}
.big{font-size:54px;font-weight:700;line-height:1;letter-spacing:-.02em}
.big small{font-size:22px;font-weight:600;color:var(--mut)}
.sub{color:var(--mut);margin-bottom:6px}
.badge{display:inline-block;padding:4px 12px;border-radius:99px;font-weight:650;font-size:13px}
.ALTA{background:var(--okbg);color:var(--ok)}.MEDIA{background:var(--mdbg);color:var(--md)}.BAIXA{background:var(--bxbg);color:var(--bx)}
.alert{padding:10px 12px;border-radius:8px;background:var(--mdbg);color:var(--md);margin-top:8px;font-size:14px}
.alert.neg{background:var(--bxbg);color:var(--bx)}
.note{padding:10px 12px;border-radius:8px;background:var(--pri2);margin-top:8px;font-size:14px}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{padding:7px 8px;border-bottom:1px solid var(--bd);text-align:left;white-space:nowrap}
th{color:var(--mut);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.03em}
.scroll{overflow-x:auto}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:20px}
@media(max-width:900px){.grid2{grid-template-columns:1fr}}
.vazio{color:var(--mut);text-align:center;padding:60px 20px}
.err{color:var(--bx);margin-top:10px;font-weight:600}
.kv{display:grid;grid-template-columns:auto 1fr;gap:4px 14px;font-size:14px}.kv span:nth-child(odd){color:var(--mut)}
td input,td select{padding:5px 7px;font-size:13px}
.sim{color:var(--ok);font-weight:600}.nao{color:var(--bx);font-weight:600}
</style>
</head>
<body>
<header>
  <h1>Recomendador de Revenimento</h1><small id="info"></small>
  <nav><button id="t1" class="on" onclick="aba(1)">Nova recomendação</button>
       <button id="t2" onclick="aba(2)">Histórico</button></nav>
</header>

<main id="aba1">
  <section class="card">
    <h2>Dados da ordem</h2>
    <form id="f" onsubmit="enviar(event)" autocomplete="off">
      <div class="row">
        <div><label>Material *</label><input name="material" list="dl_mat" placeholder="4140" required></div>
        <div><label>Usina</label><input name="usina" list="dl_usi" placeholder="SHI"></div>
      </div>
      <div class="row">
        <div><label>Corrida</label><input name="corrida" placeholder="2520729"></div>
        <div><label>CE (carbono equiv.) *</label><input name="ce" inputmode="decimal" placeholder="0,772" required></div>
      </div>
      <div class="row">
        <div><label>Dureza mín. (HBW) *</label><input name="hb_min" inputmode="decimal" placeholder="235" required></div>
        <div><label>Dureza máx. (HBW) *</label><input name="hb_max" inputmode="decimal" placeholder="262" required></div>
      </div>
      <div class="row">
        <div><label>Dimensões *</label><input name="dimensoes" placeholder="32x3510" required></div>
        <div><label>Peça</label><select name="peca" id="sel_peca"></select></div>
      </div>
      <details>
        <summary>Processo (opcional)</summary>
        <div class="row">
          <div><label>Tempo de RV (min)</label><input name="tempo_rv_min" inputmode="decimal" placeholder="auto"></div>
          <div><label>Forno</label><select name="forno" id="sel_forno"><option value="">não informado</option></select></div>
        </div>
        <div class="row">
          <div><label>Têmpera: temp. (°C)</label><input name="te_temp" inputmode="decimal" placeholder="auto"></div>
          <div><label>Têmpera: tempo (min)</label><input name="te_tempo_min" inputmode="decimal" placeholder="auto"></div>
        </div>
        <div class="row">
          <div><label>Meio de têmpera</label>
            <select name="te_meio"><option value="">auto</option><option>Óleo</option><option>Água</option><option>Ar</option></select></div>
          <div><label>Normalização prévia</label>
            <select name="normalizado"><option value="">auto</option><option value="sim">Sim</option><option value="nao">Não</option></select></div>
        </div>
      </details>
      <button class="btn" id="btn">Calcular recomendação</button>
      <div class="err" id="erro"></div>
    </form>
    <datalist id="dl_mat"></datalist><datalist id="dl_usi"></datalist>
  </section>

  <section id="res"><div class="card vazio">Preencha os dados da ordem e clique em <b>Calcular recomendação</b>.</div></section>
</main>

<main id="aba2" style="display:none;grid-template-columns:1fr">
  <section class="card">
    <h2>Histórico de recomendações (modo sombra)</h2>
    <p style="color:var(--mut);margin-top:0">Depois do tratamento, informe a temperatura realmente usada e o resultado.
    Esses dados medem o desempenho real do sistema e servem para retreinar o modelo.</p>
    <p><a class="btn2" href="/api/historico.csv" style="text-decoration:none;display:inline-block">Baixar CSV (Excel)</a></p>
    <div class="scroll"><table id="tb"></table></div>
  </section>
</main>

<script>
const $=s=>document.querySelector(s);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const f1=(x,d=1)=>x==null?'-':Number(x).toFixed(d);

async function init(){
  const o=await (await fetch('/api/opcoes')).json();
  $('#dl_mat').innerHTML=o.materiais.map(m=>`<option value="${esc(m)}">`).join('');
  $('#dl_usi').innerHTML=o.usinas.map(m=>`<option value="${esc(m)}">`).join('');
  $('#sel_peca').innerHTML=o.pecas.map(p=>`<option ${p==='BARRA'?'selected':''}>${esc(p)}</option>`).join('');
  $('#sel_forno').innerHTML+=o.fornos.map(p=>`<option>${esc(p)}</option>`).join('');
  $('#info').textContent=`modelo treinado com ${o.n_treino} ordens`;
}
function aba(n){
  $('#aba1').style.display=n==1?'grid':'none';$('#aba2').style.display=n==2?'grid':'none';
  $('#t1').className=n==1?'on':'';$('#t2').className=n==2?'on':'';
  if(n==2)historico();
}
async function enviar(ev){
  ev.preventDefault();$('#erro').textContent='';$('#btn').disabled=true;$('#btn').textContent='Calculando...';
  const d=Object.fromEntries(new FormData($('#f')).entries());
  try{
    const r=await fetch('/api/recomendar',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});
    const j=await r.json();
    if(!r.ok){$('#erro').textContent=j.erro||'Erro';}else{render(j,d);}
  }catch(e){$('#erro').textContent='Falha de comunicação com o servidor.';}
  $('#btn').disabled=false;$('#btn').textContent='Calcular recomendação';
}
function render(r,d){
  const conf=r.confianca.split(' ')[0].replace('É','E');
  const [lo,hi]=r.intervalo_80;
  let h=`<div class="card"><div class="hero">
    <div><div class="sub">Temperatura de revenimento recomendada</div>
    <div class="big">${r.temperatura_recomendada}<small> °C</small></div></div>
    <div><div class="sub">Tempo</div><div class="big" style="font-size:34px">${r.tempo_rv_min}<small style="font-size:16px"> min</small></div></div>
    <div style="margin-left:auto;text-align:right"><div class="sub">Confiança</div><span class="badge ${conf}">${esc(r.confianca.split(' ')[0])}</span></div>
  </div>
  ${faixa(r.temperatura_recomendada,lo,hi)}
  <div class="kv" style="margin-top:12px"><span>Componentes</span><span>GBM ${f1(r.gbm,0)} °C &nbsp;|&nbsp; kNN ${f1(r.knn,0)} °C</span>
  <span>Distância aos vizinhos</span><span>${f1(r.distancia_vizinhos,2)}</span>
  <span>Registro</span><span>#${r.id_registro} salvo no histórico</span></div>`;
  if(conf==='BAIXA')h+=`<div class="alert neg">Confiança BAIXA: peça fora do que o histórico cobre bem. Valide com a engenharia antes de usar.</div>`;
  r.avisos.forEach(a=>h+=`<div class="alert">⚠ ${esc(a)}</div>`);
  if(r.assumidos.length)h+=`<div class="note"><b>Valores assumidos (não informados):</b><br>${r.assumidos.map(esc).join('<br>')}</div>`;
  h+=`</div><div class="grid2"><div class="card"><h2>Sensibilidade à dureza-alvo</h2>${grafico(r,d)}${textoSens(r,d)}</div>
  <div class="card"><h2>Como ler</h2><div style="font-size:14px;color:var(--mut)">
  <p style="margin-top:0"><b>Intervalo 80%</b>: faixa em que 80% dos ciclos que aprovaram no histórico se encontram (calibrado em validação). Não é limite de segurança.</p>
  <p><b>Confiança</b>: baseada na distância às ordens históricas mais parecidas. Em validação o erro médio foi ~16 °C (alta), ~18 °C (média) e ~23 °C (baixa).</p>
  <p style="margin-bottom:0"><b>Sensibilidade</b>: quanto a temperatura recomendada muda se a dureza-alvo mudar. É a curva do modelo, não uma medição de laboratório.</p></div></div></div>
  <div class="card" style="margin-top:20px"><h2>Ordens históricas mais parecidas (1 por corrida)</h2><div class="scroll">${tabela(r.vizinhos)}</div></div>`;
  $('#res').innerHTML=h;
}
function faixa(t,lo,hi){
  const a=Math.min(lo,t)-25,b=Math.max(hi,t)+25,p=x=>((x-a)/(b-a)*100).toFixed(1);
  return `<div style="margin-top:18px"><div style="position:relative;height:34px">
   <div style="position:absolute;top:14px;left:0;right:0;height:6px;border-radius:3px;background:var(--bd)"></div>
   <div style="position:absolute;top:11px;left:${p(lo)}%;width:${p(hi)-p(lo)}%;height:12px;border-radius:6px;background:var(--pri);opacity:.35"></div>
   <div style="position:absolute;top:5px;left:calc(${p(t)}% - 3px);width:6px;height:24px;border-radius:3px;background:var(--pri)"></div></div>
   <div style="display:flex;justify-content:space-between;font-size:13px;color:var(--mut)"><span>${lo} °C</span><span>intervalo 80%</span><span>${hi} °C</span></div></div>`;
}
function grafico(r,d){
  const x=r.sensibilidade.hb,y=r.sensibilidade.t,W=480,H=240,m={l:46,r:14,t:12,b:34};
  const x0=Math.min(...x),x1=Math.max(...x),y0=Math.min(...y)-10,y1=Math.max(...y)+10;
  const sx=v=>m.l+(v-x0)/(x1-x0)*(W-m.l-m.r),sy=v=>H-m.b-(v-y0)/(y1-y0)*(H-m.t-m.b);
  const pts=x.map((v,i)=>`${sx(v).toFixed(1)},${sy(y[i]).toFixed(1)}`).join(' ');
  const hmin=parseFloat(String(d.hb_min).replace(',','.')),hmax=parseFloat(String(d.hb_max).replace(',','.'));
  let s=`<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto">`;
  s+=`<rect x="${sx(hmin)}" y="${m.t}" width="${sx(hmax)-sx(hmin)}" height="${H-m.t-m.b}" fill="var(--pri)" opacity=".12"/>`;
  for(let i=0;i<=4;i++){const v=y0+(y1-y0)*i/4;s+=`<line x1="${m.l}" x2="${W-m.r}" y1="${sy(v)}" y2="${sy(v)}" stroke="var(--bd)"/><text x="${m.l-6}" y="${sy(v)+4}" text-anchor="end" font-size="11" fill="var(--mut)">${v.toFixed(0)}</text>`;}
  for(let i=0;i<=4;i++){const v=x0+(x1-x0)*i/4;s+=`<text x="${sx(v)}" y="${H-m.b+16}" text-anchor="middle" font-size="11" fill="var(--mut)">${v.toFixed(0)}</text>`;}
  s+=`<polyline points="${pts}" fill="none" stroke="var(--pri)" stroke-width="2.5"/>`;
  s+=`<text x="${W/2}" y="${H-4}" text-anchor="middle" font-size="11" fill="var(--mut)">dureza-alvo (HBW) — faixa especificada sombreada</text>`;
  s+=`<text x="12" y="${H/2}" font-size="11" fill="var(--mut)" transform="rotate(-90 12 ${H/2})" text-anchor="middle">temperatura (°C)</text></svg>`;
  return s;
}
function textoSens(r,d){
  let t=`<p style="font-size:14px;margin-bottom:0">Inclinação estimada: <b>${f1(r.inclinacao_C_por_HB,2)} °C por HBW</b>.`;
  if(r.janela_especificacao_C!=null){
    const lo=r.intervalo_80[0],hi=r.intervalo_80[1];
    t+=` A faixa de dureza especificada equivale a ~<b>${f1(r.janela_especificacao_C,0)} °C</b> de janela.`;
    if(r.janela_especificacao_C<(hi-lo)/2)t+=`</p><div class="alert neg">Janela de especificação estreita frente à incerteza: risco de retrabalho elevado.</div><p>`;
  }else t+=' Sem sensibilidade confiável para este caso.';
  return t+'</p>';
}
function tabela(v){
  if(!v.length)return '<div class="vazio">Sem vizinhos para este material.</div>';
  let h='<table><tr><th>Ordem</th><th>Usina</th><th>Corrida</th><th>CE</th><th>Dureza alvo</th><th>Dimensões</th><th>T 1ª tent.</th><th>Tempo</th><th>Aprovou 1ª</th><th>T efetiva</th><th>Distância</th></tr>';
  v.forEach(o=>h+=`<tr><td>${esc(o.Ordem)}</td><td>${esc(o.Usina==='nan'?'-':o.Usina)}</td><td>${esc(o.Corrida)}</td><td>${f1(o.CE,3)}</td><td>${f1(o.HB,1)}</td><td>${esc(o.Dimensoes)}</td><td>${f1(o.T1,0)} °C</td><td>${f1(o.t1,0)} min</td><td class="${o.aprov_1a?'sim':'nao'}">${o.aprov_1a?'Sim':'Não'}</td><td><b>${f1(o.Teq,0)} °C</b></td><td>${f1(o.dist,2)}</td></tr>`);
  return h+'</table>';
}
async function historico(){
  const L=await (await fetch('/api/historico')).json();
  if(!L.length){$('#tb').innerHTML='<tr><td class="vazio">Nenhuma recomendação registrada ainda.</td></tr>';return;}
  let h='<tr><th>#</th><th>Data</th><th>Material</th><th>Usina</th><th>Corrida</th><th>Dureza</th><th>Dim.</th><th>T recom.</th><th>Intervalo</th><th>Conf.</th><th>T usada</th><th>Resultado</th><th>Obs.</th></tr>';
  L.slice().reverse().forEach(r=>{
    const op=['','Aprovou de 1ª','Reprovou: dureza alta','Reprovou: dureza baixa'].map(x=>`<option ${r.resultado===x?'selected':''}>${x}</option>`).join('');
    h+=`<tr><td>${r.id}</td><td>${esc(r.data_hora)}</td><td>${esc(r.material)}</td><td>${esc(r.usina==='nan'?'-':r.usina)}</td><td>${esc(r.corrida)}</td>
    <td>${esc(r.hb_min)}-${esc(r.hb_max)}</td><td>${esc(r.dimensoes)}</td><td><b>${esc(r.T_recomendada)} °C</b></td><td>${esc(r.T_min_80)}-${esc(r.T_max_80)}</td><td>${esc(r.confianca)}</td>
    <td><input style="width:80px" value="${esc(r.T_usada)}" onchange="salvar(${r.id},'T_usada',this.value)"></td>
    <td><select onchange="salvar(${r.id},'resultado',this.value)">${op}</select></td>
    <td><input style="width:160px" value="${esc(r.observacao)}" onchange="salvar(${r.id},'observacao',this.value)"></td></tr>`;});
  $('#tb').innerHTML=h;
}
async function salvar(id,campo,valor){
  await fetch('/api/historico/atualizar',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id,[campo]:valor})});
}
init();
</script>
</body>
</html>
'''


def main():
  porta = PORTA
  while True:
    try:
      srv = ThreadingHTTPServer(('127.0.0.1', porta), Handler)
      break
    except OSError:
      porta += 1
  url = f'http://127.0.0.1:{porta}'
  print('=' * 60)
  print(' Recomendador de Revenimento - interface web')
  print(f' Abra no navegador: {url}')
  print(' (Ctrl+C para encerrar)')
  print('=' * 60)
  threading.Timer(1.0, lambda: webbrowser.open(url)).start()
  try:
    srv.serve_forever()
  except KeyboardInterrupt:
    print('\nEncerrado.')


if __name__ == '__main__':
  main()