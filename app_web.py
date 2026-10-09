import os
import sys
import json
import csv
import threading
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler, ThreadingHTTPServer

ARQ_LOG = 'registro_recomendacoes.csv'
TRAVA = threading.Lock()

try:
    import recomendador_revenimento as rr
    MODELO = rr.carregar()
except Exception as e:
    sys.exit(f'Erro ao carregar recomendador_revenimento.py: {e}')

def num(val, nome):
    if val is None or str(val).strip() == '':
        raise ValueError(f'O campo "{nome}" é obrigatório.')
    try:
        return float(val)
    except ValueError:
        raise ValueError(f'O valor do campo "{nome}" deve ser numérico.')

def gravar_log(dados_req, resp):
    with TRAVA:
        existe = os.path.exists(ARQ_LOG)
        with open(ARQ_LOG, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not existe:
                writer.writerow(['DataHora', 'Aco', 'C', 'Mn', 'Si', 'Cr', 'Ni', 'Mo', 
                                'Temp_Tempera', 'HB_Min', 'HB_Max', 'Temp_Rev_Recom', 'HB_Est', 'Confianca'])
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                dados_req.get('aco', 'N/A'),
                dados_req.get('c'), dados_req.get('mn'), dados_req.get('si'),
                dados_req.get('cr'), dados_req.get('ni'), dados_req.get('mo'),
                dados_req.get('temp_tempera'), dados_req.get('hb_min'), dados_req.get('hb_max'),
                resp.get('temp_rev'), resp.get('hb_estimada'), resp.get('confianca')
            ])

def ler_log():
    with TRAVA:
        if not os.path.exists(ARQ_LOG):
            return b""
        with open(ARQ_LOG, 'rb') as f:
            return f.read()

HTML_INTERFACE = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>Recomendador de Revenimento</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f4f6f9; }
        .container { max-width: 650px; margin: auto; background: white; padding: 25px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h2 { color: #333; text-align: center; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        label { font-weight: bold; font-size: 13px; display: block; margin-top: 8px; }
        input { width: 90%; padding: 8px; margin-top: 4px; border: 1px solid #ccc; border-radius: 4px; }
        button { width: 100%; padding: 12px; margin-top: 20px; background-color: #0056b3; color: white; border: none; border-radius: 4px; font-size: 16px; cursor: pointer; }
        button:hover { background-color: #003d80; }
        #resultado { margin-top: 20px; padding: 15px; border-radius: 4px; display: none; }
        .sucesso { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .erro { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .ALTA { color: green; font-weight: bold; }
        .MEDIA { color: orange; font-weight: bold; }
        .BAIXA { color: red; font-weight: bold; }
        .download-link { display: block; text-align: center; margin-top: 15px; }
    </style>
</head>
<body>
<div class="container">
    <h2>Recomendador de Revenimento</h2>
    <form id="formRecomendacao">
        <label>Identificação do Aço / Lote:</label>
        <input type="text" id="aco" value="42CrMo4">
        
        <div class="grid">
            <div>
                <label>% Carbono (C):</label>
                <input type="number" step="0.01" id="c" value="0.42">
                <label>% Manganês (Mn):</label>
                <input type="number" step="0.01" id="mn" value="0.75">
                <label>% Silício (Si):</label>
                <input type="number" step="0.01" id="si" value="0.25">
                <label>% Cromo (Cr):</label>
                <input type="number" step="0.01" id="cr" value="1.05">
            </div>
            <div>
                <label>% Níquel (Ni):</label>
                <input type="number" step="0.01" id="ni" value="0.15">
                <label>% Molibdênio (Mo):</label>
                <input type="number" step="0.01" id="mo" value="0.22">
                <label>Temp. Têmpera (°C):</label>
                <input type="number" id="temp_tempera" value="850">
                <label>Dureza Mínima (HBW):</label>
                <input type="number" id="hb_min" value="280">
            </div>
        </div>
        <label>Dureza Máxima (HBW):</label>
        <input type="number" id="hb_max" value="320">
        
        <button type="button" onclick="calcular()">Calcular Recomendação</button>
    </form>

    <div id="resultado"></div>
    <a class="download-link" href="/api/historico.csv" target="_blank">Baixar Histórico (CSV)</a>
</div>

<script>
function calcular() {
    const d = {
        aco: document.getElementById('aco').value,
        c: document.getElementById('c').value,
        mn: document.getElementById('mn').value,
        si: document.getElementById('si').value,
        cr: document.getElementById('cr').value,
        ni: document.getElementById('ni').value,
        mo: document.getElementById('mo').value,
        temp_tempera: document.getElementById('temp_tempera').value,
        hb_min: document.getElementById('hb_min').value,
        hb_max: document.getElementById('hb_max').value
    };

    fetch('/api/recomendar', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(d)
    })
    .then(r => r.json().then(data => ({status: r.status, body: data})))
    .then(res => {
        const div = document.getElementById('resultado');
        div.style.display = 'block';
        if (res.status === 200) {
            const r = res.body;
            const confClass = r.confianca.split(' ')[0].replace('É','E').toUpperCase();
            div.className = 'sucesso';
            div.innerHTML = `<strong>Recomendação:</strong><br>
                Temperatura de Revenimento: <b>${r.temp_rev} °C</b><br>
                Dureza Estimada: <b>${r.hb_estimada} HBW</b><br>
                Confiança: <span class="${confClass}">${r.confianca}</span>`;
        } else {
            div.className = 'erro';
            div.innerHTML = `<strong>Erro:</strong> ${res.body.erro}`;
        }
    })
    .catch(() => {
        const div = document.getElementById('resultado');
        div.style.display = 'block';
        div.className = 'erro';
        div.innerHTML = 'Erro na comunicação com o servidor.';
    });
}
</script>
</body>
</html>
"""

class Handler(BaseHTTPRequestHandler):
    def _enviar(self, codigo, conteudo, tipo='application/json', headers=None):
        self.send_response(codigo)
        self.send_header('Content-Type', tipo)
        if headers:
            for k, v in headers.items():
                self.send_header(k, v)
        self.end_headers()
        if isinstance(conteudo, str):
            conteudo = conteudo.encode('utf-8')
        self.wfile.write(conteudo)

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            self._enviar(200, HTML_INTERFACE, 'text/html; charset=utf-8')
        elif self.path == '/api/historico.csv':
            dados = ler_log()
            self._enviar(200, dados, 'text/csv; charset=utf-8',
                         {'Content-Disposition': 'attachment; filename=registro_recomendacoes.csv'})
        else:
            self._enviar(404, json.dumps({'erro': 'Rota não encontrada'}))

    def do_POST(self):
        if self.path == '/api/recomendar':
            length = int(self.headers.get('Content-Length', 0))
            corpo = self.rfile.read(length)
            try:
                d = json.loads(corpo.decode('utf-8'))
                
                # Conversão e Validação das entradas
                c = num(d.get('c'), 'Carbono (C)')
                mn = num(d.get('mn'), 'Manganês (Mn)')
                si = num(d.get('si'), 'Silício (Si)')
                cr = num(d.get('cr'), 'Cromo (Cr)')
                ni = num(d.get('ni'), 'Níquel (Ni)')
                mo = num(d.get('mo'), 'Molibdênio (Mo)')
                temp_tempera = num(d.get('temp_tempera'), 'Temp. Têmpera')
                hb_min = num(d.get('hb_min'), 'Dureza mínima')
                hb_max = num(d.get('hb_max'), 'Dureza máxima')

                if hb_min >= hb_max:
                    raise ValueError('A dureza mínima deve ser menor que a máxima.')

                res = rr.recomendar(MODELO, c, mn, si, cr, ni, mo, temp_tempera, hb_min, hb_max)
                gravar_log(d, res)
                self._enviar(200, json.dumps(res))

            except ValueError as ve:
                self._enviar(400, json.dumps({'erro': str(ve)}))
            except Exception as e:
                self._enviar(500, json.dumps({'erro': f'Erro interno: {str(e)}'}))
        else:
            self._enviar(404, json.dumps({'erro': 'Rota não encontrada'}))

def iniciar_servidor(porta=8080):
    server = ThreadingHTTPServer(('0.0.0.0', porta), Handler)
    print(f"Servidor iniciado na porta {porta}. Aceda a http://localhost:{porta}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor encerrado.")

if __name__ == '__main__':
    iniciar_servidor()