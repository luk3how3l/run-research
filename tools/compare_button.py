"""Local browser button for offline comparisons; no GUI packages required."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import threading

PAGE = '''<!doctype html><meta charset="utf-8"><title>Bib comparison</title>
<style>body{font:18px system-ui;max-width:700px;margin:50px auto}label{display:block;margin:16px 0}input{width:100%;padding:8px}button,select{padding:12px}pre{white-space:pre-wrap}</style>
<h1>Race bib comparison</h1><p>Legacy cascade · SmolVLM · Pixtral</p>
<form id="form"><label>Device <select name="device"><option>cpu</option><option>cuda</option></select></label>
<label>Allocated GPU UUID<input name="gpu_uuid"></label>
<label>SmolVLM local directory (optional)<input name="smolvlm_dir"></label>
<label>Pixtral local directory (optional)<input name="pixtral_dir"></label>
<button id="run">Run comparison</button></form><pre id="status">Ready</pre>
<p>Results print in the launching terminal and are saved under results/comparison-*/comparison.csv.
CPU runs mark Pixtral blocked. CUDA requires an allocated GPU and local weights.</p>
<script>form.onsubmit=async e=>{e.preventDefault();let r=await fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(Object.fromEntries(new FormData(form)))});statusUpdate(await r.json())};
function statusUpdate(s){document.getElementById('status').textContent=s.message;document.getElementById('run').disabled=s.running}
setInterval(async()=>{statusUpdate(await(await fetch('/status')).json())},1000)</script>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    process = None
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, value, code=200):
            data = json.dumps(value).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == '/':
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(PAGE.encode())
            elif self.path == '/status':
                running = process is not None and process.poll() is None
                self.reply(dict(running=running, message='Running. See terminal for progress.' if running else
                                'Ready' if process is None else f'Finished (exit {process.returncode}). See terminal and CSV.'))
            else:
                self.send_error(404)

        def do_POST(self):
            nonlocal process
            # Require same-origin JSON; reject cross-site browser requests.
            origin = self.headers.get('Origin')
            if self.path != '/run' or self.headers.get('Content-Type') != 'application/json' or (origin and origin != 'http://' + self.headers.get('Host', '')):
                self.send_error(403)
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length < 16384:
                    raise ValueError('Invalid request size')
                values = json.loads(self.rfile.read(length))
                if values.get('device') not in ('cpu', 'cuda'):
                    raise ValueError('Select cpu or cuda')
                command = [sys.executable, '-m', 'benchmark.runner', '--device', values['device']]
                for name in ('gpu_uuid', 'smolvlm_dir', 'pixtral_dir'):
                    value = values.get(name, '').strip()
                    if value:
                        command += ['--' + name.replace('_', '-'), value]
                with lock:
                    if process is not None and process.poll() is None:
                        self.reply(dict(running=True, message='Comparison already running'), 409)
                        return
                    process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1])
                self.reply(dict(running=True, message='Started. See terminal for progress.'))
            except (ValueError, TypeError, AttributeError) as exc:
                self.reply(dict(running=False, message=str(exc)), 400)

        def log_message(self, *args):
            pass

    print(f'Open http://127.0.0.1:{args.port} and click Run comparison.', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
