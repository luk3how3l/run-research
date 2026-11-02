"""Local browser annotation tool. Standard library only; never loads ML models."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import tempfile
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

from benchmark.manifest import sha256_file, validate
from benchmark.scoring import PROMPT

FIELDS = ['image_path', 'bib', 'sha256', 'status', 'target_identity', 'split', 'provenance']
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup = path.with_name(path.name + '.backup-' + secrets.token_hex(6))
        shutil.copy2(path, backup)
    fd, tmp = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class Collection:
    def __init__(self, root, images, csv_path):
        self.root = Path(root).resolve()
        self.csv_path = Path(csv_path).resolve()
        images = Path(images).resolve()
        if not images.is_relative_to(self.root):
            raise ValueError('Images must be inside the repository root')
        paths = sorted(p for p in images.rglob('*') if p.suffix.lower() in EXTENSIONS and p.is_file())
        if not paths:
            raise ValueError('No supported images found')
        old = {}
        if self.csv_path.exists():
            with self.csv_path.open(newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames != FIELDS:
                    raise ValueError('Unexpected annotation CSV columns')
                for row in reader:
                    if row['image_path'] in old:
                        raise ValueError('Duplicate CSV image path')
                    old[row['image_path']] = row
        self.rows = []
        hashes = set()
        for p in paths:
            if not p.resolve().is_relative_to(self.root):
                raise ValueError('Image symlink escapes root')
            rel = p.relative_to(self.root).as_posix()
            digest = sha256_file(p)
            if digest in hashes:
                raise ValueError('Duplicate image content: ' + rel)
            hashes.add(digest)
            row = old.pop(rel, None)
            if row and row['sha256'] != digest:
                raise ValueError('Image changed since annotation: ' + rel)
            self.rows.append(row or dict(zip(FIELDS, [rel, '', digest, 'pending', 'central foreground runner', 'development', ''])))
        if old:
            raise ValueError('CSV contains images missing from this collection; use the original image directory')

    def save(self, index, values):
        row = dict(self.rows[index])
        for k in ('bib', 'status', 'target_identity', 'split', 'provenance'):
            if not isinstance(values.get(k), str):
                raise ValueError('Missing text field: ' + k)
            row[k] = values[k].strip()
        if row['status'] not in {'reviewed', 'skipped', 'unreadable', 'ambiguous'}:
            raise ValueError('Invalid status')
        if row['split'] not in {'development', 'evaluation'}:
            raise ValueError('Invalid split')
        if not row['target_identity']:
            raise ValueError('Describe the target runner')
        if row['status'] == 'reviewed' and not re.fullmatch('[0-9]+', row['bib']):
            raise ValueError('Reviewed bib must contain digits only; leading zeros are preserved')
        if row['status'] != 'reviewed' and row['bib']:
            raise ValueError('Clear the bib for skipped/unreadable/ambiguous images')
        if row['split'] == 'evaluation' and not row['provenance']:
            raise ValueError('Evaluation images need a source/permission note')
        if sha256_file(self.root / row['image_path']) != row['sha256']:
            raise ValueError('Image changed; annotation refused')
        rows = list(self.rows)
        rows[index] = row
        import io
        output = io.StringIO(newline='')
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        atomic_write(self.csv_path, output.getvalue())
        self.rows = rows

    def export(self, destination, expected):
        if any(r['status'] != 'reviewed' for r in self.rows):
            raise ValueError('Every image must be reviewed with a digit label before export; unresolved images are not silently excluded')
        records = [dict(id='img-' + hashlib.sha256(r['image_path'].encode()).hexdigest()[:16],
                        path=r['image_path'], bib=r['bib'], sha256=r['sha256'],
                        split=r['split'], review_status='reviewed', target_identity=r['target_identity'],
                        provenance=r['provenance']) for r in self.rows]
        manifest = {'version': 1, 'prompt': PROMPT, 'records': records}
        errors, warnings = validate(manifest, self.root, expected)
        if errors:
            raise ValueError('\n'.join(errors))
        atomic_write(destination, json.dumps(manifest, indent=2) + '\n')
        return warnings


HTML = '''<!doctype html><meta charset="utf-8"><title>Bib annotation</title>
<style>body{font:16px system-ui;max-width:1100px;margin:20px auto;padding:10px}img{max-width:100%;max-height:65vh}input,select,button{font:inherit;margin:5px;padding:6px}#message{color:#a20}label{display:block}</style>
<h1>Bib annotation</h1><p>Label one designated runner. Never use predictions to choose the ground truth. Save before navigating.</p>
<p id="progress"></p><img id="photo"><form id="form">
<label>Bib digits <input id="bib" autocomplete="off" inputmode="numeric"></label>
<label>Status <select id="status"><option>reviewed</option><option>skipped</option><option>unreadable</option><option>ambiguous</option></select></label>
<label>Target runner <input id="target_identity" size="55"></label>
<label>Split <select id="split"><option>development</option><option>evaluation</option></select></label>
<label>Source / permission note <input id="provenance" size="65"></label>
<button type="button" id="prev">Previous</button><button>Save and next</button><button type="button" id="next">Next without saving</button></form><p id="message"></p>
<script>
let rows=[],i=0,token='TOKEN';const keys=['bib','status','target_identity','split','provenance'];const el=id=>document.getElementById(id);
function show(){const r=rows[i];el('progress').textContent=`${i+1}/${rows.length}: ${r.image_path} — ${rows.filter(x=>x.status==='reviewed').length} reviewed`;el('photo').src='/image?i='+i;keys.forEach(k=>el(k).value=(k==='status'&&r[k]==='pending')?'reviewed':r[k]);el('bib').focus();}
fetch('/rows').then(r=>r.json()).then(r=>{rows=r;i=Math.max(0,rows.findIndex(x=>x.status!=='reviewed'));show()});
el('prev').onclick=()=>{i=Math.max(0,i-1);show()};el('next').onclick=()=>{i=Math.min(rows.length-1,i+1);show()};
el('form').onsubmit=async e=>{e.preventDefault();let values={};keys.forEach(k=>values[k]=el(k).value);try{let r=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token,index:i,values})});let data=await r.json();if(!r.ok)throw Error(data.error);rows[i]=data;el('message').textContent='Saved';i=Math.min(rows.length-1,i+1);show()}catch(e){el('message').textContent=e.message}};
</script>'''


def handler(collection, token):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, data, content_type='application/json'):
            payload = data if isinstance(data, bytes) else json.dumps(data).encode()
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == '/':
                return self.reply(200, HTML.replace('TOKEN', token).encode(), 'text/html; charset=utf-8')
            if url.path == '/rows':
                return self.reply(200, collection.rows)
            if url.path == '/image':
                try:
                    i = int(parse_qs(url.query)['i'][0])
                    if not 0 <= i < len(collection.rows):
                        raise ValueError('Invalid index')
                    p = (collection.root / collection.rows[i]['image_path']).resolve()
                    if not p.is_relative_to(collection.root):
                        raise ValueError('Path escapes root')
                    import mimetypes
                    return self.reply(200, p.read_bytes(), mimetypes.guess_type(p)[0] or 'application/octet-stream')
                except (ValueError, KeyError, OSError):
                    return self.reply(400, {'error': 'Invalid image'})
            self.reply(404, {'error': 'Not found'})

        def do_POST(self):
            try:
                if self.path != '/save':
                    raise ValueError('Unknown endpoint')
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 32768:
                    raise ValueError('Invalid request size')
                data = json.loads(self.rfile.read(length))
                if not secrets.compare_digest(data.get('token', ''), token):
                    raise ValueError('Invalid session token')
                i = data['index']
                if type(i) is not int or not 0 <= i < len(collection.rows):
                    raise ValueError('Invalid index')
                collection.save(i, data['values'])
                self.reply(200, collection.rows[i])
            except (ValueError, KeyError, TypeError, OSError) as exc:
                self.reply(400, {'error': str(exc)})
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--images', default='media')
    parser.add_argument('--csv', default='annotations/bibs.csv')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--export', help='Export complete reviewed collection to manifest, then exit')
    parser.add_argument('--expected-count', type=int, help='Required with --export')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    try:
        c = Collection(root, root / args.images, root / args.csv)
        if args.export:
            if not args.expected_count or args.expected_count < 1:
                parser.error('--export requires positive --expected-count')
            if (root / args.export).resolve() == c.csv_path:
                parser.error('Export must not overwrite annotation CSV')
            for warning in c.export(root / args.export, args.expected_count):
                print('Warning:', warning)
            print('Exported', args.export)
            return
        server = HTTPServer(('127.0.0.1', args.port), handler(c, secrets.token_hex(32)))
        print(f'Open http://127.0.0.1:{args.port} — Ctrl+C stops; saved labels persist.', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    except (ValueError, OSError) as exc:
        parser.exit(1, str(exc) + '\n')


if __name__ == '__main__':
    main()
