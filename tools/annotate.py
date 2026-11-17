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


def sync_directory(path):
    """Make directory entry changes durable alongside fsynced file contents."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


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
        sync_directory(path.parent)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


class Collection:
    def __init__(self, root, images, csv_path):
        self.root = Path(root).resolve()
        self.csv_path = self.safe_path(csv_path)
        self.journal = self.safe_path(str(self.csv_path) + '.transaction')
        images = self.safe_path(images)
        for name in ('unannotated', 'annotated', 'needs_review'):
            self.safe_path(self.root / 'media' / name).mkdir(parents=True, exist_ok=True)
        self.recover()
        old = []
        if self.csv_path.exists():
            with self.csv_path.open(newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames != FIELDS:
                    raise ValueError('Unexpected annotation CSV columns')
                old = list(reader)
        by_hash, by_path = {}, {}
        for row in old:
            self.safe_path(self.root / row['image_path'])
            if row['sha256'] in by_hash or row['image_path'] in by_path:
                raise ValueError('Duplicate CSV image identity or path')
            by_hash[row['sha256']] = row
            by_path[row['image_path']] = row
        paths = set()
        # Scan only the requested queue, managed directories, and recorded paths.
        # Imported datasets outside that scope are never automatically moved.
        for directory in (images, *(self.root / 'media' / n for n in
                                    ('unannotated', 'annotated', 'needs_review'))):
            paths.update(p for p in directory.rglob('*')
                         if p.suffix.lower() in EXTENSIONS and p.is_file())
        for row in old:
            path = self.safe_path(self.root / row['image_path'])
            if path.is_file():
                paths.add(path)
        found = {}
        for path in sorted(paths):
            self.safe_path(path)
            rel = path.relative_to(self.root).as_posix()
            digest = sha256_file(path)
            if rel in by_path and by_path[rel]['sha256'] != digest:
                raise ValueError('Image changed since annotation: ' + rel)
            if digest in found:
                raise ValueError('Duplicate image content: ' + rel)
            found[digest] = rel
        self.rows = []
        for row in old:
            if row['sha256'] not in found:
                raise ValueError('CSV image missing: ' + row['image_path'])
            self.rows.append(dict(row, image_path=found[row['sha256']]))
        for digest, rel in found.items():
            if digest not in by_hash:
                self.rows.append(dict(zip(FIELDS, [rel, '', digest, 'pending',
                                                  'central foreground runner', 'development', ''])))
        if self.rows != old and old:
            self.persist(self.rows)

    def safe_path(self, path):
        path = Path(path)
        if not path.is_absolute():
            path = self.root / path
        if not path.resolve().is_relative_to(self.root):
            raise ValueError('Path escapes repository root')
        # Reject image/destination symlinks even when they point inside the root.
        for part in (path, *path.parents):
            if part == self.root:
                break
            if part.is_symlink():
                raise ValueError('Symbolic links are not supported: ' + str(part))
        return path

    def persist(self, rows):
        import io
        output = io.StringIO(newline='')
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        atomic_write(self.safe_path(self.csv_path), output.getvalue())

    def move(self, source, destination, digest):
        source, destination = self.safe_path(source), self.safe_path(destination)
        if sha256_file(source) != digest:
            raise ValueError('Image changed; annotation refused')
        # link/unlink provides a no-clobber move on the repository filesystem.
        os.link(source, destination)
        sync_directory(destination.parent)
        source.unlink()
        sync_directory(source.parent)

    def recover(self):
        if not self.journal.exists():
            return
        tx = json.loads(self.journal.read_text())
        source = self.safe_path(self.root / tx['source'])
        destination = self.safe_path(self.root / tx['destination'])
        committed = False
        if self.csv_path.exists():
            with self.csv_path.open(newline='', encoding='utf-8') as f:
                committed = tx['row'] in list(csv.DictReader(f))
        keep, remove = (destination, source) if committed else (source, destination)
        for path in (source, destination):
            if path.exists() and sha256_file(path) != tx['row']['sha256']:
                raise ValueError('Transaction image changed; manual recovery required')
        if not keep.exists():
            if not remove.exists():
                raise ValueError('Transaction image missing; manual recovery required')
            self.move(remove, keep, tx['row']['sha256'])
        elif remove.exists():
            if not os.path.samefile(keep, remove):
                raise ValueError('Transaction collision; manual recovery required')
            remove.unlink()
            sync_directory(remove.parent)
        self.journal.unlink()
        sync_directory(self.journal.parent)
        return committed

    def save(self, index, values):
        row = dict(self.rows[index])
        for k in ('bib', 'status', 'target_identity', 'split', 'provenance'):
            if not isinstance(values.get(k), str):
                raise ValueError('Missing text field: ' + k)
            row[k] = values[k].strip()
        if row['status'] not in {'pending', 'reviewed', 'skipped', 'unreadable', 'ambiguous'}:
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
        source = self.safe_path(self.root / row['image_path'])
        if sha256_file(source) != row['sha256']:
            raise ValueError('Image changed; annotation refused')
        directory = ('annotated' if row['status'] == 'reviewed' else
                     'unannotated' if row['status'] == 'pending' else 'needs_review')
        destination = self.safe_path(self.root / 'media' / directory / source.name)
        rows = list(self.rows)
        row['image_path'] = destination.relative_to(self.root).as_posix()
        rows[index] = row
        if source != destination and destination.exists():
            raise ValueError('Destination filename collision: ' + destination.name)
        if self.journal.exists():
            raise ValueError('Unfinished transaction; restart to recover')
        if source == destination:
            self.persist(rows)
        else:
            tx = dict(source=source.relative_to(self.root).as_posix(),
                      destination=row['image_path'], row=row)
            atomic_write(self.journal, json.dumps(tx))
            try:
                self.move(source, destination, row['sha256'])
                self.persist(rows)
            except Exception:
                if self.recover():
                    self.rows = rows
                raise
            self.rows = rows
            self.journal.unlink()
            sync_directory(self.journal.parent)
        self.rows = rows

    def export(self, destination, expected):
        reviewed = [r for r in self.rows if r['status'] == 'reviewed']
        if not reviewed:
            raise ValueError('No reviewed images available for export')
        records = [dict(id='img-' + r['sha256'],
                        path=r['image_path'], bib=r['bib'], sha256=r['sha256'],
                        split=r['split'], review_status='reviewed', target_identity=r['target_identity'],
                        provenance=r['provenance']) for r in reviewed]
        manifest = {'version': 1, 'prompt': PROMPT, 'records': records}
        errors, warnings = validate(manifest, self.root, expected)
        if errors:
            raise ValueError('\n'.join(errors))
        destination = self.safe_path(destination)
        if destination in (self.csv_path, self.journal) or destination in [self.root / r["image_path"] for r in self.rows]:
            raise ValueError("Export must not overwrite collection files")
        atomic_write(destination, json.dumps(manifest, indent=2) + '\n')
        return warnings


HTML = '''<!doctype html><meta charset="utf-8"><title>Bib annotation</title>
<style>body{font:16px system-ui;max-width:1100px;margin:20px auto;padding:10px}img{max-width:100%;max-height:65vh}input,select,button{font:inherit;margin:5px;padding:6px}#message{color:#a20}label{display:block}</style>
<h1>Bib annotation</h1><p>Label one designated runner. Never use predictions to choose the ground truth. Save before navigating.</p>
<label>View <select id="view"><option value="pending">Annotation queue</option><option value="all">All images / edit labels</option></select></label><p id="progress"></p><p id="filename"></p><img id="photo"><form id="form">
<label>Bib digits <input id="bib" autocomplete="off" inputmode="numeric"></label>
<label>Status <select id="status"><option>reviewed</option><option>skipped</option><option>unreadable</option><option>ambiguous</option></select></label>
<label>Target runner <input id="target_identity" size="55"></label>
<label>Split <select id="split"><option>development</option><option>evaluation</option></select></label>
<label>Source / permission note <input id="provenance" size="65"></label>
<button type="button" id="prev">Previous</button><button>Save and next</button><button type="button" id="next">Next without saving</button></form><p id="message"></p>
<script>
let rows=[],i=0,token='TOKEN';const keys=['bib','status','target_identity','split','provenance'];const el=id=>document.getElementById(id);
function queue(){return rows.map((r,j)=>(el('view').value==='all'||r.status==='pending')?j:-1).filter(j=>j>=0)}
function show(){const q=queue(),r=rows[i];el('progress').textContent=`${q.length} pending — ${rows.filter(x=>x.status==='reviewed').length} reviewed — ${rows.filter(x=>['skipped','unreadable','ambiguous'].includes(x.status)).length} needs review`;
el('form').hidden=!q.length;el('photo').hidden=!q.length;el('filename').textContent=q.length?r.image_path:'All available images have been annotated.';if(!q.length)return;
el('photo').src='/image?i='+i+'&v='+Date.now();keys.forEach(k=>el(k).value=(k==='status'&&r[k]==='pending')?'reviewed':r[k]);el('bib').focus();}
el('view').onchange=()=>{i=queue()[0]??0;show()};
fetch('/rows').then(r=>r.json()).then(r=>{rows=r;i=queue()[0]??0;show()}).catch(e=>el('message').textContent=e.message);
el('prev').onclick=()=>{let q=queue();i=q[Math.max(0,q.indexOf(i)-1)];show()};el('next').onclick=()=>{let q=queue();i=q[Math.min(q.length-1,q.indexOf(i)+1)];show()};
el('form').onsubmit=async e=>{e.preventDefault();const button=e.submitter;button.disabled=true;let values={};keys.forEach(k=>values[k]=el(k).value);try{let r=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({token,index:i,values})});let data=await r.json();if(!r.ok)throw Error(data.error);rows[i]=data;el('message').textContent='Saved';i=queue().find(j=>j>i)??queue()[0]??0;show()}catch(e){el('message').textContent=e.message}finally{button.disabled=false}};
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
                    p = collection.safe_path(collection.root / collection.rows[i]['image_path'])
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
    parser.add_argument('--images', default='media/unannotated')
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
