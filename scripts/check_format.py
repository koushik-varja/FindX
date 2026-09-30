from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXT={'.py','.ts','.tsx','.mjs','.md','.yml','.yaml','.json','.toml','.ini','.css','.html'}
issues=[]
for p in ROOT.rglob('*'):
    if not p.is_file() or p.suffix.lower() not in EXT: continue
    if any(part in {'.git','node_modules','.venv','venv'} for part in p.parts): continue
    try: text=p.read_text(encoding='utf-8')
    except UnicodeDecodeError: continue
    if '\t' in text: issues.append(f'{p.relative_to(ROOT)}: tab character')
    for i,line in enumerate(text.splitlines(),1):
        if line.rstrip()!=line: issues.append(f'{p.relative_to(ROOT)}:{i}: trailing whitespace')
    if text and not text.endswith('\n'): issues.append(f'{p.relative_to(ROOT)}: missing final newline')
if issues:
    print('\n'.join(issues)); raise SystemExit(1)
print('format hygiene ok')
