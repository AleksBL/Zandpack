import ast, sys, re

CP = '/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine/checkpoint_2026-10-08_lorentzfit/refine_sampling.py'
AI = '/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine/AI_fit.py'

src_cp = open(CP).read()
src_ai = open(AI).read()
tree_cp = ast.parse(src_cp)
tree_ai = ast.parse(src_ai)

funcs_cp = {n.name: n for n in tree_cp.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
funcs_ai = {n.name: n for n in tree_ai.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}

PREFIXES = ['_lf.varproj.', '_lf.validate.', '_lf.prunes.', '_lf.constraints.',
            '_lf.data.', '_lf.sampling.', '_lf.', '_vp.', '_prune.', '_con.',
            '_val.', '_data.', '_samp.']

def seg(src, node):
    lines = src.splitlines()
    return '\n'.join(lines[node.lineno-1: node.end_lineno])

def normalize(text):
    # strip comments and normalize whitespace for token-ish comparison
    text = re.sub(r'#.*', '', text)
    for p in PREFIXES:
        text = text.replace(p, '')
    text = re.sub(r'\s+', ' ', text)
    # drop import lines and the _import_lorentz_fit machinery
    out = [t for t in text.split('\n')]
    return ' '.join(t for t in out if 'import' not in t)

# in class bodies, methods also need compare for CombinedFit
report = []
for name, node_cp in funcs_cp.items():
    if name not in funcs_ai:
        report.append((name, 'MISSING in AI_fit'))
        continue
    a, b = normalize(seg(src_cp, node_cp)), normalize(seg(src_ai, funcs_ai[name]))
    if a != b:
        report.append((name, 'DIFFERS'))

for name, st in report:
    print(f'{name}: {st}')
if not report:
    print('all shared defs identical after normalization')
