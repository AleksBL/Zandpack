import ast, re

CP = '/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine/checkpoint_2026-10-08_lorentzfit/refine_sampling.py'
AI = '/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine/AI_fit.py'
src_cp, src_ai = open(CP).read(), open(AI).read()
tree_cp, tree_ai = ast.parse(src_cp), ast.parse(src_ai)
fc = {n.name: n for n in tree_cp.body if isinstance(n, ast.FunctionDef)}
fa = {n.name: n for n in tree_ai.body if isinstance(n, ast.FunctionDef)}

def seg(src, node):
    lines = src.splitlines()
    return '\n'.join(lines[node.lineno-1: node.end_lineno])

PREFIXES = ['_lf.varproj.', '_lf.validate.', '_lf.data.', '_lf.sampling.', '_lf.',
            '_vp.', '_prune.', '_con.', '_val.', '_data.', '_samp.']

def norm(text):
    text = re.sub(r'#.*', '', text)
    for p in PREFIXES:
        text = text.replace(p, '')
    return re.sub(r'\s+', ' ', text)

for name in ['fit_boosted', '_fit_one']:
    a, b = norm(seg(src_cp, fc[name])), norm(seg(src_ai, fa[name]))
    print(f'===== {name} ORIGINAL (checkpoint):\n{a}\n')
    print(f'===== {name} AI_fit:\n{b}\n')
