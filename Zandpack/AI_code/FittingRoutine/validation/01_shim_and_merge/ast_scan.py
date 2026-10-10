import ast, builtins, sys

path = '/home/aleks/Skrivebord/ApertureTestCenter/FittingRoutine/AI_fit.py'
src = open(path).read()
tree = ast.parse(src)

defined = set(dir(builtins))
for node in ast.walk(tree):
    if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
        defined.add(node.id)
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        defined.add(node.name)
        if isinstance(node, ast.FunctionDef):
            a = node.args
            for arg in a.posonlyargs + a.args + a.kwonlyargs:
                defined.add(arg.arg)
            if a.vararg: defined.add(a.vararg.arg)
            if a.kwarg: defined.add(a.kwarg.arg)
        for kw in getattr(node, 'args', None) and [] or []:
            pass
    elif isinstance(node, ast.arg):
        defined.add(node.arg)
    elif isinstance(node, ast.Import):
        for al in node.names:
            defined.add(al.asname or al.name.split('.')[0])
    elif isinstance(node, ast.ImportFrom):
        for al in node.names:
            defined.add(al.asname or al.name)
    elif isinstance(node, ast.ExceptHandler):
        if node.name:
            defined.add(node.name)
    elif isinstance(node, (ast.Global, ast.Nonlocal)):
        for n in node.names:
            defined.add(n)
    # comprehension targets
    elif isinstance(node, ast.comprehension):
        for tgt in ast.walk(node.target):
            if isinstance(tgt, ast.Name):
                defined.add(tgt.id)

undefined = []
for node in ast.walk(tree):
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
        if node.id not in defined:
            undefined.append((node.id, node.lineno))

# function-local check is coarse (global store counts); also report
# per-function loads of names that are not module-level anywhere
print("loads of names never defined anywhere (global scan):")
seen = {}
for name, line in undefined:
    seen.setdefault(name, []).append(line)
if not seen:
    print("  none")
for name, lines in sorted(seen.items()):
    print(f"  {name}: lines {lines[:6]}")
sys.exit(0)
