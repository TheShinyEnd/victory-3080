"""Writes tools/quality-catalog.json from crates/victory-core/src/quality.rs (the same shape as
`victory quality --list --json`), for when the settings window has no built victory.exe to ask.
Run from the repository root:  python tools/gen-quality-catalog.py"""
import json, re

src = open('crates/victory-core/src/quality.rs', encoding='utf-8').read()
body = src[src.index('pub static OPTIONS'):src.index('pub fn find')]


def args_of(text, start):
    """Top-level comma separated arguments of the call whose '(' is at text[start]."""
    depth, cur, out, i, instr = 0, '', [], start, False
    while i < len(text):
        c = text[i]
        if instr:
            cur += c
            if c == chr(92):
                i += 1
                cur += text[i]
            elif c == '"':
                instr = False
        elif c == '"':
            instr = True
            cur += c
        elif c in '([{':
            depth += 1
            if depth > 1:
                cur += c
        elif c in ')]}':
            depth -= 1
            if depth == 0:
                out.append(cur.strip())
                return out, i
            cur += c
        elif c == ',' and depth == 1:
            out.append(cur.strip())
            cur = ''
        else:
            cur += c
        i += 1
    raise SystemExit('unbalanced')


options = []
pos = 0
while True:
    m = re.search(r'\bopt!\(', body[pos:])
    if not m:
        break
    at = pos + m.end() - 1
    a, end = args_of(body, at)
    pre = body[max(0, pos + m.start() - 300):pos + m.start()]
    wrap = pre[pre.rfind('Opt {'):] if 'Opt {' in pre else ''
    key, env, group, kind, presets, help_, cost = a
    k = kind.strip()
    words, mn, mx, kd = [], 0, 0, 'text'
    if k.startswith('ON1') or k.startswith('Kind::Bool'):
        kd = 'bool'
    elif k.startswith('Kind::Num'):
        kd = 'num'
        mm = re.search(r'min:\s*([\d.]+),\s*max:\s*([\d.]+)', k)
        mn, mx = float(mm[1]), float(mm[2])
    elif k.startswith('Kind::Choice'):
        kd = 'choice'
        words = re.findall(r'\("([^"]*)",\s*"[^"]*"\)', k)
    pr = re.findall(r'"([^"]*)"', presets)
    options.append(dict(key=json.loads(key), env=json.loads(env), group=json.loads(group), kind=kd, words=words,
                        min=mn, max=mx, presets=pr, live='live: true' in wrap, needs_hooks='needs_hooks: true' in wrap,
                        help=json.loads(help_), cost=json.loads(cost)))
    pos = end + 1

groups = re.findall(r'"([^"]+)"', src[src.index('pub const GROUPS'):src.index('/// Every option')])
presets = re.findall(r'"([^"]+)"', src[src.index('pub const PRESETS'):src.index('/// Whether this build')])
hooks = re.search(r'pub const HOOKS_BUILT: bool = (\w+)', src)[1] == 'true'
out = dict(presets=presets, groups=groups, hooks_built=hooks, options=options)
json.dump(out, open('tools/quality-catalog.json', 'w', encoding='utf-8', newline='\n'), indent=1)
print(len(options), 'options')
