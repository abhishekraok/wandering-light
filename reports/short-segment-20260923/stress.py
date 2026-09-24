"""Out-of-sampler stress: longer lists and bigger values. An input counts only if
the fused witness matches the reference on it, so the test is fair to the behaviour."""
import json, random, sys
sys.path.insert(0, '.')
from experiments.staged_witnesses.check import BY_NAME
from wandering_light.stack_pbe import check

def scale(a, k):
    if isinstance(a, bool): return a
    if isinstance(a, int): return a * k
    if isinstance(a, (list, tuple)): return type(a)(scale(x, k) for x in a)
    return a

def longer(args, extra):
    out = []
    for a, *es in zip(args, *extra):
        out.append(a + sum((e for e in es), type(a)()) if isinstance(a, list) else a)
    return tuple(out)

S = sys.argv[1]
for line in open(S):
    r = json.loads(line)
    b = BY_NAME[r['behaviour']]; prog = tuple(r['program'].split())
    rng = random.Random(7)
    tried = fails = 0; example = None
    for i in range(300):
        args = b.sample(rng)
        if i % 3 == 0: s = longer(args, [b.sample(rng), b.sample(rng)])
        elif i % 3 == 1: s = scale(args, 13)
        else: s = scale(longer(args, [b.sample(rng)]), 7)
        try:
            if check(b, s) is not None: continue
        except Exception: continue
        tried += 1
        try: bad = check(b, s, prog)
        except Exception as e: bad = repr(e)
        if bad is not None:
            fails += 1; example = example or str(bad)[:200]
    print(json.dumps({'behaviour': r['behaviour'], 'tried': tried, 'fails': fails, 'example': example}), flush=True)
