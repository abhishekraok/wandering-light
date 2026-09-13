# ruff: noqa: RUF001
"""Build the interactive comparison page from `page_data.json`.

    uv run python reports/stack-pbe-v4-run-comparison-20260913/build_page.py

Writes `index.html` (standalone, open it in a browser) beside this file and,
with `--artifact PATH`, the same page without the document skeleton for the
claude.ai Artifact publisher. Reads `paired.json(.gz)` from
`experiments/stack_pbe_compare_runs.py`, keeps what the page needs (examples,
per-op traces, verdicts) and adds per-run counts; the five featured cases and
their commentary live in the template below.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent

HEAD = r"""<title>Luna and Astra Part Ways</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&family=JetBrains+Mono:wght@400;600&display=swap">
<style>
:root{
  color-scheme:light;
  --ground:#ECEFF3; --surface:#FFFFFF; --surface-2:#F5F7F9; --ink:#17202A; --ink-2:#4E5A66; --ink-3:#7B8794;
  --line:#D5DBE2; --line-2:#E6EAEF; --accent:#2F4A6D; --accent-ink:#FFFFFF; --focus:#2F4A6D;
  --luna:#5A55C9; --luna-soft:#ECEBF8; --astra:#0E9484; --astra-soft:#E1F3F0;
  --solved:#1F8A4C; --memorised:#C98500; --failed:#D64545; --illegal:#8B95A1;
  --solved-soft:#DDF3E5; --memorised-soft:#FBEFCF; --failed-soft:#FADADA; --illegal-soft:#E6E9ED;
  --mark:#B4467C; --mark-soft:#F8E4EE; --sym:#6B7785; --sym-soft:#EEF1F4;
  --tile:#FFFFFF; --tile-line:#C9D1DA; --shadow:0 1px 2px rgba(23,32,42,.06),0 10px 28px rgba(23,32,42,.07);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    color-scheme:dark;
    --ground:#141820; --surface:#1C222C; --surface-2:#222A35; --ink:#E8ECF1; --ink-2:#AEB8C4; --ink-3:#8290A0;
    --line:#2E3845; --line-2:#28313D; --accent:#9DB4D8; --accent-ink:#141820; --focus:#9DB4D8;
    --luna:#8580E8; --luna-soft:#282747; --astra:#25A898; --astra-soft:#173430;
    --solved:#227D48; --memorised:#B8891A; --failed:#E4626A; --illegal:#8592A0;
    --solved-soft:#1C3A28; --memorised-soft:#3B3115; --failed-soft:#43242A; --illegal-soft:#2A323C;
    --mark:#E07AB0; --mark-soft:#3E2534; --sym:#98A4B2; --sym-soft:#28313C;
    --tile:#232B36; --tile-line:#3A4552; --shadow:none;
  }
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --ground:#141820; --surface:#1C222C; --surface-2:#222A35; --ink:#E8ECF1; --ink-2:#AEB8C4; --ink-3:#8290A0;
  --line:#2E3845; --line-2:#28313D; --accent:#9DB4D8; --accent-ink:#141820; --focus:#9DB4D8;
  --luna:#8580E8; --luna-soft:#282747; --astra:#25A898; --astra-soft:#173430;
  --solved:#227D48; --memorised:#B8891A; --failed:#E4626A; --illegal:#8592A0;
  --solved-soft:#1C3A28; --memorised-soft:#3B3115; --failed-soft:#43242A; --illegal-soft:#2A323C;
  --mark:#E07AB0; --mark-soft:#3E2534; --sym:#98A4B2; --sym-soft:#28313C;
  --tile:#232B36; --tile-line:#3A4552; --shadow:none;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--ground);color:var(--ink);font-family:"Source Serif 4",Georgia,"Times New Roman",serif;font-size:17px;line-height:1.55;font-variant-numeric:tabular-nums}
.wrap{max-width:1120px;margin:0 auto;padding:0 24px}
h1,h2,h3,.display{font-family:"Bricolage Grotesque","Helvetica Neue",Arial,sans-serif;text-wrap:balance;letter-spacing:-.01em}
h1{font-size:clamp(34px,5vw,50px);font-weight:800;line-height:1.02;margin:0 0 14px}
h2{font-size:28px;font-weight:700;line-height:1.1;margin:0 0 8px}
h3{font-size:19px;font-weight:700;margin:0 0 6px}
p{margin:0 0 12px;max-width:68ch}
a{color:var(--accent)}
code,.mono{font-family:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.88em}
p code,li code,td code{background:var(--surface-2);border:1px solid var(--line-2);border-radius:4px;padding:0 4px;white-space:nowrap}
.eyebrow{font-family:"Bricolage Grotesque",sans-serif;font-size:12px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--ink-3);margin:0 0 8px}
.muted{color:var(--ink-2)}
.small{font-size:14px}
:focus-visible{outline:2px solid var(--focus);outline-offset:2px}
button{font:inherit;color:inherit}
section{padding:40px 0 8px}
.masthead{padding:56px 0 24px}
.masthead p.lede{font-size:20px;line-height:1.45;max-width:60ch}
.runline{display:flex;gap:10px;flex-wrap:wrap;margin-top:18px;align-items:center}
.runchip{display:inline-flex;align-items:center;gap:8px;padding:6px 12px 6px 8px;border-radius:999px;font-family:"Bricolage Grotesque",sans-serif;font-weight:700;font-size:14px;background:var(--surface);border:1px solid var(--line)}
.runchip i{width:10px;height:10px;border-radius:50%;display:inline-block}
.runchip.luna i{background:var(--luna)} .runchip.astra i{background:var(--astra)}
.runchip span{font-weight:500;color:var(--ink-2)}
/* scoreboard */
.plates{display:grid;grid-template-columns:1fr 1fr;gap:18px}
@media (max-width:820px){.plates{grid-template-columns:1fr}}
.plate{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:18px 20px 16px;box-shadow:var(--shadow);border-top:4px solid var(--line)}
.plate.luna{border-top-color:var(--luna)} .plate.astra{border-top-color:var(--astra)}
.plate h3{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}
.plate h3 .mono{font-weight:600;font-size:15px;color:var(--ink-2)}
.plate .route{color:var(--ink-2);font-size:14px;margin:0 0 14px}
.big{display:flex;align-items:baseline;gap:8px;font-family:"Bricolage Grotesque",sans-serif}
.big b{font-size:44px;font-weight:800;line-height:1;letter-spacing:-.02em}
.big span{color:var(--ink-2);font-size:14px}
.vbar{display:flex;gap:2px;height:16px;margin:14px 0 8px;border-radius:4px;overflow:hidden;background:var(--surface-2)}
.vbar div{height:100%}
.vbar .s{background:var(--solved)} .vbar .m{background:var(--memorised)} .vbar .w{background:var(--failed)} .vbar .x{background:var(--illegal)}
.keys{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:13px;color:var(--ink-2);font-family:"Bricolage Grotesque",sans-serif;font-weight:500}
.keys i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:6px;vertical-align:baseline}
.keys .s i{background:var(--solved)} .keys .m i{background:var(--memorised)} .keys .w i{background:var(--failed)} .keys .x i{background:var(--illegal)}
.overst{margin-top:12px;font-size:14px;color:var(--ink-2);border-top:1px dashed var(--line);padding-top:10px}
.overst b{color:var(--ink)}
.cont{margin-top:22px;display:grid;grid-template-columns:auto 1fr;gap:18px 26px;align-items:start}
@media (max-width:820px){.cont{grid-template-columns:1fr}}
table.ct{border-collapse:collapse;font-family:"Bricolage Grotesque",sans-serif;font-size:14px}
table.ct th,table.ct td{padding:6px 12px;text-align:right;border-bottom:1px solid var(--line-2)}
table.ct th{font-weight:700;color:var(--ink-2);text-align:right}
table.ct th.rowh{text-align:left}
table.ct td.hot{background:var(--surface);font-weight:700}
table.ct td.zero{color:var(--ink-3)}
table.ct caption{caption-side:top;text-align:left;font-weight:700;padding:0 0 6px;color:var(--ink-2);font-size:13px;letter-spacing:.06em;text-transform:uppercase}
/* strip */
.strip{overflow-x:auto;padding:6px 0 4px}
.grid{display:grid;grid-template-columns:64px repeat(40,26px);gap:3px 2px;align-items:center;width:max-content}
.grid .lab{font-family:"Bricolage Grotesque",sans-serif;font-size:12px;font-weight:700;color:var(--ink-2);padding-right:8px;text-align:right}
.grid .badge{height:20px;display:flex;align-items:flex-end;justify-content:center;font-family:"Bricolage Grotesque",sans-serif;font-size:12px;font-weight:800;color:var(--accent)}
.grid .badge b{width:18px;height:18px;border-radius:50%;background:var(--accent);color:var(--accent-ink);display:inline-flex;align-items:center;justify-content:center;font-size:11px}
.cell{width:24px;height:24px;border-radius:5px;border:0;padding:0;cursor:pointer;display:block;position:relative;background:var(--illegal-soft)}
.cell.solved{background:var(--solved)} .cell.memorised{background:var(--memorised)} .cell.failed.illegal{background:var(--illegal)} .cell.failed.wrong{background:var(--failed)}
.cell::after{content:attr(data-g);position:absolute;inset:0;border-radius:5px;box-shadow:inset 0 0 0 1.5px rgba(0,0,0,.08);display:flex;align-items:center;justify-content:center;font-family:"Bricolage Grotesque",sans-serif;font-size:12px;font-weight:800;color:#fff;line-height:1}
.cell.failed.illegal::after{color:#fff}
.cell.same{opacity:.45}
.cell:hover{transform:translateY(-1px)}
.grid .len{font-family:"JetBrains Mono",monospace;font-size:10px;color:var(--ink-3);text-align:center;padding-top:2px}
.stripcap{min-height:24px;font-size:14px;color:var(--ink-2);margin-top:8px;font-family:"Bricolage Grotesque",sans-serif}
.stripcap b{color:var(--ink)}
.legend{display:flex;flex-wrap:wrap;gap:8px 18px;font-size:13px;color:var(--ink-2);font-family:"Bricolage Grotesque",sans-serif;font-weight:500;margin-top:8px}
.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}
/* cases */
.case{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:26px 28px 22px;margin:22px 0;box-shadow:var(--shadow)}
.case header{display:grid;grid-template-columns:auto 1fr;gap:6px 18px;align-items:start;margin-bottom:12px}
.casenum{width:44px;height:44px;border-radius:50%;background:var(--accent);color:var(--accent-ink);display:flex;align-items:center;justify-content:center;font-family:"Bricolage Grotesque",sans-serif;font-weight:800;font-size:20px;grid-row:span 2}
.case header .eyebrow{margin:6px 0 2px}
.case h2 code{font-size:.8em;background:var(--surface-2);border:1px solid var(--line-2);padding:1px 7px;border-radius:6px}
.case .summary{font-style:italic;color:var(--ink-2);margin:0 0 6px;font-size:18px}
.pills{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.pill{display:inline-flex;align-items:center;gap:6px;border-radius:999px;padding:3px 10px 3px 8px;font-family:"Bricolage Grotesque",sans-serif;font-size:13px;font-weight:700;border:1px solid transparent}
.pill i{width:8px;height:8px;border-radius:50%;display:inline-block}
.pill.solved{background:var(--solved-soft);color:var(--solved)} .pill.solved i{background:var(--solved)}
.pill.memorised{background:var(--memorised-soft);color:var(--memorised)} .pill.memorised i{background:var(--memorised)}
.pill.failed{background:var(--failed-soft);color:var(--failed)} .pill.failed i{background:var(--failed)}
.pill.illegal{background:var(--illegal-soft);color:var(--illegal)} .pill.illegal i{background:var(--illegal)}
.pill .who{font-weight:500;color:var(--ink-2);margin-right:2px}
.lesson{margin:14px 0 18px;padding:0 0 0 18px;border-left:3px solid var(--line)}
.lesson p{max-width:72ch}
.twocol{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr);gap:22px;align-items:start}
@media (max-width:900px){.twocol{grid-template-columns:1fr}}
.block h4{font-family:"Bricolage Grotesque",sans-serif;font-size:12px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--ink-3);margin:0 0 8px}
table.ex{border-collapse:collapse;width:100%;font-size:13.5px}
table.ex th{font-family:"Bricolage Grotesque",sans-serif;font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3);text-align:left;padding:4px 8px;border-bottom:1px solid var(--line)}
table.ex th.r,table.ex td.r{text-align:center;width:52px}
table.ex td{padding:5px 8px;border-bottom:1px solid var(--line-2);vertical-align:top;font-family:"JetBrains Mono",monospace;white-space:nowrap}
table.ex tr.sec td{font-family:"Bricolage Grotesque",sans-serif;font-size:11px;letter-spacing:.1em;text-transform:uppercase;color:var(--ink-3);padding-top:10px;border-bottom:0;white-space:normal}
table.ex tr.pick{cursor:pointer}
table.ex tr.pick:hover td{background:var(--surface-2)}
table.ex tr.cur td{background:var(--surface-2)}
table.ex tr.cur td:first-child{box-shadow:inset 3px 0 0 var(--accent)}
table.ex td.arrow{color:var(--ink-3);width:22px;text-align:center}
.exwrap{overflow-x:auto}
.mark{display:inline-flex;align-items:center;justify-content:center;width:22px;height:22px;border-radius:50%;font-family:"Bricolage Grotesque",sans-serif;font-weight:800;font-size:13px}
.mark.pass{background:var(--solved-soft);color:var(--solved)} .mark.wrong{background:var(--failed-soft);color:var(--failed)} .mark.illegal{background:var(--illegal-soft);color:var(--illegal)} .mark.none{color:var(--ink-3)}
.prog{border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:12px;cursor:pointer;background:var(--surface);transition:border-color .15s}
.prog:hover{border-color:var(--ink-3)}
.prog.on{border-color:var(--accent);box-shadow:0 0 0 2px color-mix(in srgb,var(--accent) 22%,transparent)}
.prog.luna{border-left:4px solid var(--luna)} .prog.astra{border-left:4px solid var(--astra)}
.prog .top{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:8px}
.prog .who{font-family:"Bricolage Grotesque",sans-serif;font-weight:800;font-size:15px}
.prog .who small{font-weight:500;color:var(--ink-2);font-size:13px;margin-left:8px}
.tape{display:flex;flex-wrap:wrap;gap:4px;font-family:"JetBrains Mono",monospace;font-size:13px;line-height:1}
.tok{padding:4px 6px;border-radius:5px;background:var(--surface-2);border:1px solid var(--line-2);color:var(--ink)}
.tok.br{background:var(--mark-soft);color:var(--mark);border-color:transparent;font-weight:600}
.tok.done{opacity:.55}
.tok.cur{background:var(--accent);color:var(--accent-ink);border-color:var(--accent);opacity:1}
.tok.ref{background:var(--failed);color:#fff;border-color:var(--failed);opacity:1}
.prog .note{font-size:13px;color:var(--ink-2);margin-top:8px}
details.wit{margin-top:6px;font-size:13px;color:var(--ink-2)}
details.wit summary{cursor:pointer;font-family:"Bricolage Grotesque",sans-serif;font-weight:700;font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--ink-3)}
details.wit .tape{margin-top:8px}
/* tracer */
.tracer{margin-top:20px;border:1px solid var(--line);border-radius:12px;background:var(--surface-2);padding:16px 18px 18px}
.tracer:focus-visible{outline:2px solid var(--focus)}
.tbar{display:flex;flex-wrap:wrap;gap:10px 16px;align-items:center;margin-bottom:12px}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:var(--surface)}
.seg button{border:0;background:transparent;padding:6px 12px;font-family:"Bricolage Grotesque",sans-serif;font-weight:700;font-size:13px;cursor:pointer;color:var(--ink-2)}
.seg button.on{background:var(--ink);color:var(--surface)}
.seg button.luna.on{background:var(--luna);color:#fff} .seg button.astra.on{background:var(--astra);color:#fff}
.tbar select{font:inherit;font-family:"JetBrains Mono",monospace;font-size:13px;padding:6px 8px;border:1px solid var(--line);border-radius:8px;background:var(--surface);color:var(--ink);max-width:100%}
.steps{display:flex;align-items:center;gap:6px;flex:1 1 320px;min-width:260px}
.steps button{border:1px solid var(--line);background:var(--surface);border-radius:8px;width:34px;height:32px;cursor:pointer;font-size:14px;font-family:"Bricolage Grotesque",sans-serif;font-weight:700}
.steps button:hover{border-color:var(--ink-3)}
.steps input[type=range]{flex:1;accent-color:var(--accent);min-width:120px}
.readout{font-family:"Bricolage Grotesque",sans-serif;font-size:13px;color:var(--ink-2);min-width:150px}
.readout b{color:var(--ink)}
.tbody{display:grid;grid-template-columns:minmax(0,1fr) minmax(220px,300px);gap:18px;align-items:start}
@media (max-width:760px){.tbody{grid-template-columns:1fr}}
.tapebox{background:var(--surface);border:1px solid var(--line-2);border-radius:10px;padding:12px 14px}
.tapebox .lbl,.stackbox .lbl{font-family:"Bricolage Grotesque",sans-serif;font-size:11px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--ink-3);margin-bottom:8px;display:flex;justify-content:space-between;gap:10px}
.status{margin-top:12px;padding:10px 12px;border-radius:8px;font-size:14px;font-family:"Bricolage Grotesque",sans-serif;font-weight:500;border:1px solid var(--line-2);background:var(--surface)}
.status.pass{border-color:var(--solved);color:var(--solved);background:var(--solved-soft)}
.status.wrong{border-color:var(--failed);color:var(--failed);background:var(--failed-soft)}
.status.illegal{border-color:var(--failed);background:var(--failed-soft);color:var(--ink)}
.status.illegal b{color:var(--failed)}
.status.run{color:var(--ink-2)}
.status code{background:transparent;border:0;padding:0}
.stackbox{background:var(--surface);border:1px solid var(--line-2);border-radius:10px;padding:12px 14px;min-height:180px}
.stack{display:flex;flex-direction:column-reverse;gap:5px;align-items:stretch;padding-bottom:8px;border-bottom:3px solid var(--ink-3);min-height:96px;justify-content:flex-start}
.tile{position:relative;border:1px solid var(--tile-line);background:var(--tile);border-radius:7px;padding:7px 10px;font-family:"JetBrains Mono",monospace;font-size:13px;line-height:1.35;word-break:break-word;box-shadow:0 1px 0 var(--line)}
.tile.int{border-left:4px solid var(--accent)}
.tile.list{border-left:4px solid var(--astra)}
.tile.bool{border-left:4px solid var(--memorised)}
.tile.mark{border:1px dashed var(--mark);color:var(--mark);background:var(--mark-soft);font-weight:600;text-align:center}
.tile.sym{border-left:4px solid var(--sym);color:var(--sym);font-style:italic;background:var(--sym-soft)}
.tile.top::after{content:"top";position:absolute;right:8px;top:6px;font-family:"Bricolage Grotesque",sans-serif;font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:var(--ink-3)}
.tile.new{animation:drop .22s ease-out}
@keyframes drop{from{transform:translateY(-10px);opacity:.2}to{transform:none;opacity:1}}
.stack .empty{color:var(--ink-3);font-family:"Bricolage Grotesque",sans-serif;font-size:13px;text-align:center;padding:30px 0 20px}
.capnote{font-size:12px;color:var(--ink-3);margin-top:8px;font-family:"Bricolage Grotesque",sans-serif}
.kbd{font-family:"JetBrains Mono",monospace;font-size:11px;border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:0 5px;background:var(--surface)}
@media (prefers-reduced-motion: reduce){.tile.new{animation:none}.cell:hover{transform:none}}
/* essays */
.essay{max-width:72ch}
.essay h3{margin-top:22px}
.essay ul,.essay ol{padding-left:22px;margin:0 0 14px}
.essay li{margin-bottom:8px}
.aside{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px 20px;margin:16px 0;max-width:72ch}
.aside .eyebrow{margin-bottom:4px}
/* browse */
.browsebar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:6px}
.browsebar select{font:inherit;font-family:"JetBrains Mono",monospace;font-size:13px;padding:8px 10px;border:1px solid var(--line);border-radius:8px;background:var(--surface);color:var(--ink);max-width:100%}
/* footer */
footer{padding:36px 0 60px;border-top:1px solid var(--line);margin-top:30px;font-size:14px;color:var(--ink-2)}
footer h3{font-size:16px}
footer table{border-collapse:collapse;font-family:"Bricolage Grotesque",sans-serif;font-size:13.5px}
footer td,footer th{padding:5px 12px 5px 0;text-align:left;vertical-align:top;border-bottom:1px solid var(--line-2)}
footer th{color:var(--ink-3);font-weight:700;letter-spacing:.06em;text-transform:uppercase;font-size:11px}
footer .ok{color:var(--solved);font-weight:700} footer .miss{color:var(--memorised);font-weight:700}
</style>"""

BODY = r"""<script id="data" type="application/json">__DATA__</script>
<div class="wrap">
<header class="masthead">
  <p class="eyebrow">stack_pbe_v4 · eval sample · seed 17 · forty tasks · two saved runs, regraded</p>
  <h1>Luna and Astra Part Ways</h1>
  <p class="lede">Two frontier models wrote one program each for the same forty stack-machine tasks. Their verdicts differ on twenty. This page steps through five of them, op by op, so the stack itself shows where each solver went wrong.</p>
  <div class="runline">
    <span class="runchip luna"><i></i>luna <span>gpt-5.6-luna · OpenAI API · effort high · 2026-09-01</span></span>
    <span class="runchip astra"><i></i>astra <span>gpt-6-astra · codex exec · effort high · 2026-09-10</span></span>
  </div>
</header>

<section id="scoreboard">
  <p class="eyebrow">Scoreboard</p>
  <h2>Same tasks, same grader, one candidate per task</h2>
  <p class="muted">A program is <b>solved</b> when it reproduces every shown pair <em>and</em> every held-out pair; <b>memorised</b> when it fits the shown pairs and fails the hidden ones; <b>failed</b> otherwise, which splits into programs that run and compute the wrong thing and programs the machine refuses to run.</p>
  <div class="plates" id="plates"></div>
  <div class="cont">
    <table class="ct" id="contingency"><caption>Paired verdicts</caption></table>
    <div class="small muted" id="contnote"></div>
  </div>
</section>

<section id="stripsec">
  <p class="eyebrow">All forty, by witness length</p>
  <h2>Where they part ways</h2>
  <p class="muted">One column per task, sorted by the length of the corpus witness (the shortest known program). Numbered columns are the five cases below. Faded columns are tasks where both runs got the same verdict. Click any cell to load that task into the explorer at the bottom.</p>
  <div class="strip"><div class="grid" id="grid"></div></div>
  <div class="stripcap" id="stripcap">Hover a column for its behaviour.</div>
  <div class="legend"><span><i style="background:var(--solved)"></i>● solved</span><span><i style="background:var(--memorised)"></i>◐ memorised</span><span><i style="background:var(--failed)"></i>✗ failed: runs, wrong output</span><span><i style="background:var(--illegal)"></i>‼ failed: machine refused an op</span></div>
</section>

<section id="cases">
  <p class="eyebrow">Five cases</p>
  <h2>Step through the programs</h2>
  <p class="muted">Each case shows the task as the solver saw it (the shown pairs), the pairs it was graded on (held out), both candidates, and a tracer. In the tracer, <span class="kbd">←</span> <span class="kbd">→</span> step one op at a time; click a program or an example row to trace it. Tiles are stack values, bottom to top; a dashed tile is an open <code>[</code> mark and the italic tiles above it are ops being captured into a quotation.</p>
  <div id="caselist"></div>
</section>

<section id="modes" class="essay">
  <p class="eyebrow">What the twenty say</p>
  <h2>Two solvers, two ways to be wrong</h2>
  <div id="modestext"></div>
</section>

<section id="capture" class="essay">
  <p class="eyebrow">For the next run</p>
  <h2>What future runs should capture</h2>
  <div id="capturetext"></div>
</section>

<section id="browse">
  <p class="eyebrow">Explorer</p>
  <h2>Browse all forty</h2>
  <div class="browsebar"><label for="pick" class="small muted">Task</label><select id="pick"></select></div>
  <div id="browsecase"></div>
</section>

<footer id="provenance">
  <h3>Provenance and alignment</h3>
  <div id="provtext"></div>
</footer>
</div>

<script>
(function(){
const DATA = JSON.parse(document.getElementById('data').textContent);
const TASKS = DATA.tasks;
const byPos = Object.fromEntries(TASKS.map(t => [t.position, t]));
const RUNS = ['luna','astra'];
const RUN = {
  luna: {name:'gpt-5.6-luna', route:'OpenAI Responses API · effort high · run 2026-09-01', file:'reports/stack-pbe-v4-baseline-20260901/luna_v4_sample40.json'},
  astra:{name:'gpt-6-astra', route:'codex exec (CLI 0.153.4) · effort high · run 2026-09-10', file:'reports/stack-pbe-v4-codex-exec-20260910/astra_high_v4_sample40.json'},
};
const GLYPH = {pass:'✓', wrong:'✗', illegal:'‼'};
const VN = {solved:'solved', memorised:'memorised', failed:'failed'};

function h(tag, attrs, ...kids){
  const el = document.createElement(tag);
  const isAttrs = attrs !== null && typeof attrs === 'object' && !attrs.nodeType && !Array.isArray(attrs);
  if (!isAttrs && attrs !== undefined) kids.unshift(attrs);
  if (isAttrs) for (const [k,v] of Object.entries(attrs)){
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'html') el.innerHTML = v;
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat(Infinity)){
    if (kid === null || kid === undefined || kid === false) continue;
    el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return el;
}
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function kindOf(s){
  if (s === 'MARK') return 'mark';
  if (s === 'true' || s === 'false') return 'bool';
  if (/^-?\d+$/.test(s)) return 'int';
  if (s === 'nil' || s[0] === '(') return 'list';
  return 'sym';
}
function examplesOf(t){
  return [...t.shown.map((p,i) => ({kind:'shown', i, input:p[0], expected:p[1]})),
          ...t.held_out.map((p,i) => ({kind:'held', i, input:p[0], expected:p[1]}))];
}
function resultOf(t, run, ex){
  const r = t.runs[run];
  if (!r.parsed) return null;
  const k = t.shown.length;
  return ex < k ? r.shown[ex] : r.held[ex - k];
}
function failKind(r){ return r.v === 'failed' ? (r.legal ? 'wrong' : 'illegal') : r.v; }
function verdictPill(run, r){
  const fk = failKind(r);
  const cls = r.v === 'failed' ? 'failed' + (fk === 'illegal' ? ' illegal' : '') : r.v;
  const label = r.v === 'failed' ? (fk === 'illegal' ? 'failed · refused' : 'failed · wrong output') : r.v;
  return h('span', {class:'pill ' + (fk === 'illegal' ? 'illegal' : r.v)}, h('i'), h('span', {class:'who'}, run), label);
}
function tape(tokens, opts){
  opts = opts || {};
  return h('div', {class:'tape'}, tokens.map((tok, i) => {
    let cls = 'tok';
    if (tok === '[' || tok === ']') cls += ' br';
    if (opts.cur !== undefined && i < opts.cur) cls += ' done';
    if (opts.cur !== undefined && i === opts.cur) cls += ' cur';
    if (opts.ref !== undefined && i === opts.ref) cls += ' ref';
    return h('span', {class:cls}, tok);
  }));
}

/* ---------- scoreboard ---------- */
(function scoreboard(){
  const plates = document.getElementById('plates');
  for (const run of RUNS){
    const s = DATA.meta.stats[run];
    const wrong = s.runs_but_wrong, illegal = s.illegal;
    const pct = n => (100*n/40).toFixed(1) + '%';
    plates.append(h('div', {class:'plate ' + run},
      h('h3', run, h('span', {class:'mono'}, RUN[run].name)),
      h('p', {class:'route'}, RUN[run].route),
      h('div', {class:'big'}, h('b', s.solved + '/40'), h('span', 'solved on held-out data · ' + pct(s.solved))),
      h('div', {class:'vbar', title:'solved / memorised / wrong / refused'},
        h('div', {class:'s', style:'width:' + (100*s.solved/40) + '%'}),
        h('div', {class:'m', style:'width:' + (100*s.memorised/40) + '%'}),
        h('div', {class:'w', style:'width:' + (100*wrong/40) + '%'}),
        h('div', {class:'x', style:'width:' + (100*illegal/40) + '%'})),
      h('div', {class:'keys'},
        h('span', {class:'s'}, h('i'), s.solved + ' solved'),
        h('span', {class:'m'}, h('i'), s.memorised + ' memorised'),
        h('span', {class:'w'}, h('i'), wrong + ' wrong output'),
        h('span', {class:'x'}, h('i'), illegal + ' refused by the machine')),
      h('div', {class:'overst'}, 'A grader with only the shown pairs would have said ', h('b', pct(s.shown_fit)), ' — an overstatement of ', h('b', (100*(s.shown_fit - s.solved)/40).toFixed(1) + ' points'), '.')
    ));
  }
  const c = DATA.meta.contingency;
  const tbl = document.getElementById('contingency');
  const V = ['solved','memorised','failed'];
  tbl.append(h('tr', h('th', {class:'rowh'}, 'luna ↓ · astra →'), V.map(v => h('th', v))));
  for (const p of V){
    tbl.append(h('tr', h('th', {class:'rowh'}, p), V.map(q => {
      const n = c[p + '|' + q];
      return h('td', {class: n === 0 ? 'zero' : (p !== q ? 'hot' : '')}, n);
    })));
  }
  document.getElementById('contnote').innerHTML = 'The twenty disagreements all run one way: on every one of them luna failed, and astra either solved (8) or memorised (12). On 17 of the 20, luna’s program never ran to completion. Where the two agree and are not both solved, they are surprisingly alike: on <code>adjacent_sign_change_count</code> (#9) and <code>axis_point_count</code> (#14) both wrote the <em>identical</em> memorising program, and on <code>aligned_matches</code> (#30) both wrote the identical sampler-shortcut program that the baseline report already flagged as a corpus defect rather than a solve.';
})();

/* ---------- strip ---------- */
const FEATURED = [
  {pos:25, title:'The threshold lives outside the body', run:'luna', ex:0},
  {pos:38, title:'A fold body must leave exactly one value', run:'luna', ex:0},
  {pos:3,  title:'Three examples admit a simpler story', run:'astra', exInput:'[ (3 -3 3 8 -6) ]'},
  {pos:8,  title:'A proxy that is right twelve times in thirteen', run:'astra', exInput:'[ 9 -2 -1 ]'},
  {pos:5,  title:'Both encode the answers; one encodes better', run:'luna', ex:1},
];
const featuredNo = Object.fromEntries(FEATURED.map((f,i) => [f.pos, i+1]));

(function strip(){
  const grid = document.getElementById('grid');
  const cap = document.getElementById('stripcap');
  const order = [...TASKS].sort((a,b) => a.witness_length - b.witness_length || a.position - b.position);
  grid.append(h('div', {class:'lab'}, ''));
  order.forEach(t => grid.append(h('div', {class:'badge'}, featuredNo[t.position] ? h('b', featuredNo[t.position]) : '')));
  for (const run of RUNS){
    grid.append(h('div', {class:'lab'}, run));
    order.forEach(t => {
      const r = t.runs[run];
      const same = t.runs.luna.v === t.runs.astra.v;
      const fk = failKind(r);
      const title = '#' + t.position + ' ' + t.behaviour + ' · witness ' + t.witness_length + ' ops · ' + run + ': ' + (r.v === 'failed' ? 'failed (' + (fk === 'illegal' ? 'refused' : 'wrong output') + ')' : r.v);
      const glyph = r.v === 'solved' ? '●' : r.v === 'memorised' ? '◐' : fk === 'illegal' ? '‼' : '✗';
      grid.append(h('button', {class:'cell ' + r.v + ' ' + fk + (same ? ' same' : ''), title, 'aria-label': title, 'data-g': glyph,
        onmouseenter: () => { cap.innerHTML = '<b>#' + t.position + ' <code>' + esc(t.behaviour) + '</code></b> · witness ' + t.witness_length + ' ops · luna: ' + esc(verdictText(t.runs.luna)) + ' · astra: ' + esc(verdictText(t.runs.astra)); },
        onclick: () => { loadBrowse(t.position); document.getElementById('browse').scrollIntoView({behavior:'smooth', block:'start'}); }
      }));
    });
  }
  grid.append(h('div', {class:'lab'}, 'ops'));
  order.forEach(t => grid.append(h('div', {class:'len'}, t.witness_length)));
})();
function verdictText(r){ return r.v === 'failed' ? 'failed (' + (r.legal ? 'wrong output' : 'refused') + ')' : r.v; }

/* ---------- case component ---------- */
function renderCase(t, opts){
  opts = opts || {};
  const exs = examplesOf(t);
  const state = {run: opts.run || 'luna', ex: 0, step: 0};
  if (opts.exInput){ const i = exs.findIndex(e => e.input === opts.exInput); state.ex = i >= 0 ? i : 0; }
  else if (opts.ex !== undefined) state.ex = opts.ex;
  const art = h('article', {class:'case', id: opts.id || ('task-' + t.position)});
  const head = h('header',
    opts.number ? h('div', {class:'casenum'}, opts.number) : null,
    h('div', h('p', {class:'eyebrow'}, 'Task #' + t.position + ' of the sample · corpus index ' + t.task_index + ' · witness ' + t.witness_length + ' ops · ' + t.tags.join(', ')),
      h('h2', opts.title ? [opts.title, ' ', h('code', t.behaviour)] : h('code', t.behaviour)),
      h('p', {class:'summary'}, t.summary),
      h('div', {class:'pills'}, RUNS.map(run => verdictPill(run, t.runs[run])))));
  art.append(head);
  if (opts.lesson) art.append(h('div', {class:'lesson', html: opts.lesson}));

  // examples table
  const tbl = h('table', {class:'ex'});
  tbl.append(h('tr', h('th', 'input stack'), h('th', ''), h('th', 'expected output'), h('th', {class:'r'}, 'luna'), h('th', {class:'r'}, 'astra')));
  const rows = [];
  exs.forEach((e, idx) => {
    if (e.i === 0) tbl.append(h('tr', {class:'sec'}, h('td', {colspan:5}, e.kind === 'shown' ? 'shown to the solver (' + t.shown.length + ')' : 'held out, used only for grading (' + t.held_out.length + ')')));
    const tr = h('tr', {class:'pick', tabindex:0, onclick: () => { state.ex = idx; state.step = null; render(); }, onkeydown: ev => { if (ev.key === 'Enter' || ev.key === ' '){ ev.preventDefault(); state.ex = idx; state.step = null; render(); } }},
      h('td', e.input), h('td', {class:'arrow'}, '→'), h('td', e.expected),
      RUNS.map(run => {
        const r = resultOf(t, run, idx);
        if (!r) return h('td', {class:'r'}, h('span', {class:'mark none', title:'candidate does not parse'}, '–'));
        const title = r.r === 'pass' ? 'reproduces the expected stack' : r.r === 'wrong' ? 'ran, produced ' + r.g : 'refused at op ' + (r.ra + 1) + ' (' + t.runs[run].c.split(' ')[r.ra] + '): ' + r.why;
        return h('td', {class:'r'}, h('span', {class:'mark ' + r.r, title}, GLYPH[r.r]));
      }));
    rows.push(tr); tbl.append(tr);
  });
  // programs
  const progs = {};
  const progCol = h('div', {class:'block'}, h('h4', 'the two candidates'));
  for (const run of RUNS){
    const r = t.runs[run];
    const toks = r.c ? r.c.split(' ') : [];
    const p = h('div', {class:'prog ' + run, tabindex:0, role:'button', onclick: () => { state.run = run; state.step = null; render(); }, onkeydown: ev => { if (ev.key === 'Enter' || ev.key === ' '){ ev.preventDefault(); state.run = run; state.step = null; render(); } }},
      h('div', {class:'top'}, h('span', {class:'who'}, run, h('small', toks.length + ' tokens')), verdictPill('', r)),
      toks.length ? tape(toks) : h('div', {class:'note'}, '(empty answer)'),
      !r.parsed && toks.length ? h('div', {class:'note'}, 'Does not parse: a token is not an op of the basis.') : null);
    progs[run] = p; progCol.append(p);
  }
  progCol.append(h('details', {class:'wit'}, h('summary', 'corpus witness (' + t.witness_length + ' ops)'), tape(t.witness.split(' '))));
  art.append(h('div', {class:'twocol'}, h('div', {class:'block'}, h('h4', 'examples'), h('div', {class:'exwrap'}, tbl)), progCol));

  // tracer
  const tracer = h('div', {class:'tracer', tabindex:0, 'aria-label':'program tracer'});
  const seg = h('div', {class:'seg'});
  const segBtn = {};
  for (const run of RUNS){ segBtn[run] = h('button', {class:run, onclick: () => { state.run = run; state.step = null; render(); }}, run); seg.append(segBtn[run]); }
  const sel = h('select', {onchange: () => { state.ex = +sel.value; state.step = null; render(); }});
  exs.forEach((e, i) => sel.append(h('option', {value:i}, (e.kind === 'shown' ? 'shown ' : 'held-out ') + (e.i + 1) + ' · ' + e.input)));
  const range = h('input', {type:'range', min:0, max:0, value:0, oninput: () => { state.step = +range.value; render(false); }});
  const readout = h('div', {class:'readout'});
  const tapeBox = h('div', {class:'tapebox'});
  const stackBox = h('div', {class:'stackbox'});
  const status = h('div', {class:'status'});
  const mk = (txt, fn, label) => h('button', {onclick: fn, 'aria-label': label}, txt);
  tracer.append(
    h('div', {class:'tbar'}, seg, sel,
      h('div', {class:'steps'},
        mk('⇤', () => { state.step = 0; render(false); }, 'first'),
        mk('←', () => { state.step = Math.max(0, state.step - 1); render(false); }, 'back one op'),
        range,
        mk('→', () => { state.step = Math.min(+range.max, state.step + 1); render(false); }, 'forward one op'),
        mk('⇥', () => { state.step = +range.max; render(false); }, 'last')),
      readout),
    h('div', {class:'tbody'}, h('div', tapeBox, status), stackBox),
    h('div', {class:'capnote'}, 'Every trace was computed with the repository interpreter (wandering_light.stack_machine) from the saved candidate; nothing here was re-sampled from a model.'));
  tracer.addEventListener('keydown', ev => {
    if (ev.target.tagName === 'SELECT') return;
    if (ev.key === 'ArrowRight'){ ev.preventDefault(); state.step = Math.min(+range.max, state.step + 1); render(false); }
    if (ev.key === 'ArrowLeft'){ ev.preventDefault(); state.step = Math.max(0, state.step - 1); render(false); }
    if (ev.key === 'Home'){ ev.preventDefault(); state.step = 0; render(false); }
    if (ev.key === 'End'){ ev.preventDefault(); state.step = +range.max; render(false); }
  });
  art.append(tracer);

  let prevLen = -1, prevKey = '';
  function render(full){
    if (full === undefined) full = true;
    const r = t.runs[state.run];
    const res = resultOf(t, state.run, state.ex);
    const toks = r.c ? r.c.split(' ') : [];
    for (const run of RUNS){ segBtn[run].classList.toggle('on', run === state.run); progs[run].classList.toggle('on', run === state.run); }
    sel.value = state.ex;
    rows.forEach((tr, i) => tr.classList.toggle('cur', i === state.ex));
    if (!res){
      range.max = 0; range.value = 0; readout.textContent = '';
      tapeBox.replaceChildren(h('div', {class:'lbl'}, h('span', 'program')), toks.length ? tape(toks) : h('div', {class:'note'}, '(empty answer)'));
      stackBox.replaceChildren(h('div', {class:'lbl'}, h('span', 'stack')), h('div', {class:'stack'}, h('div', {class:'empty'}, 'nothing to run')));
      status.className = 'status run'; status.textContent = 'This candidate does not parse, so the machine never ran it. It is graded as failed.';
      return;
    }
    const steps = res.s, refused = res.ra !== null && res.ra !== undefined;
    const last = steps.length - 1;
    const key = state.run + '|' + state.ex;
    if (state.step === null || state.step === undefined || key !== prevKey){
      state.step = refused ? last : last;
      if (key !== prevKey) prevLen = -1;
    }
    prevKey = key;
    state.step = Math.max(0, Math.min(last, state.step));
    range.max = last; range.value = state.step;
    const atRefusal = refused && state.step === last;
    const cur = state.step === 0 ? undefined : state.step - 1;
    tapeBox.replaceChildren(h('div', {class:'lbl'}, h('span', 'program · ' + state.run), h('span', toks.length + ' tokens')),
      tape(toks, {cur: atRefusal ? undefined : cur, ref: atRefusal ? res.ra : undefined}));
    if (state.step === 0) readout.innerHTML = '<b>start</b> · input stack, before any op';
    else if (atRefusal) readout.innerHTML = 'op <b>' + (res.ra + 1) + '</b> of ' + toks.length + ' · <b>' + esc(toks[res.ra]) + '</b> refused';
    else readout.innerHTML = 'after op <b>' + state.step + '</b> of ' + toks.length + ' · <code>' + esc(toks[state.step - 1]) + '</code>';
    // stack
    const st = steps[state.step];
    const stack = h('div', {class:'stack'});
    if (!st.length) stack.append(h('div', {class:'empty'}, 'empty stack'));
    let markOpen = false;
    st.forEach((v, i) => {
      const k = kindOf(v);
      if (k === 'mark') markOpen = true;
      const tile = h('div', {class:'tile ' + k + (i === st.length - 1 ? ' top' : '') + (i >= prevLen && prevLen >= 0 && full === false ? ' new' : '')}, v);
      if (k === 'sym' && !markOpen) tile.classList.remove('sym');
      stack.append(tile);
    });
    prevLen = st.length;
    stackBox.replaceChildren(h('div', {class:'lbl'}, h('span', 'stack'), h('span', st.length + (st.length === 1 ? ' value' : ' values'))), stack);
    // status
    if (atRefusal){
      status.className = 'status illegal';
      status.innerHTML = 'The machine refused op ' + (res.ra + 1) + ', <code><b>' + esc(toks[res.ra]) + '</b></code>: <b>' + esc(res.why) + '</b>. An illegal program scores as failed on every example, shown or not.';
    } else if (state.step === last){
      if (res.r === 'pass'){ status.className = 'status pass'; status.innerHTML = 'Done. Output <code>' + esc(res.g) + '</code> matches the expected stack.'; }
      else { status.className = 'status wrong'; status.innerHTML = 'Done. Output <code>' + esc(res.g) + '</code> — expected <code>' + esc(res.e.replace(/^\[ | \]$/g, '')) + '</code>.'; }
    } else {
      status.className = 'status run';
      status.textContent = (last - state.step) + ' op' + (last - state.step === 1 ? '' : 's') + ' to go' + (refused ? ' before the machine refuses' : '') + '.';
    }
  }
  render();
  art._state = state;
  return art;
}

/* ---------- the five cases ---------- */
const LESSONS = {
  25: `<p>The witness is six ops: <code>[ lt? ] swap cons filter</code>. The number sitting under the list has to get <em>into</em> the filter body, because a body runs on one element at a time in a fresh stack that holds only that element. Luna wrote <code>swap [ swap lt? ] filter swap drop</code>, leaving the threshold on the outer stack and asking the body to <code>swap</code> it into position. Inside the body there is one value, so the machine refuses — <em>swap: needs 2 value(s), stack has 1</em> — on every input, shown and held out. Step to op 5 and watch: the threshold <code>9</code> is still on the outer stack when <code>filter</code> starts.</p>
<p>Astra wrote <code>quote [ lt? ] cat filter</code>: wrap the threshold as a one-item quotation <code>(9)</code>, glue it onto <code>(lt?)</code>, and filter with <code>(9 lt?)</code>. That is closure capture in this language, and it is the whole trick. Trace astra's program and the quotation grows on the stack before <code>filter</code> consumes it. Task #27 (<code>n_minus_each</code>) is the same story word for word: luna <code>swap [ sub ] map swap drop</code>, astra <code>quote [ swap sub ] cat map</code>.</p>`,
  38: `<p>Running totals need two things carried through the list: the sum so far and the output built so far. Luna's plan is right — seed a pair, fold, reverse — and it dies on bookkeeping. Its fold body <code>swap uncons swap [ add ] dip over cons</code> leaves two values on the body's stack, and a fold body must leave exactly one (the next accumulator). The machine refuses at the <code>fold</code>: <em>body must leave exactly one value, left 2</em>. The trace stops at op 16 with the seed <code>(0 nil)</code> and the body quotation sitting there, never run to completion.</p>
<p>Astra packs both things into <em>one</em> list accumulator whose head is the running sum: seed <code>(0)</code>, body <code>[ dup uncons swap drop ] dip add cons</code> peels the current sum off the head, adds the element, and conses the new sum on. After the fold, <code>nil [ cons ] fold</code> reverses and <code>uncons drop</code> removes the seed. Eighteen tokens against a 25-op witness. The same lesson at 49 ops is task #7 (<code>positive_balance_prefix_count</code>), where astra keeps a two-cell accumulator <code>[ 0 0 ]</code> and unpacks it with <code>i</code>, and luna dies in the same fold trying to <code>add</code> a list to a number.</p>`,
  3: `<p>Look at what luna and astra were shown: <code>(2 5 -9) → (2 5)</code>, <code>(-2 -8) → (-2)</code>, <code>(9 8 -4) → (9 8)</code>. In every pair the last element is also the only element at or below −3. Astra's answer, <code>[ 1 1 add 1 add add 0 swap lt? ] filter</code>, keeps each <code>x</code> where <code>0 &lt; x + 3</code>: a threshold filter, not a drop-last. It reproduces all three shown outputs and fails all eight hidden ones — trace it on <code>(3 -3 3 8 -6)</code> and it returns <code>(3 3 8)</code>, dropping the −3 in the middle as well as the −6 at the end. This is the memorisation the held-out pairs exist to catch: the shown sample admitted a simpler hypothesis, and the solver took it.</p>
<p>Luna's candidate is a broken pair accumulator that the machine refuses at the first <code>fold</code> (<em>uncons: top is not a list</em>). Neither program computes the behaviour. The twist is task #32, the same behaviour with a different draw of examples (<code>(-5 9 -2)</code>, <code>(4 2 1 5 7 8)</code>, <code>(8 -9 -8 -1)</code>) where no threshold separates the last element: there astra wrote the witness verbatim, <code>nil [ cons ] fold uncons drop nil [ cons ] fold</code>, and solved. Astra knows drop-last. On #3 the data let it stop early.</p>`,
  8: `<p>The input is three loose numbers on the stack; the output is <code>1</code> when at least two are positive. Luna's <code>0 [ add ] fold 0 swap lt?</code> treats the three numbers as if they were a list and is refused at <code>fold</code>: <em>third value is not a list</em>. A misreading of the input shape, refused on all thirteen examples.</p>
<p>Astra's <code>add add quote [ 0 swap lt? ] filter 0 [ drop drop 1 ] fold</code> asks a different question: is the <em>sum</em> positive? That is a plausible proxy for a positive majority, and it agrees with the real behaviour on all five shown pairs and on seven of the eight held out. The one that separates them is <code>9 -2 -1</code>: sum 6, so astra says <code>1</code>; two of three are negative, so the answer is <code>0</code>. With a one-bit output a wrong program passes each held-out pair about half the time, so eight pairs are a thin net, and this task's net caught the proxy by a single strand. The other <code>positive_majority_three</code> task in the sample (#15) shows the same thing from the other side: both models reach for <code>mul mul</code> sign tricks, and the held-out set catches both.</p>`,
  5: `<p>The list holds flat pairs, and the answer is the largest <code>|a − b|</code> within any pair. The witness is 84 ops. Neither model attempted the pair walk; both encoded the answers. Luna's program counts the list's length with <code>0 [ drop 1 add ] fold</code> and then routes it through two nested <code>ifte</code>s: length 10 → 17, length 4 → 10, anything else → 2. It is a lookup table of the shown outputs, and it even mistranscribes 11 as 10, so it fits two of the three shown pairs (the default trace shows the wrong one). It runs legally and is simply wrong.</p>
<p>Astra's <code>[ 0 lt? false eq? ] filter 0 [ drop 1 add ] fold 1 1 add 1 add mul 1 add 1 add</code> counts the non-negative elements and returns <code>3 × count + 2</code>: 5 → 17, 3 → 11, 0 → 2. Three data points, one affine function, built from the only literals the machine has (<code>1 1 add 1 add</code> is how you say 3). It fits every shown pair and fails seven of eight hidden ones. A grader without held-out pairs would have called this a solve. The same instinct at its purest is task #28 (<code>bounds_cover_list_flag</code>), where astra's <code>sub swap drop 0 1 sub 1 sub 1 sub div 1 sub</code> drops the list without reading it and fits three shown outputs from the bounds alone.</p>`,
};
(function cases(){
  const list = document.getElementById('caselist');
  FEATURED.forEach((f, i) => list.append(renderCase(byPos[f.pos], {number: i + 1, title: f.title, lesson: LESSONS[f.pos], run: f.run, ex: f.ex, exInput: f.exInput, id: 'case-' + (i + 1)})));
})();

/* ---------- essays ---------- */
document.getElementById('modestext').innerHTML = `
<p>The disagreement is not a coin flip between two similar solvers. Every one of the twenty differing tasks is a luna failure, and on seventeen of them the program never ran to completion: the machine refused an op. Astra returned a program that runs and reproduces every shown pair on 38 of 40 tasks; whether it had actually computed the behaviour was decided entirely by the eight hidden pairs.</p>
<h3>Luna loses the stack</h3>
<p>Luna's plans are often right and its bookkeeping is not. It builds quotation bodies that expect to see values on the outer stack (case 1); it carries two accumulators through a fold as two stack items when the body may return one (case 2); it misreads the input shape and folds over three scalars (case 4). Eighteen of its 22 failures are refusals. When it does give up on the algorithm and hard-code, it hard-codes badly (case 5). The baseline report's phrase for this was that on longer witnesses the model is losing the stack, not the algorithm, and the traces bear it out.</p>
<h3>Astra always fits, sometimes by any means</h3>
<p>Where astra knows the idiom it writes short, correct programs: closure capture by <code>quote … cat</code> (case 1), a list as accumulator (case 2), the witness itself when the data leaves no shortcut (task #32). Where it does not, it finds <em>some</em> function of the shown data: a threshold filter that happens to drop the last element (case 3), a sign proxy (case 4), an affine function of a count (case 5), arithmetic on the bounds that never reads the list (#28). Fifteen of its 38 shown-fits are memorisations. Its shown-only score is 95%; its real score is 57.5%. The stronger the solver, the more the held-out protocol earns.</p>
<h3>What the corpus contributes</h3>
<ul>
<li><b>Bit- and count-valued behaviours are where memorisation clusters.</b> <code>positive_majority_three</code>, <code>bounds_cover_list_flag</code>, <code>axis_point_count</code>, <code>adjacent_sign_change_count</code>, <code>widest_pair</code>: small outputs, many coincidental fits. Eight held-out pairs are weak evidence for a one-bit output (case 4 was caught by one strand).</li>
<li><b>Some shown draws underdetermine the behaviour.</b> Task #3's three pairs admit a threshold filter; #32's do not. The same solver solves one and memorises the other, so a per-task verdict on a single draw carries sampling noise from the corpus side as well as the model side.</li>
<li><b>Where both memorise, they memorise the same thing.</b> On #9 and #14 luna and astra wrote byte-identical proxy programs (count the negatives; count the zeros). The shown data pulls both toward the same simplest consistent hypothesis, which says the hypothesis is a property of the examples, not of the model.</li>
</ul>
<div class="aside"><p class="eyebrow">Caveats carried from the source reports</p><p class="small" style="margin:0">One sample per task at default temperature; same-task verdicts flip between runs of the same model (luna API vs luna through codex exec agreed on 33/40). The astra arm ran with the owner's <code>~/.codex</code> page in context (fixed later; a re-run of the low arm moved 24 → 23, within noise). The 40-task sample is easier than the full split (median witness 26 vs 40 ops). 180 of 220 behaviours were written by <code>gpt-5.6-sol</code>, so these are corpus-calibration numbers, not a model ranking.</p></div>`;

document.getElementById('capturetext').innerHTML = `
<p>Everything on this page was rebuilt from two candidate strings per task plus the corpus. That worked, but only because the sample could be re-drawn and the grader re-run. Each item below is something the comparison needed and had to recompute or infer.</p>
<ol>
<li><b>A task key in every row.</b> The luna file has no <code>task_index</code>; it was joined by re-drawing the seed-17 sample and checking behaviour and witness sequence. Record <code>task_index</code>, the corpus name, its Hub revision and payload digest in every run, so a join never depends on the loader's order or the RNG protocol staying fixed.</li>
<li><b>The run header.</b> The luna file records none of model, provider, effort, prompt version, temperature or sampling parameters, timestamp, or git SHA; the codex-era files record the first three. All of these belong in the JSON, not only in a README.</li>
<li><b>Per-example verdicts, with the refusal.</b> Store, for each shown and held-out pair, pass / wrong / refused, the produced stack when it ran, and for a refusal the op index and the machine's reason. It is cheap, it makes <em>illegal</em> analysable without a rerun, and it separates "refused on the first input" from "refused on one held-out input".</li>
<li><b>The raw model text.</b> Only the parsed candidate is saved. Keep the full response so a parse failure can be told from a wrong answer, and so reasoning summaries survive for the arms that produce them.</li>
<li><b>Generation cost for API rows.</b> The codex rows carry tokens and seconds per call; the API rows carry nothing. Same fields everywhere.</li>
<li><b>More than one sample, or a seed.</b> With one sample per task, a 3-task gap is inside run-to-run noise. Either k ≥ 2 samples per task, scored as best-of and as pass@1, or a fixed sampling seed where the provider offers one.</li>
<li><b>Output shape per behaviour.</b> Tag each behaviour's output as flag, count, scalar or list (derivable from the reference, so it can be added to the corpus manifest) so memorisation rates can be stratified, and size the held-out count to the output's entropy, which the codex-exec report already filed on its issue.</li>
<li><b>A schema version on the run JSON.</b> The two files compared here are two schemas apart with no field saying so.</li>
</ol>`;

/* ---------- browse ---------- */
const pick = document.getElementById('pick');
TASKS.forEach(t => pick.append(h('option', {value:t.position}, '#' + String(t.position).padStart(2,'0') + '  ' + t.behaviour + '  (' + t.witness_length + ' ops)  luna: ' + t.runs.luna.v + ' · astra: ' + t.runs.astra.v)));
function loadBrowse(pos){
  pick.value = pos;
  const t = byPos[pos];
  document.getElementById('browsecase').replaceChildren(renderCase(t, {run: t.runs.luna.v !== 'solved' ? 'luna' : 'astra', id:'browse-task'}));
}
pick.addEventListener('change', () => loadBrowse(+pick.value));
loadBrowse(7);

/* ---------- provenance ---------- */
(function prov(){
  const m = DATA.meta;
  const el = document.getElementById('provtext');
  const tbl = h('table');
  tbl.append(h('tr', h('th', 'check'), h('th', 'luna'), h('th', 'astra')));
  const fmt = v => v === true ? h('span', {class:'ok'}, 'agrees') : v === 'missing' ? h('span', {class:'miss'}, 'not recorded') : h('span', {class:'miss'}, String(v));
  for (const [label, key] of [['behaviour sequence vs corpus sample', 'behaviour_sequence'], ['task_index vs corpus sample', 'task_index'], ['witness vs corpus sample', 'witness']]){
    tbl.append(h('tr', h('td', label), RUNS.map(run => h('td', fmt(m.runs[run].alignment[key])))));
  }
  tbl.append(h('tr', h('td', 'regrade with the current grader vs saved verdict'), RUNS.map(run => h('td', h('span', {class:'ok'}, (40 - m.runs[run].regrade_mismatches) + '/40 agree')))));
  tbl.append(h('tr', h('td', 'run fields not recorded'), RUNS.map(run => { const miss = Object.entries(m.runs[run].run_fields).filter(([k,v]) => v === 'missing').map(([k]) => k); return h('td', miss.length ? h('span', {class:'miss'}, miss.join(', ')) : h('span', {class:'ok'}, 'none')); })));
  tbl.append(h('tr', h('td', 'per-candidate fields absent'), RUNS.map(run => { const cf = m.runs[run].candidate_fields; const miss = Object.entries(cf).filter(([k,n]) => n === 0 && k !== 'errors').map(([k]) => k); return h('td', miss.length ? h('span', {class:'miss'}, miss.join(', ')) : h('span', {class:'ok'}, 'none')); })));
  el.append(h('p', 'Corpus ', h('code', m.corpus), ' (Hub revision 00c2a999, manifest-verified), split ', h('code', m.split), ', seed ', h('code', m.seed), ', 40 tasks drawn the way ', h('code', 'experiments/stack_pbe_baseline.py'), ' draws them. Files: ', h('code', RUN.luna.file), ' and ', h('code', RUN.astra.file), '. Paired data and traces: ', h('code', 'paired.json'), ' from ', h('code', 'experiments/stack_pbe_compare_runs.py'), '; this page from ', h('code', 'build_page.py'), ' beside it.'));
  el.append(tbl);
  el.append(h('p', {style:'margin-top:12px'}, 'Nothing here was guessed: where a file lacks a field the table says so, and the join used only fields both the file and the corpus carry. The luna file predates the schema that records task_index and the run header; the astra file omits the witness by design (it is 58% of an artifact and lives in the pinned corpus). "errors" is written only on a failed generation, so its absence means no call failed.'));
})();
})();
</script>"""


def load_paired(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return json.load(handle)


def slim(paired: dict) -> dict:
    """What the page needs from `paired.json`: examples, per-op traces, verdict counts."""
    labels = list(paired["runs"])

    def example(x):
        t = x["trace"]
        return {
            "i": x["input"],
            "e": x["expected"],
            "r": x["result"],
            "g": x["got"],
            "s": t["steps"],
            "ra": t["refused_at"],
            "why": t["reason"],
        }

    tasks = []
    for r in paired["tasks"]:
        row = {
            k: r[k]
            for k in (
                "position",
                "task_index",
                "behaviour",
                "summary",
                "witness",
                "witness_length",
                "tags",
                "shown",
                "held_out",
            )
        }
        row["runs"] = {}
        for label in labels:
            e = r["runs"][label]
            row["runs"][label] = {
                "c": e["candidate"],
                "v": e["verdict"],
                "legal": e["legal"],
                "parsed": e["parsed"],
                "shown": [example(x) for x in e["shown"]],
                "held": [example(x) for x in e["held_out"]],
            }
        tasks.append(row)
    stats = {}
    for label in labels:
        verdicts = Counter(t["runs"][label]["v"] for t in tasks)
        legal = sum(t["runs"][label]["legal"] for t in tasks)
        wrong = sum(
            1
            for t in tasks
            if t["runs"][label]["legal"] and t["runs"][label]["v"] == "failed"
        )
        stats[label] = {
            "solved": verdicts["solved"],
            "memorised": verdicts["memorised"],
            "failed": verdicts["failed"],
            "illegal": len(tasks) - legal,
            "runs_but_wrong": wrong,
            "shown_fit": verdicts["solved"] + verdicts["memorised"],
        }
    meta = {k: paired[k] for k in ("corpus", "split", "seed", "runs", "contingency")}
    return {"meta": {**meta, "stats": stats}, "tasks": tasks}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paired",
        default=str(HERE / "paired.json.gz"),
        help=".json or .json.gz from stack_pbe_compare_runs.py",
    )
    parser.add_argument("--out", default=str(HERE / "index.html"))
    parser.add_argument(
        "--artifact",
        default=None,
        help="also write the page without the document skeleton",
    )
    args = parser.parse_args(argv)
    data = json.dumps(
        slim(load_paired(Path(args.paired))), separators=(",", ":")
    ).replace("</", "<\\/")
    body = BODY.replace("__DATA__", data)
    Path(args.out).write_text(
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        + HEAD
        + "\n</head>\n<body>\n"
        + body
        + "\n</body>\n</html>\n"
    )
    print(f"wrote {args.out} ({Path(args.out).stat().st_size:,} bytes)")
    if args.artifact:
        Path(args.artifact).write_text(HEAD + "\n" + body + "\n")
        print(f"wrote {args.artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
