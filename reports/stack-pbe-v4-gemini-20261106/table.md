| model | effort | solved | memorised | tasks | errors | out tok med / mean | think tok mean | prompt tok med | $/task | finish | served | s/call med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `gemini-4-argon` | low | **12** | 0 | 40 | {'ReadTimeout': 1} | 21 / 30 | 0 | 566 | 0.0029 | {'STOP': 39} | gemini-4-argon (39/40) | 14 |
| `gemini-4-argon` | medium | **19** | 7 | 40 | {'ReadTimeout': 7} | 566 / 2,062 | 2,042 | 566 | 0.0232 | {'STOP': 32, 'PROHIBITED_CONTENT': 1} | gemini-4-argon (33/40) | 31 |
| `gemini-4-argon` | high | **23** | 5 | 40 | {'ReadTimeout': 12} | 3,552 / 4,912 | 4,890 | 578 | 0.0517 | {'STOP': 28} | gemini-4-argon (28/40) | 59 |
| `gemini-3.8-flash` | low | **17** | 4 | 40 | {'RemoteProtocolError': 2, 'ReadTimeout': 3} | 2,043 / 4,681 | 4,664 | 544 | 0.0185 | {'STOP': 34, 'PROHIBITED_CONTENT': 1} | gemini-3.8-flash (35/40) | 74 |
| `gemini-3.8-flash` | medium | **19** | 2 | 40 | {'ReadTimeout': 10} | 5,900 / 9,575 | 9,558 | 554 | 0.0369 | {'STOP': 30} | gemini-3.8-flash (30/40) | 98 |
| `gemini-3.8-flash` | high | **9** | 3 | 36 PARTIAL | {'ReadTimeout': 20, 'HTTPStatusError': 1} | 17,690 / 15,772 | 15,760 | 539 | 0.0601 | {'STOP': 15} | gemini-3.8-flash (15/36) | 149 |
