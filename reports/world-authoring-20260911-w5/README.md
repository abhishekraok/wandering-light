# world-authoring-500 (#127), wave 5: 100 MBPP / CodeARC behaviours authored by codex exec

Behaviours sampled by `experiments/world_behaviours/extract.py` (seed 17, MBPP remainder and CodeARC, excluding the probe and waves 1–4, hypothesis-breaking sampler on every source); programs authored by four `codex exec` agents (gpt-6-astra, high effort) in macro form and migrated to the canonical library; every program re-verified on every test by `report.py`.

## Findings (written with the 2026-09-11 wave 5; the tables below regenerate)

- **98 of 100 verified** (49/50 MBPP, 49/50 CodeARC); the misses are one
  CPython set-iteration order (`mbpp-193`, the third of its kind) and one
  fuel exhaustion (a prime sieve to large bounds). Over the probe and five
  waves MBPP stands at 283/286 and CodeARC at 229/234.
- **Cost:** 318k tokens, 24 agent-minutes, 10 minutes wall.
- **Library: 4 macros per 100** (129 → 133): `enumerate` (three programs),
  `highest-pow2` (two), `choose2` (also written in wave 2), and `index-of`,
  the concept two authors across waves wrote as private helpers with
  different behaviour on a missing element. A task-specific pair helper used
  by two MBPP programs (`first-even-odd`) stayed inlined: two uses of one
  problem family is not a concept. Growth per 100 over the five waves: 16, 7,
  8, 4, 4.
- **Overlap with v4:** 7 exact witness matches, all scalar one-liners.
- **The sampler-passing pool is nearly spent:** this wave tried 96 MBPP
  candidates to keep 50 and 69 CodeARC to keep 50. About 96 MBPP and 31
  CodeARC behaviours remain eligible.

## Verified

| source | verified | median witness length (`mark-28` ops) | median macro-form length |
|---|---|---|---|
| codearc | 49/50 (98%) | 158 | 20 |
| mbpp | 49/50 (98%) | 109 | 13 |
| **all** | 98/100 (98%) | 128.0 | 16.5 |

Claimed-vs-verified mismatches: 0; missing ids: 0; duplicates: 0.

By author: canonical 98/100 (98%).

## Witness length against `stack_pbe_v4`

| bucket (ops) | this probe (verified) | v4 behaviours (train + eval, 220) |
|---|---|---|
| 1-5 | 6 | 15 |
| 6-10 | 3 | 14 |
| 11-20 | 8 | 34 |
| 21-40 | 6 | 37 |
| 41+ | 75 | 120 |
| median | 128.0 | 43.0 |

Exact matches with a v4 witness: 7 (codearc-103, codearc-9, mbpp-168, mbpp-62, mbpp-750, mbpp-798, mbpp-865). Same skeleton as a v4 witness: 7.

## Macros

133 macros defined across `macros*.txt`; median expansion ratio (`mark-28` ops per macro-form token) on verified programs: 7.2.

| macro | verified programs using it |
|---|---|
| `head` | 25 |
| `2` | 24 |
| `len` | 22 |
| `inc` | 21 |
| `sum` | 19 |
| `unpack2` | 17 |
| `range` | 16 |
| `mod` | 16 |
| `take` | 14 |
| `pack2` | 14 |
| `dec` | 14 |
| `gt?` | 14 |
| `odd?` | 13 |
| `even?` | 12 |
| `drop-n` | 11 |
| `reverse` | 10 |
| `pick2` | 10 |
| `le?` | 10 |
| `sq` | 10 |
| `range1` | 10 |

## Unsolved

| id | source | passed | note |
|---|---|---|---|
| codearc-991 | codearc | 11/34 | why: odd-candidate sieve and prime-index lookup exceed the 100000-step fuel limit on larger bounds; best candidate retai |
| mbpp-193 | mbpp | 23/27 | why: Python set iteration requires hash-table probing and resize order; sorted deduplication passes 23/27, failing four  |

## Every verified program

| id | len | macro form |
|---|---|---|
| mbpp-750 | 1 | `cat` |
| mbpp-106 | 2 | `swap cat` |
| mbpp-499 | 2 | `dup add` |
| mbpp-234 | 4 | `dup sq mul` |
| mbpp-502 | 5 | `mod` |
| mbpp-798 | 5 | `sum` |
| codearc-9 | 6 | `[ 0 gt? ] filter` |
| mbpp-865 | 6 | `[ mul ] swap cons map` |
| mbpp-436 | 8 | `[ 0 lt? ] filter head` |
| codearc-103 | 13 | `count` |
| codearc-144 | 13 | `[ product ] dip mod` |
| codearc-148 | 13 | `[ odd? ] filter` |
| mbpp-168 | 13 | `count` |
| mbpp-339 | 15 | `[ eq? ] [ swap drop ] [ drop drop 2 ] ifte` |
| codearc-203 | 16 | `inc 2 div sq swap 2 div sq sub` |
| codearc-425 | 16 | `2 div over sub dup [ sub ] dip pack2` |
| mbpp-62 | 16 | `min` |
| mbpp-378 | 23 | `[ null? ] [ ] [ reverse uncons [ reverse ] dip cons ] ifte` |
| codearc-458 | 27 | `dup [ swap count 1 eq? ] swap cons filter sum` |
| codearc-799 | 27 | `pack3 [ 3 mul 4 div ] map product` |
| mbpp-162 | 28 | `0 max2 dup inc 2 div over over sub inc mul swap drop` |
| codearc-365 | 35 | `[ uncons quote ] dip [ [ [ over head add ] dip eq? ] [ drop drop ] [ drop cons ] ifte ] swap cons fold reverse` |
| mbpp-161 | 35 | `[ swap contains 0 eq? ] swap cons filter` |
| codearc-159 | 45 | `dup [ even? ] filter head-or-neg1 swap [ odd? ] filter head-or-neg1 sub` |
| mbpp-784 | 45 | `dup [ even? ] filter head-or-neg1 swap [ odd? ] filter head-or-neg1 mul` |
| mbpp-881 | 45 | `dup [ even? ] filter head-or-neg1 swap [ odd? ] filter head-or-neg1 add` |
| mbpp-510 | 50 | `swap [ 0 gt? ] filter nil 1 cons [ over swap [ mul ] swap cons map cat ] fold tail swap [ le? ] swap cons filter len` |
| codearc-173 | 57 | `sort` |
| codearc-45 | 58 | `[ abs 10 mod ] dip abs 10 mod mul` |
| codearc-702 | 58 | `dup [ odd? ] filter swap [ even? ] filter [ over len over len lt? ] [ drop ] [ swap drop ] ifte head` |
| codearc-399 | 59 | `0 0 pack2 [ [ unpack2 ] dip rot add over max2 pack2 ] fold unpack2 max2` |
| codearc-434 | 59 | `[ null? ] [ ] [ uncons nil swap pack2 [ [ unpack2 ] dip [ over over gt? ] [ swap ] [ ] ifte [ cons ] dip pack2 ] fold unpack2 cons reverse ] ifte` |
| mbpp-471 | 63 | `[ take product ] dip mod` |
| mbpp-867 | 68 | `take sum 2 mod 2 swap sub` |
| codearc-717 | 73 | `pack3 dup [ unpack3 mul mul ] dip dup [ unpack3 [ add ] dip dup dec mul mul 2 div ] dip third dup dec over 2 mul dec mul mul 6 div add add` |
| mbpp-728 | 75 | `zip [ sum ] map` |
| mbpp-288 | 76 | `[ take ] dip [ swap sq swap mod 1 eq? ] swap cons filter len` |
| mbpp-682 | 76 | `zip [ unpack2 mul ] map` |
| mbpp-101 | 78 | `[ drop ] dip dec at` |
| mbpp-358 | 80 | `zip [ unpack2 mod ] map` |
| codearc-151 | 90 | `take dup [ swap [ eq? not ] swap cons filter len ] swap cons map sum 2 div` |
| mbpp-166 | 109 | `take dup [ even? ] filter len choose2 swap [ odd? ] filter len choose2 add` |
| codearc-293 | 115 | `over sub inc range swap [ add ] swap cons map` |
| codearc-447 | 118 | `[ dup dec mul ] dip 2 mul add 4 mul isqrt inc 2 div` |
| codearc-125 | 123 | `pows2 swap [ le? ] swap cons filter last` |
| mbpp-388 | 123 | `highest-pow2` |
| codearc-1073 | 127 | `pack2 zeros16 dup cat swap [ drop unpack2 [ over over mul 0 eq? ] [ ] [ [ over over 2 mul ge? ] [ dup [ 2 mul mod ] dip ] [ [ over over swap 2 mul ge? ] [ over 2 mul mod ] [ ] ifte ] ifte ] ifte pack2 ] fold` |
| codearc-177 | 127 | `adjacent [ product ] map` |
| mbpp-470 | 128 | `adjacent [ unpack2 add ] map` |
| mbpp-726 | 128 | `adjacent [ unpack2 mul ] map` |
| codearc-574 | 131 | `[ range1 ] dip [ swap [ add ] dip mod ] swap cons 0 swap fold inc` |
| mbpp-302 | 135 | `[ dup 0 lt? ] [ drop 1 ] [ highest-pow2 ] ifte` |
| codearc-922 | 149 | `differences [ abs ] map sum` |
| codearc-73 | 158 | `[ 10 gt? ] filter [ dup odd? bool>int swap digits head odd? bool>int mul 1 eq? ] filter len` |
| codearc-51 | 172 | `[ decimal-digits [ even? ] filter null? ] filter sort` |
| mbpp-453 | 172 | `[ dup 0 eq? ] [ drop 1 ] [ [ dup 2 eq? ] [ drop 3 ] [ dup range1 [ even? ] filter swap [ swap divides? ] swap cons filter sum ] ifte ] ifte` |
| mbpp-25 | 175 | `[ sort ] dip over last [ take ] dip 1 pack2 [ [ unpack2 ] dip [ pick2 eq? ] [ drop pack2 ] [ rot drop swap over mul pack2 ] ifte ] fold second` |
| codearc-570 | 184 | `[ null? ] [ drop 0 1 pack2 ] [ [ dup len 1 eq? ] [ head 1 ] [ unpack2 ] ifte [ dup 0 lt? ] [ neg [ neg ] dip ] [ ] ifte over over gcd [ pack2 ] dip [ div ] swap cons map ] ifte` |
| mbpp-754 | 191 | `[ zip ] dip zip [ unpack2 swap unpack2 over eq? bool>int [ eq? bool>int ] dip mul 1 eq? ] filter [ head head ] map` |
| mbpp-170 | 199 | `over sub inc range swap [ add ] swap cons map swap [ swap at ] swap cons map sum` |
| codearc-983 | 212 | `enumerate [ second odd? ] filter [ head ] map head-or-neg1` |
| codearc-295 | 213 | `dup len range zip [ unpack2 [ odd? ] [ drop sq ] [ drop ] ifte ] map sum` |
| codearc-778 | 217 | `dup dedup swap [ swap dup [ count ] dip swap pack2 ] swap cons map dup seconds dup min swap max [ over eq? ] [ drop drop drop nil ] [ swap drop [ [ second ] dip eq? ] swap cons filter [ head ] map sort ] ifte` |
| codearc-606 | 222 | `decimal-digits [ sq decimal-digits ] map flatten from-digits` |
| codearc-113 | 224 | `enumerate [ head even? ] filter seconds [ even? ] filter sum` |
| codearc-393 | 226 | `range [ drop 0 ] map [ over min dup [ add ] dip swap [ remove1 ] dip cons ] fold max` |
| codearc-770 | 229 | `[ dup 0 count 1 gt? ] [ min ] [ [ dup [ 0 lt? ] filter len odd? ] [ [ dup 0 contains 1 eq? ] [ [ 0 lt? ] filter min ] [ [ 0 lt? ] filter max ] ifte ] [ [ dup [ 0 ge? ] filter null? ] [ [ 0 lt? ] filter min ] [ [ 0 ge? ] filter min ] ifte ] ifte ] ifte` |
| mbpp-908 | 253 | `take enumerate [ unpack2 eq? ] filter [ head ] map head-or-neg1` |
| codearc-896 | 261 | `differences 1 0 0 1 pack4 [ [ unpack4 ] dip [ dup 0 gt? ] [ drop [ [ drop dup ] dip ] dip ] [ [ dup 0 lt? ] [ drop max2 pick2 pick2 sub ] [ drop ] ifte ] ifte inc [ inc ] dip3 pack4 ] fold unpack4 max2 [ drop drop ] dip` |
| codearc-797 | 269 | `dup 8 mul inc isqrt dec 2 div range1 swap [ swap dup [ mod ] dip [ dup odd? ] [ drop 0 eq? ] [ 2 div eq? ] ifte ] swap cons filter len` |
| codearc-138 | 272 | `[ head 0 lt? ] [ [ abs decimal-digits ] map flatten from-digits neg ] [ [ decimal-digits ] map flatten from-digits ] ifte` |
| codearc-620 | 272 | `dup len range swap [ swap drop-n dup head 0 0 pack3 [ over head min2 over second inc over over mul pick3 third max2 pack3 swap drop ] fold third ] swap cons map 0 [ max2 ] fold` |
| codearc-1111 | 273 | `dup [ slice-index take reverse ] dip 3 sub range swap [ drop dup 3 take sum cons ] fold reverse` |
| mbpp-763 | 288 | `[ sort ] dip take differences [ neg ] map 100 sq sq 100 mul sq [ min2 ] fold` |
| mbpp-681 | 298 | `[ even? ] [ drop 2 ] [ dup isqrt range1 2 drop-n over [ swap divides? ] swap cons filter swap quote cat head ] ifte` |
| mbpp-84 | 308 | `range1 2 drop-n nil 1 cons 1 cons 0 cons [ over last swap over sub [ over swap nth ] dip pick2 swap nth add quote cat ] fold last` |
| codearc-1024 | 315 | `[ over len over ge? ] [ slice-index take ] [ over len over over sub range [ swap drop ] dip rot reverse rot [ swap drop over swap take sum cons ] swap cons fold reverse ] ifte` |
| mbpp-852 | 324 | `dup len range swap [ [ over len lt? ] [ [ over over nth 0 lt? ] [ over swap nth remove1 ] [ drop ] ifte ] [ drop ] ifte ] fold` |
| mbpp-545 | 325 | `[ dup 1 eq? ] [ drop 0 ] [ [ dup 0 gt? ] [ dup log2 pow2 sub ] [ ] ifte [ dup even? ] [ inc ] [ dec ] ifte ] ifte` |
| codearc-776 | 330 | `sort reverse dup len 2 sub range swap [ swap drop-n 3 take ] swap cons map [ dup head swap tail sum lt? ] filter [ sum ] map 0 [ max2 ] fold` |
| codearc-219 | 345 | `[ decimal-digits ] map flatten nil [ [ dup 9 eq? ] [ [ over null? ] [ cons ] [ [ over head 8 eq? ] [ drop tail dup 0 1 pack2 [ [ unpack2 ] dip 7 eq? bool>int mul dup [ add ] dip pack2 ] fold head over over take [ drop-n ] dip [ 8 cons 9 cons ] dip swap cat ] [ cons ] ifte ] ifte ] [ cons ] ifte ] fold reverse` |
| mbpp-323 | 364 | `over over drop-n [ 0 max2 take dup [ 0 lt? ] filter swap [ 0 ge? ] filter over len over len min2 range -rot nil pack3 [ drop dup head head over second head pack2 over third swap cat [ dup head tail swap second tail ] dip pack3 ] fold unpack3 [ cat ] dip swap cat ] dip cat` |
| mbpp-511 | 386 | `[ dup 1 le? ] [ ] [ dup isqrt range1 tail swap 0 pack2 [ [ unpack2 ] dip [ pick2 over sq ge? ] [ pack3 zeros16 dup cat swap [ drop unpack3 [ pick2 over mod 0 eq? ] [ rot over div -rot dup [ add ] dip ] [ ] ifte pack3 ] fold unpack3 drop pack2 ] [ drop pack2 ] ifte ] fold sum ] ifte` |
| mbpp-724 | 404 | `over abs decimal-digits len over mul inc [ pow ] dip range swap 0 pack2 [ drop unpack2 over 10 mod add swap 10 div swap pack2 ] fold second` |
| mbpp-371 | 415 | `pack3 zeros16 dup cat swap [ drop [ dup second over third le? ] [ dup second over third add 2 div [ over head over at over eq? ] [ inc 1 swap set-at ] [ dec 2 swap set-at ] ifte ] [ ] ifte ] fold second` |
| codearc-355 | 421 | `[ dup 0 lt? ] [ abs dup pows2 swap [ le? ] swap cons filter 1 drop-n reverse swap [ swap div 2 mod 2 mul dec ] swap cons map -1 cons ] [ dup pows2 swap [ le? ] swap cons filter 1 drop-n reverse swap [ swap div 2 mod 2 mul dec ] swap cons map ] ifte 1 cons` |
| codearc-1087 | 460 | `dup [ inc range 3 drop-n swap [ over over dec nth 4 mul pick2 pick2 2 sub nth 5 mul sub pick2 pick2 3 sub nth 3 mul add swap drop quote cat ] fold ] dip at` |
| codearc-452 | 486 | `over abs over abs max2 dup isqrt range1 [ odd? ] filter [ dup dup mul mul ] map swap [ le? ] swap cons filter [ decimal-digits [ even? ] filter null? ] filter dup reverse [ neg ] map swap cat -rot [ between ] swap cons swap cons filter` |
| mbpp-383 | 569 | `[ dup 0 gt? ] [ dup bitlen range [ even? ] filter [ pow2 ] map sum bxor ] [ ] ifte` |
| codearc-337 | 582 | `[ over dup sum swap len div gt? ] [ drop drop nil ] [ over over [ swap sub 0 max2 ] swap cons map sum [ drop ] dip range swap [ drop dup min over over [ eq? bool>int [ unpack2 ] dip add 1 min2 [ dup 0 eq? ] [ swap inc swap ] [ ] ifte pack2 ] swap cons [ 0 0 pack2 ] dip fold head swap inc set-at dup max over over [ eq? bool>int [ unpack2 ] dip add 1 min2 [ dup 0 eq? ] [ swap inc swap ] [ ] ifte pack2 ] swap cons [ 0 0 pack2 ] dip fold head swap dec set-at ] fold ] ifte` |
| codearc-455 | 723 | `dup len 3 mul range swap dup len dec 0 swap pack3 [ drop unpack3 [ over over lt? ] [ [ pick2 pick2 nth 0 lt? ] [ [ inc ] dip ] [ [ pick2 over nth 0 gt? ] [ dec ] [ pack3 dup unpack3 pack3 dup unpack3 pick2 swap nth set-at swap dup [ third ] dip unpack3 drop nth set-at swap unpack3 [ drop ] dip2 ] ifte ] ifte ] [ ] ifte pack3 ] fold head` |
| mbpp-907 | 745 | `dup [ sq 10 add inc 2 div range [ 2 mul dec ] map dup len range 2 drop-n swap [ [ over len lt? ] [ over swap nth [ dup len range swap zip ] dip [ swap head [ dup 0 eq? ] [ drop drop true ] [ swap mod 0 eq? not ] ifte ] swap cons filter seconds ] [ drop ] ifte ] fold ] dip inc slice-index take 1 drop-n` |
| mbpp-316 | 801 | `over len dec 0 swap -1 pack5 zeros16 dup cat swap [ drop [ dup third over fourth le? ] [ dup third over fourth add 2 div [ over head over at pick2 second eq? ] [ dup [ 4 swap set-at ] dip inc 2 swap set-at ] [ [ over head over at pick2 second gt? ] [ dec 3 swap set-at ] [ inc 2 swap set-at ] ifte ] ifte ] [ ] ifte ] fold fifth` |
| mbpp-627 | 1056 | `0 0 pack5 zeros16 dup cat swap [ drop [ dup fifth 0 eq? ] [ [ dup second over third gt? ] [ dup third inc 3 swap set-at 4 1 set-at ] [ [ dup head over second at over second eq? not ] [ dup second 3 swap set-at 4 1 set-at ] [ dup second over third add [ dup 0 lt? ] [ neg 2 div neg ] [ 2 div ] ifte [ over head over at over eq? ] [ inc 1 swap set-at ] [ 2 swap set-at ] ifte ] ifte ] ifte ] [ ] ifte ] fold fourth` |
| mbpp-714 | 1080 | `[ dup 1 le? ] [ 1 eq? not bool>int ] [ dup isqrt range1 1 drop-n over [ swap divides? ] swap cons filter [ prime? ] filter over [ over powers reverse tail swap [ swap divides? ] swap cons filter len dup 8 mul inc isqrt dec 2 div [ pow ] dip pack2 ] swap cons map dup [ head ] map product rot swap div 1 eq? not bool>int swap [ second ] map sum add ] ifte` |
| codearc-664 | 1180 | `[ over over gt? ] [ drop drop 0 quote ] [ dup decimal-digits [ drop 9 ] map 0 1 pack2 [ [ unpack2 ] dip over mul [ 2 mul ] dip rot 10 inc mul add swap pack2 ] fold head isqrt range1 [ range-between ] dip [ 0 quote ] dip [ [ dup ] dip [ [ dup 10 lt? ] [ dup ] [ [ dup 100 lt? ] [ dup 10 div 10 inc mul swap 10 mod 2 mul add dup ] [ decimal-digits dup 0 1 pack2 [ [ unpack2 ] dip over mul [ 2 mul ] dip rot 10 inc mul add swap pack2 ] fold head swap 0 0 0 pack3 [ [ unpack3 ] dip pick3 inc mul [ swap 10 mul ] dip add dup rot add [ inc ] dip2 pack3 ] fold third ] ifte ] ifte over over min2 [ [ eq? ] [ drop ] [ gcd ] ifte ] dip ] dip [ pick2 1 eq? ] [ drop drop drop 0 ] [ [ over eq? bool>int ] dip swap [ swap [ swap [ dup sq pick2 le? ] [ [ over over mod 0 eq? ] [ [ over over div eq? ] [ drop drop 1 ] [ drop drop 2 ] ifte ] [ drop drop 0 ] ifte ] [ drop drop 0 ] ifte ] swap cons map sum dec ] dip sub ] ifte [ pick2 head over lt? ] [ [ swap drop nil ] dip cons swap quote cat ] [ [ pick2 head over eq? ] [ drop quote cat ] [ drop drop ] ifte ] ifte ] swap cons fold ] ifte` |
| mbpp-392 | 1378 | `dup range1 1 drop-n nil 1 cons 0 cons [ dup 2 5 range-between swap [ swap div ] swap cons map pick2 [ swap [ dup 32 8 mul ge? ] [ nth ] [ [ dup 32 4 mul ge? ] [ [ tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail ] dip 32 4 mul sub ] [ ] ifte [ dup 32 dup add ge? ] [ [ tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail ] dip 32 dup add sub ] [ ] ifte [ dup 32 ge? ] [ [ tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail ] dip 32 sub ] [ ] ifte [ dup 16 ge? ] [ [ tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail tail ] dip 16 sub ] [ ] ifte [ dup 8 ge? ] [ [ tail tail tail tail tail tail tail tail ] dip 8 sub ] [ ] ifte [ dup 4 ge? ] [ [ tail tail tail tail ] dip 4 sub ] [ ] ifte [ dup 2 ge? ] [ [ tail tail ] dip 2 sub ] [ ] ifte [ dup 1 ge? ] [ [ tail ] dip 1 sub ] [ ] ifte drop head ] ifte ] swap cons map sum max2 quote cat ] fold swap at` |
| codearc-695 | 4063 | `swap 100 2 div 2 sq sq sq sq 2 div sq sq mul [ 100 [ dup 0 lt? ] [ neg [ neg ] dip ] [ ] ifte [ over 0 eq? ] [ drop ] [ [ over 0 lt? ] [ [ neg ] dip [ 2 sq sq sq sq 2 div sq sq mul ] dip over over div pows2 pows2-step pows2-step swap [ le? ] swap cons filter len 100 2 div 3 add sub pow2 dup [ mul ] dip [ over over div [ pick2 pick2 mod 2 mul pick2 gt? ] [ inc ] [ [ pick2 pick2 mod 2 mul pick2 eq? ] [ [ dup odd? ] [ inc ] [ ] ifte ] [ ] ifte ] ifte [ drop drop ] dip ] dip mul neg ] [ [ 2 sq sq sq sq 2 div sq sq mul ] dip over over div pows2 pows2-step pows2-step swap [ le? ] swap cons filter len 100 2 div 3 add sub pow2 dup [ mul ] dip [ over over div [ pick2 pick2 mod 2 mul pick2 gt? ] [ inc ] [ [ pick2 pick2 mod 2 mul pick2 eq? ] [ [ dup odd? ] [ inc ] [ ] ifte ] [ ] ifte ] ifte [ drop drop ] dip ] dip mul ] ifte ] ifte 2 sq sq sq sq 2 div sq sq swap sub 2 sq sq sq sq 2 div sq sq [ dup 0 lt? ] [ neg [ neg ] dip ] [ ] ifte [ over 0 eq? ] [ drop ] [ [ over 0 lt? ] [ [ neg ] dip [ 2 sq sq sq sq 2 div sq sq mul ] dip over over div pows2 pows2-step pows2-step swap [ le? ] swap cons filter len 100 2 div 3 add sub pow2 dup [ mul ] dip [ over over div [ pick2 pick2 mod 2 mul pick2 gt? ] [ inc ] [ [ pick2 pick2 mod 2 mul pick2 eq? ] [ [ dup odd? ] [ inc ] [ ] ifte ] [ ] ifte ] ifte [ drop drop ] dip ] dip mul neg ] [ [ 2 sq sq sq sq 2 div sq sq mul ] dip over over div pows2 pows2-step pows2-step swap [ le? ] swap cons filter len 100 2 div 3 add sub pow2 dup [ mul ] dip [ over over div [ pick2 pick2 mod 2 mul pick2 gt? ] [ inc ] [ [ pick2 pick2 mod 2 mul pick2 eq? ] [ [ dup odd? ] [ inc ] [ ] ifte ] [ ] ifte ] ifte [ drop drop ] dip ] dip mul ] ifte ] ifte [ dup 0 lt? ] [ neg [ neg ] dip ] [ ] ifte [ over 0 eq? ] [ drop ] [ [ over 0 lt? ] [ [ neg ] dip [ 2 sq sq sq sq 2 div sq sq mul ] dip over over div pows2 pows2-step pows2-step swap [ le? ] swap cons filter len 100 2 div 3 add sub pow2 dup [ mul ] dip [ over over div [ pick2 pick2 mod 2 mul pick2 gt? ] [ inc ] [ [ pick2 pick2 mod 2 mul pick2 eq? ] [ [ dup odd? ] [ inc ] [ ] ifte ] [ ] ifte ] ifte [ drop drop ] dip ] dip mul neg ] [ [ 2 sq sq sq sq 2 div sq sq mul ] dip over over div pows2 pows2-step pows2-step swap [ le? ] swap cons filter len 100 2 div 3 add sub pow2 dup [ mul ] dip [ over over div [ pick2 pick2 mod 2 mul pick2 gt? ] [ inc ] [ [ pick2 pick2 mod 2 mul pick2 eq? ] [ [ dup odd? ] [ inc ] [ ] ifte ] [ ] ifte ] ifte [ drop drop ] dip ] dip mul ] ifte ] ifte ] fold mul 2 sq sq sq sq 2 div sq sq [ dup 0 lt? ] [ neg [ neg ] dip ] [ ] ifte [ over 0 eq? ] [ drop ] [ [ over 0 lt? ] [ [ neg ] dip [ 2 sq sq sq sq 2 div sq sq mul ] dip over over div pows2 pows2-step pows2-step swap [ le? ] swap cons filter len 100 2 div 3 add sub pow2 dup [ mul ] dip [ over over div [ pick2 pick2 mod 2 mul pick2 gt? ] [ inc ] [ [ pick2 pick2 mod 2 mul pick2 eq? ] [ [ dup odd? ] [ inc ] [ ] ifte ] [ ] ifte ] ifte [ drop drop ] dip ] dip mul neg ] [ [ 2 sq sq sq sq 2 div sq sq mul ] dip over over div pows2 pows2-step pows2-step swap [ le? ] swap cons filter len 100 2 div 3 add sub pow2 dup [ mul ] dip [ over over div [ pick2 pick2 mod 2 mul pick2 gt? ] [ inc ] [ [ pick2 pick2 mod 2 mul pick2 eq? ] [ [ dup odd? ] [ inc ] [ ] ifte ] [ ] ifte ] ifte [ drop drop ] dip ] dip mul ] ifte ] ifte 2 sq sq sq sq 2 div sq sq ceil-div` |
