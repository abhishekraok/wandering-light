# world-authoring-20260911-w5 raw (author dialects)

raw author outputs before the merge pass


## Verified

| source | verified | median witness length (`mark-28` ops) | median macro-form length |
|---|---|---|---|
| codearc | 49/50 (98%) | 158 | 15 |
| mbpp | 49/50 (98%) | 109 | 11 |
| **all** | 98/100 (98%) | 128.0 | 12.5 |

Claimed-vs-verified mismatches: 0; missing ids: 0; duplicates: 0.

By author: a 26/26 (100%); b 25/26 (96%); c 24/24 (100%); d 23/24 (96%).

## Witness length against `stack_pbe_v4`

| bucket (ops) | this probe (verified) | v4 behaviours (train + eval, 220) |
|---|---|---|
| 1-5 | 6 | 15 |
| 6-10 | 3 | 14 |
| 11-20 | 8 | 34 |
| 21-40 | 6 | 37 |
| 41+ | 75 | 120 |
| median | 128.0 | 43.0 |

Exact matches with a v4 witness: 7 (mbpp-62, mbpp-168, mbpp-750, codearc-9, mbpp-865, mbpp-798, codearc-103). Same skeleton as a v4 witness: 7.

## Macros

188 macros defined across `macros*.txt`; median expansion ratio (`mark-28` ops per macro-form token) on verified programs: 10.6.

| macro | verified programs using it |
|---|---|
| `len` | 18 |
| `sum` | 15 |
| `2` | 15 |
| `head` | 14 |
| `take` | 13 |
| `range` | 13 |
| `unpack2` | 12 |
| `inc` | 11 |
| `drop-n` | 9 |
| `range1` | 9 |
| `mod` | 9 |
| `odd?` | 9 |
| `dec` | 8 |
| `max2` | 8 |
| `pack2` | 8 |
| `reverse` | 8 |
| `even?` | 7 |
| `isqrt` | 7 |
| `gt?` | 7 |
| `sort` | 6 |

## Unsolved

| id | source | passed | note |
|---|---|---|---|
| mbpp-193 | mbpp | 23/27 | why: Python set iteration requires hash-table probing and resize order; sorted deduplication passes 23/27, failing four  |
| codearc-991 | codearc | 11/34 | why: odd-candidate sieve and prime-index lookup exceed the 100000-step fuel limit on larger bounds; best candidate retai |

## Every verified program

| id | len | macro form |
|---|---|---|
| mbpp-750 | 1 | `cat` |
| mbpp-499 | 2 | `dup add` |
| mbpp-106 | 2 | `swap cat` |
| mbpp-234 | 4 | `dup sq mul` |
| mbpp-502 | 5 | `mod` |
| mbpp-798 | 5 | `sum` |
| codearc-9 | 6 | `[ 0 gt? ] filter` |
| mbpp-865 | 6 | `[ mul ] swap cons map` |
| mbpp-436 | 8 | `[ 0 lt? ] filter head` |
| mbpp-168 | 13 | `count` |
| codearc-144 | 13 | `[ product ] dip mod` |
| codearc-148 | 13 | `[ odd? ] filter` |
| codearc-103 | 13 | `count` |
| mbpp-339 | 15 | `[ eq? ] [ swap drop ] [ drop drop 2 ] ifte` |
| mbpp-62 | 16 | `min` |
| codearc-203 | 16 | `inc 2 div sq swap 2 div sq sub` |
| codearc-425 | 16 | `2 div over sub dup [ sub ] dip pack2` |
| mbpp-378 | 23 | `[ null? ] [ ] [ reverse uncons [ reverse ] dip cons ] ifte` |
| codearc-799 | 27 | `pack3 [ 3 mul 4 div ] map product` |
| codearc-458 | 27 | `dup [ swap count 1 eq? ] swap cons filter sum` |
| mbpp-162 | 28 | `0 max2 dup inc 2 div over over sub inc mul swap drop` |
| mbpp-161 | 35 | `[ swap contains 0 eq? ] swap cons filter` |
| codearc-365 | 35 | `[ uncons quote ] dip [ kc-consecutive-step ] swap cons fold reverse` |
| mbpp-881 | 45 | `dup [ even? ] filter head-or-neg1 swap [ odd? ] filter head-or-neg1 add` |
| mbpp-784 | 45 | `kb-first-even-odd mul` |
| codearc-159 | 45 | `kb-first-even-odd sub` |
| mbpp-510 | 50 | `swap [ 0 gt? ] filter nil 1 cons [ over swap [ mul ] swap cons map cat ] fold tail swap [ le? ] swap cons filter len` |
| codearc-173 | 57 | `sort` |
| codearc-45 | 58 | `[ kc-unit-digit ] dip kc-unit-digit mul` |
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
| mbpp-166 | 109 | `take dup [ even? ] filter len ka-choose2 swap [ odd? ] filter len ka-choose2 add` |
| codearc-293 | 115 | `over sub inc range swap [ add ] swap cons map` |
| codearc-447 | 118 | `[ dup dec mul ] dip 2 mul add 4 mul isqrt inc 2 div` |
| codearc-125 | 123 | `pows2 swap [ le? ] swap cons filter last` |
| mbpp-388 | 123 | `kb-highest-power` |
| codearc-177 | 127 | `adjacent [ product ] map` |
| codearc-1073 | 127 | `pack2 zeros16 dup cat swap [ drop unpack2 [ over over mul 0 eq? ] [ ] [ [ over over 2 mul ge? ] [ dup [ 2 mul mod ] dip ] [ [ over over swap 2 mul ge? ] [ over 2 mul mod ] [ ] ifte ] ifte ] ifte pack2 ] fold` |
| mbpp-470 | 128 | `adjacent [ unpack2 add ] map` |
| mbpp-726 | 128 | `adjacent [ unpack2 mul ] map` |
| codearc-574 | 131 | `[ range1 ] dip [ swap [ add ] dip mod ] swap cons 0 swap fold inc` |
| mbpp-302 | 135 | `[ dup 0 lt? ] [ drop 1 ] [ kb-highest-power ] ifte` |
| codearc-922 | 149 | `differences [ abs ] map sum` |
| codearc-73 | 158 | `[ 10 gt? ] filter [ dup odd? bool>int swap digits head odd? bool>int mul 1 eq? ] filter len` |
| mbpp-453 | 172 | `kc-even-factor-sum` |
| codearc-51 | 172 | `[ decimal-digits [ even? ] filter null? ] filter sort` |
| mbpp-25 | 175 | `[ sort ] dip over last [ take ] dip 1 pack2 [ [ unpack2 ] dip [ pick2 eq? ] [ drop pack2 ] [ rot drop swap over mul pack2 ] ifte ] fold second` |
| codearc-570 | 184 | `[ null? ] [ drop 0 1 pack2 ] [ [ dup len 1 eq? ] [ head 1 ] [ unpack2 ] ifte [ dup 0 lt? ] [ neg [ neg ] dip ] [ ] ifte over over gcd [ pack2 ] dip [ div ] swap cons map ] ifte` |
| mbpp-754 | 191 | `[ zip ] dip zip [ unpack2 swap unpack2 over eq? bool>int [ eq? bool>int ] dip mul 1 eq? ] filter [ head head ] map` |
| mbpp-170 | 199 | `over sub inc range swap [ add ] swap cons map swap [ swap at ] swap cons map sum` |
| codearc-983 | 212 | `kc-enumerate [ second odd? ] filter [ head ] map head-or-neg1` |
| codearc-295 | 213 | `dup len range zip [ unpack2 [ odd? ] [ drop sq ] [ drop ] ifte ] map sum` |
| codearc-778 | 217 | `dup dedup swap [ swap dup [ count ] dip swap pack2 ] swap cons map dup seconds dup min swap max [ over eq? ] [ drop drop drop nil ] [ swap drop [ [ second ] dip eq? ] swap cons filter [ head ] map sort ] ifte` |
| codearc-606 | 222 | `decimal-digits [ sq decimal-digits ] map flatten from-digits` |
| codearc-113 | 224 | `kc-enumerate [ head even? ] filter seconds [ even? ] filter sum` |
| codearc-393 | 226 | `range [ drop 0 ] map [ over min dup [ add ] dip swap [ remove1 ] dip cons ] fold max` |
| codearc-770 | 229 | `kc-optimal-removal` |
| mbpp-908 | 253 | `take kc-enumerate [ unpack2 eq? ] filter [ head ] map head-or-neg1` |
| codearc-896 | 261 | `differences 1 0 0 1 pack4 [ kd-garden-step ] fold unpack4 max2 [ drop drop ] dip` |
| codearc-797 | 269 | `dup 8 mul inc isqrt dec 2 div range1 swap [ swap dup [ mod ] dip [ dup odd? ] [ drop 0 eq? ] [ 2 div eq? ] ifte ] swap cons filter len` |
| codearc-620 | 272 | `dup len range swap [ swap drop-n ka-prefix-area ] swap cons map 0 [ max2 ] fold` |
| codearc-138 | 272 | `[ head 0 lt? ] [ [ abs decimal-digits ] map flatten from-digits neg ] [ [ decimal-digits ] map flatten from-digits ] ifte` |
| codearc-1111 | 273 | `dup [ slice-index take reverse ] dip 3 sub range swap [ drop dup 3 take sum cons ] fold reverse` |
| mbpp-763 | 288 | `[ sort ] dip take differences [ neg ] map kc-ten-to-twenty [ min2 ] fold` |
| mbpp-681 | 298 | `[ even? ] [ drop 2 ] [ dup isqrt range1 2 drop-n over [ swap divides? ] swap cons filter swap quote cat head ] ifte` |
| mbpp-84 | 308 | `range1 2 drop-n nil 1 cons 1 cons 0 cons [ kd-newman-step ] fold last` |
| codearc-1024 | 315 | `[ over len over ge? ] [ slice-index take ] [ over len over over sub range [ swap drop ] dip rot reverse rot [ swap drop over swap take sum cons ] swap cons fold reverse ] ifte` |
| mbpp-852 | 324 | `dup len range swap [ [ over len lt? ] [ [ over over nth 0 lt? ] [ over swap nth remove1 ] [ drop ] ifte ] [ drop ] ifte ] fold` |
| mbpp-545 | 325 | `[ dup 1 eq? ] [ drop 0 ] [ [ dup 0 gt? ] [ dup log2 pow2 sub ] [ ] ifte [ dup even? ] [ inc ] [ dec ] ifte ] ifte` |
| codearc-776 | 330 | `sort reverse dup len 2 sub range swap [ swap drop-n 3 take ] swap cons map [ dup head swap tail sum lt? ] filter [ sum ] map 0 [ max2 ] fold` |
| codearc-219 | 345 | `[ decimal-digits ] map flatten nil [ ka-rewrite-digit ] fold reverse` |
| mbpp-323 | 364 | `over over drop-n [ 0 max2 take ka-alternate ] dip cat` |
| mbpp-511 | 386 | `[ dup 1 le? ] [ ] [ dup isqrt range1 tail swap 0 pack2 [ kd-factor-step ] fold sum ] ifte` |
| mbpp-724 | 404 | `over abs decimal-digits len over mul inc [ pow ] dip kb-decimal-sum` |
| mbpp-371 | 415 | `pack3 zeros16 dup cat swap [ drop ka-missing-step ] fold second` |
| codearc-355 | 421 | `[ dup 0 lt? ] [ abs kb-signed-bits -1 cons ] [ kb-signed-bits ] ifte 1 cons` |
| codearc-1087 | 460 | `dup [ inc range 3 drop-n swap [ kc-sequence-step ] fold ] dip at` |
| codearc-452 | 486 | `over abs over abs max2 dup isqrt range1 [ odd? ] filter [ ka-cube ] map swap [ le? ] swap cons filter [ ka-odd-digits? ] filter dup reverse [ neg ] map swap cat -rot [ between ] swap cons swap cons filter` |
| mbpp-383 | 569 | `kc-toggle-even-bits` |
| codearc-337 | 582 | `[ over dup sum swap len div gt? ] [ drop drop nil ] [ over over [ swap sub 0 max2 ] swap cons map sum [ drop ] dip range swap [ drop kb-inc-poorest kb-dec-richest ] fold ] ifte` |
| codearc-455 | 723 | `dup len 3 mul range swap dup len dec 0 swap pack3 [ drop kb-partition-step ] fold head` |
| mbpp-907 | 745 | `dup [ sq 10 add inc 2 div range [ 2 mul dec ] map dup len range 2 drop-n swap [ [ over len lt? ] [ over swap nth kb-lucky-sieve ] [ drop ] ifte ] fold ] dip inc slice-index take 1 drop-n` |
| mbpp-316 | 801 | `over len dec 0 swap -1 pack5 zeros16 dup cat swap [ drop ka-last-search-step ] fold fifth` |
| mbpp-627 | 1056 | `0 0 pack5 zeros16 dup cat swap [ drop ka-first-missing-step ] fold fourth` |
| mbpp-714 | 1080 | `[ dup 1 le? ] [ 1 eq? not bool>int ] [ dup isqrt range1 1 drop-n over [ swap divides? ] swap cons filter [ prime? ] filter over [ ka-factor-info ] swap cons map dup [ head ] map product rot swap div 1 eq? not bool>int swap [ second ] map sum add ] ifte` |
| codearc-664 | 1180 | `[ over over gt? ] [ drop drop 0 quote ] [ dup decimal-digits [ drop 9 ] map kb-subsequence-score isqrt range1 [ range-between ] dip [ 0 quote ] dip [ kb-best-score-list ] swap cons fold ] ifte` |
| mbpp-392 | 1378 | `dup range1 1 drop-n nil 1 cons 0 cons [ kd-exchange-step ] fold swap at` |
| codearc-695 | 4063 | `swap 100 2 div kd-2p60 mul [ kd-scrap-step ] fold mul kd-2p60 kd-float-ratio kd-2p60 ceil-div` |
