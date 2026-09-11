# world-authoring-500 (#127), wave 2: 100 MBPP / CodeARC behaviours authored by codex exec

Behaviours sampled by `experiments/world_behaviours/extract.py` (seed 17, MBPP remainder and CodeARC, excluding the probe and wave 1, hypothesis-breaking sampler on every source); programs authored by four `codex exec` agents (gpt-6-astra, high effort) in macro form and migrated to the canonical library; every program re-verified on every test by `report.py`.

## Findings (written with the 2026-09-11 wave 2; the tables below regenerate)

- **MBPP and CodeARC verify at 97%** (50/50 MBPP, 47/50 CodeARC; 96 as
  authored plus one identity behaviour whose author submitted the empty
  program, recorded as a two-op repair). The three misses are fuel: a
  breadth-first factorisation, a digit-window search and Langton's ant with a
  10,000-step preperiod. Yield on these two sources now stands at 96% over
  135 behaviours across the probe, wave 1 and wave 2.
- **Cheaper than wave 1.** 289k tokens and 23 agent-minutes for 100
  behaviours (59k–86k tokens per agent), 11 minutes wall; the shorter MBPP and
  CodeARC programs (macro-form median 15 and 21 tokens) need fewer
  verifier rounds than LeetCode-medium did.
- **The library gained 7 macros per 100 behaviours, down from 16.** Authors
  wrote 90 private macros; `popcount` was written by two authors, `prime?`,
  `bisect-right` and `next-palindrome` were each used by two programs, and
  `lcm`, `set-at` and a sign-correct `sxor` had each been written by an
  author in wave 1 as well, so all seven went canonical (110 → 117). The rest
  are one-behaviour step functions, inlined. No library defect this time.
- **Overlap with v4 stays negligible**: no exact witness match, one shared
  skeleton.

## Verified

| source | verified | median witness length (`mark-28` ops) | median macro-form length |
|---|---|---|---|
| codearc | 47/50 (94%) | 157 | 21 |
| mbpp | 50/50 (100%) | 115.0 | 15.0 |
| **all** | 97/100 (97%) | 144 | 17 |

Claimed-vs-verified mismatches: 0; missing ids: 0; duplicates: 0.

By author: canonical 97/100 (97%).

## Witness length against `stack_pbe_v4`

| bucket (ops) | this probe (verified) | v4 behaviours (train + eval, 220) |
|---|---|---|
| 1-5 | 2 | 15 |
| 6-10 | 3 | 14 |
| 11-20 | 7 | 34 |
| 21-40 | 10 | 37 |
| 41+ | 75 | 120 |
| median | 144 | 43.0 |

Exact matches with a v4 witness: 0 (none). Same skeleton as a v4 witness: 1.

## Macros

117 macros defined across `macros*.txt`; median expansion ratio (`mark-28` ops per macro-form token) on verified programs: 5.7.

| macro | verified programs using it |
|---|---|
| `pack2` | 21 |
| `2` | 21 |
| `inc` | 21 |
| `head` | 19 |
| `rot` | 19 |
| `range` | 17 |
| `unpack2` | 16 |
| `mod` | 14 |
| `pick2` | 14 |
| `len` | 14 |
| `max2` | 12 |
| `second` | 12 |
| `dec` | 11 |
| `le?` | 10 |
| `take` | 9 |
| `reverse` | 8 |
| `pack3` | 8 |
| `10` | 8 |
| `range1` | 8 |
| `ge?` | 7 |

## Unsolved

| id | source | passed | note |
|---|---|---|---|
| codearc-376 | codearc | 29/34 | 29/34; why: breadth-first nondecreasing factorizations exceed fuel on large products; includes reference bounds and orig |
| codearc-462 | codearc | 13/34 | why: fuel exhausted on larger ranges; current 13/34, best 14/34 before generalizing prefix-window validation; final-digi |
| codearc-721 | codearc | 5/34 | why: direct ant simulation with the reference periodic reduction passes 5/34; fuel is exhausted at n=500 and throughout  |

## Every verified program

| id | len | macro form |
|---|---|---|
| mbpp-587 | 2 | `dup drop` |
| mbpp-89 | 2 | `dec` |
| mbpp-664 | 6 | `2 div inc` |
| mbpp-581 | 7 | `2 mul over add mul` |
| mbpp-135 | 8 | `dup 2 mul dec mul` |
| mbpp-86 | 12 | `dup dec mul 3 mul inc` |
| codearc-130 | 13 | `[ even? ] filter` |
| codearc-893 | 13 | `[ lt? ] swap cons filter len` |
| mbpp-426 | 13 | `[ odd? ] filter` |
| codearc-587 | 16 | `abs neg` |
| mbpp-710 | 16 | `dup head swap last pack2` |
| mbpp-797 | 16 | `inc 2 div sq swap 2 div sq sub` |
| mbpp-609 | 21 | `over dec max2 swap [ mul ] dip div` |
| codearc-154 | 22 | `dup [ swap count 1 eq? ] swap cons filter` |
| codearc-7 | 22 | `dup [ swap count 1 eq? ] swap cons filter` |
| codearc-314 | 23 | `[ 2 mod 2 mul 1 swap sub ] map sum` |
| mbpp-968 | 23 | `over dec min2 rot mul swap div` |
| mbpp-329 | 24 | `[ 0 le? ] filter len` |
| mbpp-641 | 24 | `dup 7 mul 5 sub mul 2 div` |
| mbpp-221 | 28 | `[ even? ] filter [ null? ] [ drop -1 ] [ head ] ifte` |
| codearc-1109 | 35 | `dup rot swap sub [ sub ] dip [ neg ] dip div neg 1 max2` |
| codearc-718 | 35 | `[ swap contains 0 eq? ] swap cons filter` |
| codearc-738 | 45 | `dup min swap max over over swap sub pack3` |
| codearc-155 | 54 | `dedup product` |
| codearc-801 | 54 | `dedup sum` |
| mbpp-362 | 55 | `dup [ over count pack2 ] swap cons map uncons [ [ over second over second lt? ] [ swap drop ] [ drop ] ifte ] fold head` |
| mbpp-289 | 56 | `dup 4 div over 100 div sub over 100 4 mul div add add 7 mod` |
| mbpp-817 | 56 | `[ [ over swap divides? bool>int ] dip swap [ divides? bool>int ] dip add 0 gt? ] swap cons swap cons filter` |
| mbpp-141 | 57 | `sort` |
| mbpp-594 | 59 | `dup [ even? ] filter [ null? ] [ drop -1 ] [ head ] ifte swap [ odd? ] filter [ null? ] [ drop -1 ] [ head ] ifte sub` |
| codearc-512 | 60 | `inversions` |
| mbpp-895 | 67 | `uncons 0 swap pack2 [ [ unpack2 ] dip pick2 0 max2 add over max2 rot drop pack2 ] fold second` |
| mbpp-264 | 70 | `[ dup 2 le? ] [ 10 2 mul inc mul 2 div ] [ 2 sub 4 mul 10 2 mul inc add ] ifte` |
| mbpp-324 | 74 | `0 0 0 pack3 [ [ unpack3 ] dip [ pick3 odd? ] [ [ swap ] dip add swap ] [ add ] ifte [ inc ] dip2 pack3 ] fold tail` |
| mbpp-801 | 86 | `pack3 dedup len [ dup 3 eq? ] [ drop 0 ] [ 4 swap sub ] ifte` |
| codearc-1008 | 87 | `reverse uncons swap uncons rot over over swap sub [ max2 ] dip swap pack2 [ [ unpack2 ] dip over over max2 [ sub max2 ] dip pack2 ] fold head` |
| codearc-153 | 89 | `digit-sum` |
| codearc-580 | 96 | `0 0 pack2 quote [ [ dup ] dip [ [ unpack2 ] dip add [ inc ] dip pack2 ] swap cons map cat ] fold swap [ swap head eq? ] swap cons filter seconds swap [ le? ] swap cons filter max` |
| mbpp-479 | 105 | `[ dup 10 lt? ] [ ] [ digits head ] ifte` |
| codearc-11 | 106 | `dedup sort` |
| mbpp-831 | 107 | `take nil 0 pack2 [ [ unpack2 ] dip pick2 over count rot add [ cons ] dip pack2 ] fold second` |
| codearc-814 | 115 | `over sub inc range swap [ add ] swap cons map` |
| codearc-481 | 116 | `over swap [ rot dup rot add [ inc ] dip [ [ pick2 0 eq? ] [ drop drop drop false ] [ between ] ifte ] swap cons swap cons filter null? ] swap cons swap cons filter len` |
| codearc-677 | 116 | `[ dup 0 eq? ] [ drop 1 ] [ dec isqrt inc 2 div inc ] ifte` |
| codearc-1012 | 117 | `max-subarray-sum 0 max2` |
| mbpp-199 | 123 | `pows2 swap [ le? ] swap cons filter last` |
| codearc-142 | 125 | `uncons dup dup pack3 [ [ unpack3 ] dip [ pack2 ] dip swap [ mul ] pick2 cons map swap cons dup min swap max pick2 over max2 [ rot drop ] dip -rot pack3 ] fold head` |
| mbpp-583 | 131 | `range 1 [ dup 2 mul inc 2 mul rot mul swap 2 add div ] fold` |
| mbpp-57 | 144 | `[ sort reverse ] dip 1 max2 take from-digits` |
| codearc-472 | 145 | `[ dup 0 eq? ] [ drop 0 pack2 ] [ range swap dup pack2 [ drop unpack2 over add swap pack2 ] fold ] ifte` |
| codearc-544 | 145 | `dup [ [ dec ] dip pows2 -rot [ pick2 div [ swap div ] dip lt? ] swap cons swap cons filter last ] dip over div mul` |
| mbpp-520 | 145 | `uncons [ lcm ] fold` |
| mbpp-183 | 148 | `[ take ] dip abs [ [ [ unpack2 ] dip pick2 over [ sub abs ] swap cons map ] dip [ eq? ] swap cons filter len rot add [ cons ] dip pack2 ] swap cons nil 0 pack2 swap fold second` |
| mbpp-463 | 148 | `0 1 1 pack3 [ [ unpack3 ] dip [ dup 0 eq? ] [ drop drop drop 1 1 ] [ [ dup 0 gt? ] [ dup [ mul ] dip [ swap ] dip mul swap [ 1 min2 ] dip ] [ dup [ mul ] dip [ swap ] dip mul swap swap 1 max2 ] ifte ] ifte pick2 over max2 [ rot drop ] dip -rot pack3 ] fold head` |
| mbpp-385 | 152 | `range 3 0 2 pack3 [ drop unpack3 rot pick2 add pack3 ] fold head` |
| codearc-594 | 157 | `0 pack2 zeros16 swap [ drop unpack2 [ over 10 lt? ] [ ] [ inc [ digit-product ] dip ] ifte pack2 ] fold second` |
| codearc-715 | 162 | `[ dup pows2 swap [ swap div 2 mod ] swap cons map 0 [ swap 2 mul add ] fold swap pack2 ] map uncons [ [ over head over head lt? ] [ swap drop ] [ drop ] ifte ] fold second` |
| mbpp-620 | 162 | `take reverse nil [ over over [ [ unpack2 drop ] dip [ over over mod 0 eq? ] [ drop drop true ] [ swap mod 0 eq? ] ifte ] swap cons filter seconds 0 [ max2 ] fold inc pack2 cons ] fold seconds max` |
| mbpp-169 | 171 | `[ dup 2 le? ] [ ] [ 2 sub range 1 2 pack2 [ drop unpack2 dup 2 mul rot add pack2 ] fold second ] ifte` |
| codearc-622 | 178 | `[ dup 0 le? ] [ drop drop -1 ] [ [ abs ] dip dec range swap [ drop 10 div ] fold 10 mod ] ifte` |
| codearc-198 | 185 | `bisect-right` |
| mbpp-786 | 185 | `bisect-right` |
| mbpp-344 | 192 | `isqrt swap dec isqrt sub` |
| mbpp-911 | 203 | `sort dup reverse 3 take product swap dup last swap 2 take product mul max2` |
| mbpp-189 | 208 | `take dup len inc range1 swap [ swap contains 0 eq? ] swap cons filter head` |
| codearc-119 | 225 | `[ drop ] dip range swap [ drop ] swap quote cat map flatten max-subarray-sum` |
| mbpp-194 | 225 | `[ dup 0 lt? ] [ abs digits [ neg 10 mod ] map ] [ digits ] ifte 0 [ swap 8 mul add ] fold` |
| codearc-629 | 238 | `[ dup 0 eq? ] [ drop ] [ [ dup 0 lt? ] [ gcd neg ] [ gcd ] ifte ] ifte` |
| mbpp-501 | 238 | `gcd dup range1 swap [ swap divides? ] swap cons filter len` |
| codearc-221 | 239 | `dup len range zip [ unpack2 mul ] map 1 drop-n` |
| codearc-74 | 251 | `[ drop drop prime? ] [ drop swap drop ] [ swap drop swap drop ] ifte` |
| codearc-565 | 260 | `[ dup 0 lt? ] [ [ reverse ] dip 3 swap sub ] [ ] ifte [ dup 4 lt? ] [ nth ] [ 3 sub range swap [ drop dup tail swap unpack4 6 mul swap 10 mul sub swap 6 mul add swap sub quote cat ] fold last ] ifte` |
| mbpp-736 | 270 | `over len dup range [ 0 swap pack4 ] dip swap [ drop unpack4 [ lt? ] [ over over add 2 div [ pick4 over nth pick4 lt? ] [ inc [ swap drop ] dip swap ] [ swap drop ] ifte ] [ ] ifte pack4 ] fold third` |
| mbpp-671 | 273 | `[ [ dup 0 gt? ] [ inc pows2 swap contains 1 eq? ] [ -1 eq? ] ifte ] [ ] [ dup pows2 swap [ swap div 2 mod 0 eq? ] swap cons filter head add ] ifte` |
| codearc-507 | 276 | `dup bitlen range swap 0 pack2 [ drop unpack2 over 2 mod swap 2 mul add swap 2 div swap pack2 ] fold second` |
| codearc-391 | 285 | `[ dup 0 le? ] [ drop 0 ] [ dup log2 swap popcount add ] ifte` |
| codearc-937 | 285 | `[ dup popcount swap pack2 ] map nil [ over over [ [ over head over head eq? ] [ [ second ] dip second lt? ] [ [ head ] dip head lt? ] ifte not ] swap cons filter -rot swap over [ [ over head over head eq? ] [ [ second ] dip second lt? ] [ [ head ] dip head lt? ] ifte ] swap cons filter -rot quote swap cat cat ] fold seconds` |
| mbpp-402 | 286 | `[ over 0 eq? ] [ drop drop drop 1 ] [ [ [ over over lt? ] [ drop drop 0 ] [ dup range1 product [ dup range1 rot rot sub [ add ] swap cons map product ] dip div ] ifte ] dip mod ] ifte` |
| codearc-287 | 298 | `[ dup 9 gt? ] [ popcount ] [ ] ifte [ dup 9 gt? ] [ popcount ] [ ] ifte` |
| codearc-72 | 302 | `[ dup [ dup 0 lt? ] [ abs digits uncons neg swap sum add ] [ digit-sum ] ifte swap pack2 ] map nil [ over over [ [ head ] dip head gt? ] swap cons filter -rot swap over [ [ head ] dip head gt? not ] swap cons filter -rot quote swap cat cat ] fold seconds` |
| codearc-233 | 317 | `[ sort prefix-sums dup len range zip ] dip [ [ head ] dip ge? ] swap cons filter head second` |
| codearc-473 | 333 | `dup len over dup max swap min sub 2 add mul range swap 0 pack2 [ drop unpack2 [ over dup min swap max eq? ] [ ] [ [ [ inc 2 div ] map dup last over swap cons zip [ sum ] map ] dip inc ] ifte pack2 ] fold unpack2 swap last pack2` |
| mbpp-28 | 364 | `[ lt? ] [ drop drop 0 ] [ over over sub range1 product [ range1 product ] dip mul [ range1 product ] dip div ] ifte` |
| mbpp-849 | 367 | `dup range1 [ prime? ] filter swap [ swap mod 0 eq? ] swap cons filter sum` |
| codearc-954 | 408 | `[ dup 1 le? ] [ 1 eq? bool>int 2 mul dec ] [ dup isqrt range1 1 drop-n swap 1 pack2 [ [ unpack2 ] dip [ pick2 over sq divides? ] [ drop drop 0 pack2 ] [ [ pick2 over divides? ] [ rot swap div swap neg pack2 ] [ drop pack2 ] ifte ] ifte ] fold unpack2 [ over 1 eq? ] [ swap drop ] [ swap drop neg ] ifte ] ifte` |
| codearc-381 | 414 | `1 10 inc pack3 zeros16 swap [ drop unpack3 [ pick2 over ge? ] [ [ inc ] dip over dup inc swap 10 swap pow mul 9 mul add ] [ ] ifte pack3 ] fold unpack3 rot swap sub over over swap mod pick2 dec swap sub 10 swap pow rot mul div 10 mod` |
| codearc-810 | 445 | `[ dup 0 lt? ] [ [ over len neg lt? ] [ drop drop -1 ] [ dup [ at ] dip neg pow 1 swap div ] ifte ] [ [ over len ge? ] [ drop drop -1 ] [ dup [ nth ] dip pow ] ifte ] ifte` |
| mbpp-325 | 451 | `[ dup 3 le? ] [ ] [ range1 nil 0 cons [ over swap dup isqrt range1 -rot [ rot sq sub nth inc ] swap cons swap cons map min quote cat ] fold last ] ifte` |
| codearc-1053 | 501 | `dup 0 max2 8 sq inc 2 mul div isqrt inc 4 mul range [ dup 4 div 8 sq inc mul swap 4 mod nil 8 sq inc 4 sub cons 7 8 mul cons 9 cons 4 cons swap nth add ] map prefix-sums uncons rot [ [ pick2 over sub abs pick2 pick2 sub abs ge? ] [ drop swap drop ] [ drop drop ] ifte ] swap cons fold` |
| codearc-407 | 527 | `[ dup 0 ge? bool>int over 3 lt? bool>int mul 1 eq? ] [ over over at [ 3 range swap [ eq? not ] swap cons filter swap [ swap nth ] swap cons map sort ] dip quote cat ] [ drop sort ] ifte [ dup head 0 eq? bool>int over third 0 eq? bool>int mul 1 eq? ] [ drop -1 ] [ dup second over head sub 3 mod 0 eq? [ ] [ drop second ] [ drop drop -1 ] ifte ] ifte` |
| mbpp-245 | 541 | `over over over over drop-n [ take ] dip swap nil [ lis-best over add pack2 cons ] fold reverse seconds swap cat [ over reverse swap over over drop-n [ take ] dip swap nil [ lis-best over add pack2 cons ] fold reverse seconds swap cat reverse ] dip zip swap zip [ unpack2 swap unpack2 add swap sub ] map max` |
| mbpp-119 | 736 | `take 0 [ sxor ] fold` |
| codearc-172 | 786 | `take nil 0 pack2 [ [ unpack2 ] dip pick2 over [ sxor ] swap cons map sum rot add [ cons ] dip pack2 ] fold second` |
| codearc-289 | 917 | `next-palindrome` |
| mbpp-100 | 917 | `next-palindrome` |
| mbpp-209 | 1741 | `[ dup len 2 div range reverse swap [ pack2 zeros16 swap [ drop unpack2 dup 2 mul inc [ dup pick3 len lt? ] [ [ dup inc pick3 len lt? ] [ [ pick2 over nth pick3 pick2 inc nth ge? ] [ inc ] [ ] ifte ] [ ] ifte [ pick2 over nth pick3 pick3 nth lt? ] [ dup [ pick2 len range [ pack3 ] dip swap unpack3 [ [ pick3 pick2 eq? ] [ [ drop ] dip nth swap drop ] [ [ pick3 over eq? ] [ drop nth swap drop ] [ drop drop swap nth ] ifte ] ifte ] swap cons swap cons swap cons map ] dip ] [ drop ] ifte ] [ drop ] ifte pack2 ] fold head ] fold ] dip 0 swap set-at 0 pack2 zeros16 swap [ drop unpack2 dup 2 mul inc [ dup pick3 len lt? ] [ [ dup inc pick3 len lt? ] [ [ pick2 over nth pick3 pick2 inc nth ge? ] [ inc ] [ ] ifte ] [ ] ifte [ pick2 over nth pick3 pick3 nth lt? ] [ dup [ pick2 len range [ pack3 ] dip swap unpack3 [ [ pick3 pick2 eq? ] [ [ drop ] dip nth swap drop ] [ [ pick3 over eq? ] [ drop nth swap drop ] [ drop drop swap nth ] ifte ] ifte ] swap cons swap cons swap cons map ] dip ] [ drop ] ifte ] [ drop ] ifte pack2 ] fold head` |
| codearc-585 | 1866 | `dup 0 max2 decimal-digits len 10 swap pow isqrt range1 tail [ dup range1 tail swap [ swap mod 0 eq? ] swap cons filter len 1 eq? ] filter [ over sub inc range swap [ add ] swap cons map [ 2 ge? ] filter ] dip dup [ swap [ [ [ over eq? ] [ drop drop true ] [ mod 0 eq? not ] ifte ] swap cons filter ] fold ] dip reverse [ over 0 [ over 0 eq? ] [ ] [ swap dup 10 mod rot 10 mul add swap 10 div swap ] ifte [ over 0 eq? ] [ ] [ swap dup 10 mod rot 10 mul add swap 10 div swap ] ifte [ over 0 eq? ] [ ] [ swap dup 10 mod rot 10 mul add swap 10 div swap ] ifte [ over 0 eq? ] [ ] [ swap dup 10 mod rot 10 mul add swap 10 div swap ] ifte [ over 0 eq? ] [ ] [ swap dup 10 mod rot 10 mul add swap 10 div swap ] ifte [ over 0 eq? ] [ ] [ swap dup 10 mod rot 10 mul add swap 10 div swap ] ifte [ over 0 eq? ] [ ] [ swap dup 10 mod rot 10 mul add swap 10 div swap ] ifte [ over 0 eq? ] [ ] [ swap dup 10 mod rot 10 mul add swap 10 div swap ] ifte swap drop [ over [ over over head le? ] [ swap contains 1 eq? ] [ swap [ swap mod 0 eq? ] swap cons filter null? ] ifte ] [ swap drop eq? not ] [ drop drop drop false ] ifte ] swap cons filter dup dup [ over 0 [ over 0 eq? ] [ ] [ swap dup 10 mod rot 10 mul add swap 10 div swap ] ifte [ over 0 eq? ] [ ] [ swap dup 10 mod rot 10 mul add swap 10 div swap ] ifte [ over 0 eq? ] [ ] [ swap dup 10 mod rot 10 mul add swap 10 div swap ] ifte [ over 0 eq? ] [ ] [ swap dup 10 mod rot 10 mul add swap 10 div swap ] ifte [ over 0 eq? ] [ ] [ swap dup 10 mod rot 10 mul add swap 10 div swap ] ifte [ over 0 eq? ] [ ] [ swap dup 10 mod rot 10 mul add swap 10 div swap ] ifte [ over 0 eq? ] [ ] [ swap dup 10 mod rot 10 mul add swap 10 div swap ] ifte [ over 0 eq? ] [ ] [ swap dup 10 mod rot 10 mul add swap 10 div swap ] ifte swap drop [ pick2 lt? ] [ rot drop contains 1 eq? ] [ drop drop drop false ] ifte ] swap cons filter cat sort` |
