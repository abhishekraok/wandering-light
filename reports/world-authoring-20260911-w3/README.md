# world-authoring-500 (#127), wave 3: 100 MBPP / CodeARC behaviours authored by codex exec

Behaviours sampled by `experiments/world_behaviours/extract.py` (seed 17, MBPP remainder and CodeARC, excluding the probe and waves 1–2, hypothesis-breaking sampler on every source); programs authored by four `codex exec` agents (gpt-6-astra, high effort) in macro form and migrated to the canonical library; every program re-verified on every test by `report.py`.

## Findings (written with the 2026-09-11 wave 3; the tables below regenerate)

- **99 of 100 verified** (50/50 MBPP, 49/50 CodeARC); the miss is fuel
  (divisor sums of factor pairs up to a limit of 100). Claimed and verified
  statuses agreed on all 100. Over the probe and three waves, MBPP stands at
  184/186 and CodeARC at 130/134.
- **Cost keeps falling:** 277k tokens, 24 agent-minutes, 6 minutes wall for
  100 behaviours (61k–77k tokens per agent; no agent above 6 minutes).
- **Library: 8 macros per 100** (117 → 125). Three authors across waves had
  written `factorial`; `binomial` was used by three programs and
  `primes-below` by two; `slice-index` (Python's negative index, clamped),
  `head-or-neg1` and the sort-by-key trio (`key-gt?`, `insert-by-head`,
  `sort-by-head`) had each been written by an author in an earlier wave as
  well. Two authors wrote the same inclusive-range helper (`over sub inc range
  swap [ add ] swap cons map`), which differs from the canonical
  `range-between` on negative bounds; it stays inlined for now and is the
  candidate replacement when the library is next re-expanded for a corpus
  build. Aliasing it to `range-between` cost two witnesses until re-checked,
  which is what the migration re-verification is for.
- **Overlap with v4**: two exact witness matches (`codearc-22`, `mbpp-972`,
  both scalar arithmetic) and four shared skeletons.

## Verified

| source | verified | median witness length (`mark-28` ops) | median macro-form length |
|---|---|---|---|
| codearc | 49/50 (98%) | 221 | 34 |
| mbpp | 50/50 (100%) | 121.5 | 13.5 |
| **all** | 99/100 (99%) | 151 | 20 |

Claimed-vs-verified mismatches: 0; missing ids: 0; duplicates: 0.

By author: canonical 99/100 (99%).

## Witness length against `stack_pbe_v4`

| bucket (ops) | this probe (verified) | v4 behaviours (train + eval, 220) |
|---|---|---|
| 1-5 | 7 | 15 |
| 6-10 | 3 | 14 |
| 11-20 | 5 | 34 |
| 21-40 | 6 | 37 |
| 41+ | 78 | 120 |
| median | 151 | 43.0 |

Exact matches with a v4 witness: 2 (codearc-22, mbpp-972). Same skeleton as a v4 witness: 4.

## Macros

133 macros defined across `macros*.txt`; median expansion ratio (`mark-28` ops per macro-form token) on verified programs: 5.1.

| macro | verified programs using it |
|---|---|
| `2` | 27 |
| `inc` | 27 |
| `range` | 26 |
| `head` | 24 |
| `unpack2` | 23 |
| `len` | 21 |
| `second` | 21 |
| `pack2` | 21 |
| `pick2` | 20 |
| `dec` | 18 |
| `sum` | 18 |
| `mod` | 17 |
| `take` | 15 |
| `rot` | 15 |
| `pack3` | 14 |
| `le?` | 13 |
| `3` | 11 |
| `gt?` | 11 |
| `pick3` | 11 |
| `third` | 11 |

## Unsolved

| id | source | passed | note |
|---|---|---|---|
| codearc-92 | codearc | 2/16 | why: factor-pair divisor sums exceed 100,000 fuel, already at limit=100; best 2/16 (nonpositive limits). Includes both m |

## Every verified program

| id | len | macro form |
|---|---|---|
| codearc-22 | 1 | `add` |
| codearc-522 | 1 | `mul` |
| mbpp-458 | 1 | `mul` |
| mbpp-972 | 1 | `cat` |
| codearc-1062 | 4 | `2 div` |
| mbpp-112 | 5 | `add 2 mul` |
| mbpp-882 | 5 | `mul 2 mul` |
| codearc-139 | 7 | `[ dup sq mul ] map` |
| mbpp-354 | 7 | `[ dec ] dip mul add` |
| mbpp-205 | 8 | `[ neg dec ] map` |
| mbpp-555 | 12 | `dup inc mul 2 div dup dec mul` |
| mbpp-59 | 12 | `dup 3 mul 2 sub mul` |
| codearc-985 | 13 | `dup product [ swap div ] swap cons map` |
| mbpp-626 | 16 | `[ dup 0 lt? ] [ drop -1 ] [ sq ] ifte` |
| mbpp-697 | 20 | `[ even? ] filter len` |
| mbpp-287 | 22 | `dup inc over 2 mul inc mul mul 2 mul 3 div` |
| codearc-332 | 27 | `over over sub abs inc [ add ] dip mul 2 div` |
| codearc-1105 | 28 | `[ sum ] dip2 over over add [ swap sub inc ] dip mul mul 2 div` |
| mbpp-356 | 32 | `add 100 8 10 mul add swap sub` |
| codearc-671 | 33 | `over over mod [ [ neg ] dip mod ] dip min2` |
| mbpp-665 | 40 | `dup head over over [ eq? not ] swap cons filter -rot [ eq? ] swap cons filter cat` |
| codearc-180 | 43 | `reverse 0 0 pack2 [ [ unpack2 ] dip swap inc mul dup rot add swap pack2 ] fold head` |
| mbpp-347 | 43 | `[ over over gt? ] [ swap ] [ ] ifte over 3 mul over sub inc rot drop swap dup inc mul mul 6 div` |
| codearc-290 | 56 | `[ dup len 2 lt? ] [ drop -1 ] [ dup len over max mul swap sum sub ] ifte` |
| mbpp-940 | 57 | `sort` |
| codearc-121 | 60 | `inversions` |
| codearc-1046 | 68 | `dec dup 3 sum-multiples over 5 sum-multiples add swap 3 5 mul sum-multiples sub` |
| codearc-477 | 69 | `abs [ over over mod 2 mul over eq? ] [ drop ] [ over over mod 2 mul over gt? bool>int [ over over div ] dip add mul swap drop ] ifte` |
| mbpp-379 | 71 | `2 pick3 pick3 mul pick2 [ pick4 ] dip swap [ pick4 ] dip swap add mul add mul [ drop drop drop ] dip` |
| mbpp-282 | 76 | `zip [ unpack2 sub ] map` |
| mbpp-825 | 79 | `swap [ swap at ] swap cons map` |
| mbpp-351 | 87 | `[ take dup ] dip [ [ swap count ] dip eq? ] swap cons swap cons filter head-or-neg1` |
| mbpp-384 | 94 | `swap 1 max2 take dup min count` |
| mbpp-558 | 103 | `sub abs digit-sum` |
| mbpp-687 | 107 | `gcd` |
| mbpp-496 | 110 | `[ sort ] dip take` |
| codearc-166 | 112 | `reverse nil [ over over [ [ unpack2 drop ] dip [ over over mod 0 eq? ] [ drop drop true ] [ swap mod 0 eq? ] ifte ] swap cons filter seconds 0 [ max2 ] fold inc pack2 cons ] fold seconds max` |
| mbpp-22 | 114 | `nil nil pack2 [ [ unpack2 ] dip [ over null? ] [ [ pick2 over contains 1 eq? ] [ quote swap drop pack2 ] [ swap [ cons ] dip pack2 ] ifte ] [ drop pack2 ] ifte ] fold second [ dup null? ] [ drop -1 ] [ head ] ifte` |
| codearc-962 | 118 | `over over gcd [ add 2 sub ] dip add` |
| mbpp-93 | 120 | `pow` |
| mbpp-340 | 123 | `[ 0 gt? ] filter sort 3 take sum` |
| codearc-538 | 124 | `over over gcd [ div ] swap cons [ swap pack2 ] dip map` |
| mbpp-527 | 125 | `[ take ] dip [ [ unpack2 ] dip2 over sub pick3 swap count rot add [ cons ] dip pack2 ] swap cons [ nil 0 pack2 ] dip fold second` |
| codearc-1050 | 129 | `pick2 dec over fourth mul pick2 pick4 sub abs pick4 dec add pick2 head mul 2 pick3 second mul pick3 third add add min2 [ drop drop drop ] dip` |
| codearc-459 | 136 | `[ [ dup ] dip pow swap sub ] swap cons map sum` |
| mbpp-389 | 137 | `range 2 1 pack2 [ drop unpack2 dup rot add pack2 ] fold head` |
| mbpp-498 | 138 | `[ over over mod 0 eq? ] [ swap drop ] [ [ dup 0 lt? ] [ drop drop 1 ] [ gcd ] ifte ] ifte` |
| codearc-1019 | 144 | `pack3 sort unpack3 [ [ add ] dip le? ] [ drop drop drop 0 ] [ sq [ sq [ sq ] dip add ] dip [ eq? ] [ drop drop 2 ] [ [ gt? ] [ drop drop 1 ] [ drop drop 3 ] ifte ] ifte ] ifte` |
| codearc-642 | 151 | `[ div ] swap cons map dup -1 [ max2 ] fold inc range swap [ swap count ] swap cons map` |
| mbpp-308 | 151 | `[ [ swap [ mul ] swap cons map ] swap cons map flatten sort reverse ] dip [ dup 0 lt? ] [ over len add ] [ ] ifte take` |
| codearc-402 | 154 | `[ over 0 le? ] [ drop drop drop 1 ] [ [ pow ] dip mod ] ifte` |
| codearc-150 | 167 | `take max-subarray-sum 0 max2` |
| mbpp-559 | 167 | `take max-subarray-sum 0 max2` |
| codearc-270 | 178 | `[ dup 0 eq? ] [ drop drop 1 ] [ dec 4 mod inc [ 10 mod ] dip pow 10 mod ] ifte` |
| mbpp-295 | 179 | `dup range 2 drop-n swap [ swap divides? ] swap cons filter sum inc` |
| mbpp-516 | 186 | `dup max 1 pack2 zeros16 swap [ drop unpack2 [ over over gt? ] [ 10 mul ] [ ] ifte pack2 ] fold second [ over swap mod swap pack2 ] swap cons map sort-by-head seconds` |
| mbpp-649 | 199 | `over sub inc range swap [ add ] swap cons map swap [ swap at ] swap cons map sum` |
| mbpp-256 | 203 | `primes-below reverse len` |
| mbpp-483 | 207 | `[ dup 2 le? ] [ drop 1 ] [ dup dec range1 swap 1 1 pack3 [ [ unpack3 ] dip [ drop over 0 eq? ] [ drop pack3 ] [ swap drop dup [ mul over mod ] dip pack3 ] ifte ] fold third ] ifte` |
| codearc-736 | 210 | `[ over null? ] [ drop drop 0 ] [ over over at pack3 dup head over third [ lt? ] swap cons filter sum swap dup head over second inc take over third count swap third mul add ] ifte` |
| codearc-1082 | 221 | `[ dup 0 le? ] [ drop drop nil ] [ [ 0 max2 ] dip over over div pick2 pick2 mod pick2 pack3 [ drop drop ] dip unpack3 range -rot [ [ swap ] dip lt? bool>int add ] swap cons swap cons map ] ifte` |
| mbpp-346 | 221 | `[ range nil 1 cons [ drop reverse prefix-sums 0 cons ] fold ] dip nth` |
| codearc-302 | 225 | `dup [ even? ] filter sort reverse over [ odd? ] filter sort nil pack3 [ [ unpack3 ] dip [ even? ] [ drop [ [ uncons ] dip ] dip rot cons ] [ drop [ uncons ] dip swap cons ] ifte pack3 ] fold third reverse` |
| codearc-752 | 230 | `dup len [ [ 0 eq? not ] filter nil 0 pack2 [ [ unpack2 ] dip [ over 0 eq? ] [ swap drop pack2 ] [ [ over over eq? ] [ add cons 0 pack2 ] [ [ cons ] dip pack2 ] ifte ] ifte ] fold unpack2 [ dup 0 eq? ] [ drop ] [ cons ] ifte reverse ] dip over len sub range [ drop 0 ] map cat` |
| codearc-28 | 239 | `dup len range zip [ unpack2 mul ] map 1 drop-n` |
| codearc-828 | 241 | `dup digits len dup 10 swap pow dec 9 div [ swap inc mul ] dip sub` |
| codearc-342 | 249 | `[ inc range ] dip [ [ dup 0 lt? ] [ drop drop 0 ] [ range swap [ swap drop mul inc ] swap cons 1 swap fold ] ifte ] swap cons map sum` |
| codearc-719 | 251 | `dup len range zip 1 drop-n [ unpack2 divides? ] filter [ head ] map` |
| mbpp-348 | 261 | `2 div [ dup 0 lt? ] [ inc 1 swap div ] [ dup range1 over [ add ] swap cons map product over range1 product div swap inc div ] ifte` |
| codearc-630 | 277 | `[ range-between ] dip [ [ over 9 mod over 9 mod eq? ] [ [ [ [ over 10 lt? ] [ drop ] [ [ dup 10 mod swap 10 div ] dip dup i add ] ifte ] dup i ] dip eq? ] [ drop drop false ] ifte ] swap cons filter dup head swap last pack2` |
| codearc-871 | 297 | `[ dup len range swap zip ] dip over -rot [ [ over ] dip [ [ drop over head swap head eq? ] [ drop drop drop false ] [ [ [ second ] dip second add ] dip eq? ] ifte ] swap cons swap cons filter [ dup null? ] [ drop drop nil ] [ head head swap head swap pack2 ] ifte ] swap cons swap cons map [ null? not ] filter head` |
| codearc-699 | 299 | `[ dup 0 lt? ] [ abs digits [ [ dup 5 ge? ] [ dec ] [ ] ifte ] map 0 [ swap 9 mul add ] fold neg ] [ abs digits [ [ dup 5 ge? ] [ dec ] [ ] ifte ] map 0 [ swap 9 mul add ] fold ] ifte` |
| codearc-557 | 315 | `dup len range zip dup [ head ] map dedup swap [ swap [ [ head ] dip eq? ] swap cons filter seconds dup max swap min sub ] swap cons map max` |
| codearc-843 | 319 | `[ dup 0 eq? ] [ drop 4 ] [ dup pows2 swap [ swap div odd? ] swap cons filter head over over add pick2 over bxor pick2 div 4 div add [ drop drop ] dip ] ifte` |
| codearc-471 | 326 | `primes-below reverse swap [ ge? ] swap cons filter [ zeros16 swap [ drop [ dup 6 gt? ] [ [ [ over 10 lt? ] [ drop sq ] [ [ dup 10 mod sq swap 10 div ] dip dup i add ] ifte ] dup i ] [ ] ifte ] fold 1 eq? ] filter len` |
| mbpp-903 | 376 | `range1 [ dup bitlen swap popcount sub ] map sum` |
| codearc-696 | 427 | `swap range1 10 1 add drop-n swap [ [ decimal-digits dup tail zip ] dip [ [ unpack2 swap sub ] dip over 0 gt? bool>int [ le? bool>int ] dip mul 1 eq? not ] swap cons filter null? ] swap cons filter len` |
| mbpp-661 | 434 | `take [ dup len 2 le? ] [ sum ] [ dup 3 drop-n swap dup head over 2 take sum pick2 3 take dup sum swap min sub pick3 third pack4 swap drop [ over second over add pick2 head pick2 add pick3 fourth add max2 pick2 third max2 [ [ dup second swap third ] dip ] dip swap pack4 ] fold third ] ifte` |
| codearc-980 | 482 | `decimal-digits dup len range swap 0 pack2 quote nil swap pack2 [ drop unpack2 [ dup head dedup swap [ unpack2 10 mul pick2 add [ swap remove1 ] dip pack2 ] swap cons map ] map flatten dup seconds rot cat swap pack2 ] fold head dedup sum` |
| codearc-451 | 489 | `[ over null? bool>int over null? bool>int add 0 gt? ] [ drop drop nil ] [ over len over len add dec range -rot [ pack3 dup second len range swap [ unpack3 rot pick3 sub [ [ dup 0 lt? ] [ false ] [ over len over gt? ] ifte ] [ nth -rot swap nth mul ] [ drop drop drop drop 0 ] ifte ] swap cons map sum ] swap cons swap cons map ] ifte` |
| codearc-10 | 525 | `dup len range swap dup dup len range [ 3 mod 0 eq? ] filter swap [ swap at ] swap cons map sort [ [ pick2 3 mod 0 eq? ] [ swap drop swap 3 div nth ] [ drop swap nth ] ifte ] swap cons swap cons map` |
| codearc-711 | 547 | `nil nil pack2 [ [ unpack2 ] dip [ over null? ] [ [ pick2 over contains 1 eq? ] [ swap drop over over over over [ dup len range zip ] dip [ swap head eq? ] swap cons filter head second pick2 len pick3 pick3 [ dup len range zip ] dip [ swap head eq? ] swap cons filter head second sub pack2 [ drop drop ] dip [ drop ] dip pack2 ] [ swap [ quote cat ] dip pack2 ] ifte ] [ drop pack2 ] ifte ] fold second` |
| codearc-483 | 554 | `over over over over dec slice-index take pick2 pick2 slice-index drop-n cat [ drop drop ] dip -rot swap drop [ [ [ unpack3 ] dip ] dip pick4 inc pick4 pick4 pick4 add 100 gt? bool>int [ [ [ pick4 ] dip swap ] dip swap ] dip swap pick4 2 sub eq? bool>int add add pick4 [ pick4 ] dip swap [ pick4 ] dip swap add 100 le? bool>int [ [ [ [ pick4 ] dip swap ] dip swap ] dip swap ] dip swap [ pick4 ] dip swap 2 sub eq? not bool>int mul mul pick4 add pack3 [ drop drop drop drop drop ] dip ] swap cons 0 1 0 pack3 swap fold second` |
| mbpp-155 | 569 | `[ dup 0 gt? ] [ dup bitlen range [ odd? ] filter [ pow2 ] map sum bxor ] [ ] ifte` |
| mbpp-926 | 571 | `over factorial over factorial pick3 pick3 sub factorial mul div pick2 pick2 sub [ dup 0 eq? ] [ drop 1 ] [ range1 tail 1 0 pack2 [ [ unpack2 ] dip over over dec pick4 pick4 add mul pack2 [ drop drop drop ] dip ] fold second ] ifte mul [ drop drop ] dip` |
| mbpp-531 | 581 | `[ take ] dip range swap nil 0 cons pack2 [ drop unpack2 over over swap over len [ le? ] swap cons filter swap [ swap dec nth ] swap cons map 2 8 sq dec pow dec [ min2 ] fold inc 2 8 sq dec pow dec min2 cons pack2 ] fold second head` |
| codearc-921 | 613 | `dup 2 div range1 [ 2 ge? ] filter [ prime? ] filter [ range1 3 drop-n ] dip rot [ [ over [ swap mod 0 eq? ] swap cons filter ] dip [ over null? ] [ drop drop drop false ] [ [ drop head eq? ] [ drop drop drop false ] [ [ dup head swap last add ] dip eq? swap drop ] ifte ] ifte ] swap cons swap cons filter` |
| codearc-175 | 630 | `[ dup sq sq 1000 dup mul 1000 mul dup sq sq sq dup sq mul mul mul swap inc 1000 dup mul 1000 mul mul 0 swap pack3 zeros16 dup cat dup cat swap [ drop unpack3 over over add 2 div [ dup dup sq sq sq dup sq mul mul pick4 le? ] [ rot drop swap ] [ swap drop ] ifte pack3 ] fold second ] dip dup 6 2 1 5 pack4 from-digits mul 10 3 add 10 mul 1 add 10 mul 2 add 100 mul add 1000 dup mul 1000 mul mul [ 3 9 6 5 pack4 from-digits mul 10 1 add 10 mul 3 add 10 mul 7 add 100 mul sub mul ] dip add 1000 dup mul 1000 mul 100 100 mul mul over over mod 2 mul over [ over over eq? ] [ drop drop div dup odd? bool>int add ] [ gt? bool>int [ div ] dip add ] ifte` |
| mbpp-103 | 659 | `[ over 0 eq? ] [ drop drop 0 ] [ [ over over le? ] [ drop drop 0 ] [ pack2 dup second 2 add range swap unpack2 [ pack3 dup third inc over head sub over second pow over second inc pick2 head binomial mul swap head even? bool>int 2 mul dec mul ] swap cons swap cons map sum ] ifte ] ifte` |
| codearc-292 | 726 | `decimal-digits dup len 2 sub range1 swap [ swap over len dec over inc swap over sub inc range swap [ add ] swap cons map -rot [ rot pick2 pick2 take from-digits pick3 pick3 drop-n pick2 pick4 sub take from-digits mul pick3 pick2 drop-n from-digits mul [ drop drop drop ] dip ] swap cons swap cons map max ] swap cons map max` |
| mbpp-843 | 735 | `[ over 1 le? ] [ drop drop 1 ] [ over over len mul range -rot dup len range swap zip [ unpack2 dup quote pack3 ] map 1 1 pack3 swap [ swap drop [ over third swap ge? ] [ drop ] [ drop unpack3 pick2 uncons [ [ over third head over third head le? ] [ drop ] [ swap drop ] ifte ] fold dup head swap third head [ pick3 eq? ] [ drop [ rot ] dip [ [ over head eq? ] [ drop unpack3 tail [ dup null? ] [ drop drop drop nil ] [ pack3 ] ifte ] [ drop ] ifte ] swap cons map [ null? not ] filter -rot pack3 ] [ [ [ swap drop inc ] dip ] dip [ [ swap ] dip ] dip dup [ [ [ drop over head eq? ] [ [ drop ] dip [ unpack3 ] dip pick2 mul quote cat pack3 unpack3 tail [ dup null? ] [ drop drop drop nil ] [ pack3 ] ifte ] [ [ drop ] dip [ unpack3 ] dip pick2 mul quote cat pack3 ] ifte ] swap cons swap cons map [ null? not ] filter ] dip rot pack3 ] ifte ] ifte ] swap cons fold second ] ifte` |
| codearc-578 | 740 | `pack2 dup dup second inc range swap unpack2 [ pack3 dup head over second pow over third pick2 head binomial mul swap dup third swap head sub even? bool>int 2 mul dec mul ] swap cons swap cons map sum swap second factorial div` |
| codearc-323 | 778 | `pack2 dup second 9 div inc range swap unpack2 [ pack3 dup head dup nil 1 cons dup head 10 mul inc cons dup head 10 mul inc cons dup head 10 mul inc cons dup head 10 mul inc cons dup head 10 mul inc cons swap [ le? ] swap cons filter swap 0 pack2 [ [ unpack2 ] dip pick2 over div 9 min2 dup [ mul ] dip [ rot swap sub ] dip rot 10 mul add pack2 ] fold second 10 mul 9 add over third min2 over dup head 9 mul over second max2 swap dup head 9 mul 8 add swap third min2 over sub inc range swap [ add ] swap cons map rot drop swap [ over dup [ [ over 1000 lt? ] [ drop dup 10 div over 100 div add 9 mul sub ] [ [ dup 1000 mod dup 10 div over 100 div add 9 mul sub swap 1000 div ] dip dup i add ] ifte ] dup i add min2 swap sub 0 max2 ] swap cons map sum ] swap cons swap cons map sum` |
| mbpp-633 | 786 | `take nil 0 pack2 [ [ unpack2 ] dip pick2 over [ sxor ] swap cons map sum rot add [ cons ] dip pack2 ] fold second` |
| mbpp-203 | 827 | `sxor 0 max2 popcount` |
| mbpp-934 | 912 | `pack2 dup unpack2 min2 inc range swap unpack2 [ pack3 dup second over head binomial over third pick2 head binomial mul swap head pow2 mul ] swap cons swap cons map sum` |
| codearc-420 | 964 | `[ dup 1000 lt? ] [ range-between [ dup [ dup 100 ge? ] [ dup 100 div over 10 div 10 mod sq add swap 10 mod dup sq mul add ] [ [ dup 10 lt? ] [ ] [ dup 10 div swap 10 mod sq add ] ifte ] ifte eq? ] filter ] [ range-between [ dup [ dup 10 lt? ] [ ] [ [ dup 100 lt? ] [ dup 10 div swap 10 mod sq add ] [ [ dup 1000 lt? ] [ dup 100 div over 10 div 10 mod sq add swap 10 mod dup sq mul add ] [ [ dup 10 lt? ] [ quote ] [ dup 10 div [ dup 10 lt? ] [ quote ] [ dup 10 div digits swap 10 mod quote cat ] ifte swap 10 mod quote cat ] ifte 0 0 pack2 [ [ unpack2 ] dip swap inc dup [ zeros16 swap take swap 1 pack2 [ drop unpack2 over mul pack2 ] fold second ] dip [ add ] dip pack2 ] fold head ] ifte ] ifte ] ifte eq? ] filter ] ifte` |
| mbpp-815 | 1316 | `dup range [ dec 0 0 rot pack4 ] dip swap [ drop unpack4 [ over over le? ] [ [ pick3 pick2 nth 0 eq? ] [ pick3 pick3 pick3 pick2 pick2 pick4 pick3 nth set-at over pick4 pick4 nth set-at [ drop drop drop ] dip pick3 inc pick3 inc pick3 pack4 [ drop drop drop drop ] dip ] [ [ pick3 pick2 nth 1 eq? ] [ pick3 pick3 pick3 inc pick3 pack4 [ drop drop drop drop ] dip ] [ pick3 pick2 pick2 pick2 pick2 pick4 pick3 nth set-at over pick4 pick4 nth set-at [ drop drop drop ] dip pick3 pick3 pick3 dec pack4 [ drop drop drop drop ] dip ] ifte ] ifte ] [ pack4 ] ifte ] fold head` |
| mbpp-560 | 2578 | `cat dedup 8 range [ drop nil ] map 0 pack2 [ [ unpack2 ] dip pick2 over over over dup [ dup -1 eq? ] [ dec ] [ ] ifte pick2 len mod [ dup pick3 pick3 [ swap nth ] dip [ over null? ] [ drop drop true ] [ swap head eq? ] ifte ] [ [ drop drop ] dip ] [ drop over len 16 add range -rot over over dup [ dup -1 eq? ] [ dec ] [ ] ifte pick2 len mod over [ dup -1 eq? ] [ dec ] [ ] ifte [ dup 0 lt? ] [ 2 8 sq pow add ] [ ] ifte -1 pack3 [ drop drop ] dip -rot [ [ drop ] dip2 [ unpack3 ] dip2 [ pick2 0 lt? ] [ pick4 5 mul 1 add pick4 32 div add pick2 len mod pick4 32 div pick3 pick3 [ [ [ [ pick4 ] dip swap ] dip swap ] dip swap ] dip swap [ dup 9 add pick3 len lt? ] [ dup 9 add over sub inc range swap [ add ] swap cons map ] [ quote ] ifte -rot [ [ swap nth ] dip [ over null? ] [ drop drop true ] [ swap head eq? ] ifte ] swap cons swap cons filter [ dup null? ] [ drop -1 ] [ head ] ifte pack3 [ drop drop drop drop drop ] dip ] [ drop drop pack3 ] ifte ] swap cons swap cons fold third ] ifte swap quote set-at [ drop ] dip rot drop swap inc pack2 [ dup second 5 mul over head len dec 3 mul ge? ] [ unpack2 [ dup flatten over len 4 mul range [ drop nil ] map [ over over dup [ dup -1 eq? ] [ dec ] [ ] ifte pick2 len mod [ dup pick3 pick3 [ swap nth ] dip [ over null? ] [ drop drop true ] [ swap head eq? ] ifte ] [ [ drop drop ] dip ] [ drop over len 16 add range -rot over over dup [ dup -1 eq? ] [ dec ] [ ] ifte pick2 len mod over [ dup -1 eq? ] [ dec ] [ ] ifte [ dup 0 lt? ] [ 2 8 sq pow add ] [ ] ifte -1 pack3 [ drop drop ] dip -rot [ [ drop ] dip2 [ unpack3 ] dip2 [ pick2 0 lt? ] [ pick4 5 mul 1 add pick4 32 div add pick2 len mod pick4 32 div pick3 pick3 [ [ [ [ pick4 ] dip swap ] dip swap ] dip swap ] dip swap [ dup 9 add pick3 len lt? ] [ dup 9 add over sub inc range swap [ add ] swap cons map ] [ quote ] ifte -rot [ [ swap nth ] dip [ over null? ] [ drop drop true ] [ swap head eq? ] ifte ] swap cons swap cons filter [ dup null? ] [ drop -1 ] [ head ] ifte pack3 [ drop drop drop drop drop ] dip ] [ drop drop pack3 ] ifte ] swap cons swap cons fold third ] ifte swap quote set-at ] fold swap drop ] dip pack2 ] [ ] ifte ] fold head flatten` |
