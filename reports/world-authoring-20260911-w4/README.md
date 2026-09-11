# world-authoring-500 (#127), wave 4: 100 MBPP / CodeARC behaviours authored by codex exec

Behaviours sampled by `experiments/world_behaviours/extract.py` (seed 17, MBPP remainder and CodeARC, excluding the probe and waves 1–3, hypothesis-breaking sampler on every source); programs authored by four `codex exec` agents (gpt-6-astra, high effort) in macro form and migrated to the canonical library; every program re-verified on every test by `report.py`.

## Findings (written with the 2026-09-11 wave 4; the tables below regenerate)

- **100 of 100 verified** (50/50 MBPP, 50/50 CodeARC) after one two-op
  repair: an identity behaviour (copy a list) submitted as the empty program,
  the same case as wave 2. Over the probe and four waves MBPP stands at
  234/236 and CodeARC at 180/184.
- **Cost rose to 432k tokens** (50k–167k per agent; author d spent 167k on
  its 24) for 7 minutes wall and 20 agent-minutes. The variance is the
  behaviours, not the settings: one agent drew the long tail.
- **Library: 4 macros per 100** (125 → 129): `adjacent` (two authors, five
  programs), `differences`, `triangle` (two authors), `ceil-div` (an author in
  wave 2 had written it too). 64 one-offs inlined. The rate has fallen from 16
  to 7, 8, 4 per 100 as the pool's easier MBPP problems come up.
- **Overlap with v4 grows as the pool thins**: 9 exact witness matches
  (scalar one-liners such as `mbpp-8`, `mbpp-404`) and 11 shared skeletons in
  100. The wider-pool sampler is now reaching MBPP's simplest problems; the
  v4-eval overlap removal in the step-3 training plan matters more here.

## Verified

| source | verified | median witness length (`mark-28` ops) | median macro-form length |
|---|---|---|---|
| codearc | 50/50 (100%) | 135.0 | 20.0 |
| mbpp | 50/50 (100%) | 106.5 | 11.0 |
| **all** | 100/100 (100%) | 126.0 | 18.5 |

Claimed-vs-verified mismatches: 0; missing ids: 0; duplicates: 0.

By author: canonical 100/100 (100%).

## Witness length against `stack_pbe_v4`

| bucket (ops) | this probe (verified) | v4 behaviours (train + eval, 220) |
|---|---|---|
| 1-5 | 6 | 15 |
| 6-10 | 1 | 14 |
| 11-20 | 14 | 34 |
| 21-40 | 8 | 37 |
| 41+ | 71 | 120 |
| median | 126.0 | 43.0 |

Exact matches with a v4 witness: 9 (codearc-111, codearc-128, mbpp-250, mbpp-404, mbpp-443, mbpp-514, mbpp-8, mbpp-924, mbpp-929). Same skeleton as a v4 witness: 11.

## Macros

133 macros defined across `macros*.txt`; median expansion ratio (`mark-28` ops per macro-form token) on verified programs: 6.7.

| macro | verified programs using it |
|---|---|
| `len` | 23 |
| `inc` | 21 |
| `head` | 20 |
| `unpack2` | 20 |
| `range` | 19 |
| `pack2` | 16 |
| `sum` | 15 |
| `dec` | 15 |
| `gt?` | 15 |
| `2` | 15 |
| `pick2` | 13 |
| `count` | 11 |
| `max2` | 11 |
| `3` | 11 |
| `take` | 11 |
| `second` | 10 |
| `-1` | 9 |
| `10` | 9 |
| `reverse` | 8 |
| `pack3` | 8 |

## Unsolved

| id | source | passed | note |
|---|---|---|---|

## Every verified program

| id | len | macro form |
|---|---|---|
| mbpp-127 | 1 | `mul` |
| mbpp-789 | 1 | `mul` |
| mbpp-847 | 2 | `dup drop` |
| mbpp-35 | 4 | `dup inc mul` |
| mbpp-514 | 5 | `sum` |
| mbpp-8 | 5 | `[ sq ] map` |
| mbpp-447 | 7 | `[ dup sq mul ] map` |
| codearc-128 | 12 | `min2` |
| codearc-288 | 12 | `[ divides? ] swap cons filter` |
| mbpp-404 | 12 | `min2` |
| mbpp-924 | 12 | `max2` |
| codearc-111 | 13 | `count` |
| mbpp-250 | 13 | `count` |
| mbpp-629 | 13 | `[ even? ] filter` |
| mbpp-929 | 13 | `count` |
| mbpp-279 | 14 | `dup 4 mul 3 sub mul` |
| mbpp-435 | 14 | `10 mod` |
| codearc-1063 | 16 | `add swap sub 0 max2` |
| mbpp-443 | 16 | `min` |
| codearc-918 | 18 | `[ even? ] filter sum` |
| mbpp-267 | 18 | `dup sq 4 mul dec mul 3 div` |
| codearc-328 | 23 | `pack3 [ triangle ] map product` |
| mbpp-632 | 24 | `dup [ 0 eq? not ] filter swap [ 0 eq? ] filter cat` |
| mbpp-672 | 24 | `max2 max2` |
| codearc-446 | 26 | `[ dup 0 eq? ] [ ] [ dec 9 mod inc ] ifte` |
| codearc-49 | 26 | `2 div 2 mul [ over over gt? ] [ drop drop -1 ] [ swap drop ] ifte` |
| mbpp-42 | 29 | `drop dup [ swap count 1 gt? ] swap cons filter sum` |
| codearc-535 | 30 | `dup [ swap count even? ] swap cons filter` |
| codearc-830 | 36 | `dup [ swap count ] swap cons map 0 [ max2 ] fold` |
| mbpp-236 | 47 | `[ over over lt? ] [ drop drop -1 ] [ over over sub inc triangle [ 2 mul sub inc triangle ] dip add ] ifte` |
| codearc-1048 | 48 | `[ [ pick2 pick2 count over lt? ] [ drop cons ] [ drop drop ] ifte ] swap cons nil swap fold reverse` |
| codearc-336 | 56 | `[ null? ] [ drop 0 ] [ dup max over min sub inc swap len sub ] ifte` |
| codearc-949 | 56 | `1 0 pack2 [ [ unpack2 ] dip [ over add 100 gt? ] [ swap drop [ inc ] dip ] [ add ] ifte pack2 ] fold head` |
| mbpp-428 | 57 | `sort` |
| codearc-396 | 65 | `[ dup len 3 lt? ] [ drop 0 ] [ dup sum over max sub swap min sub ] ifte` |
| mbpp-953 | 70 | `[ dup 0 le? ] [ drop drop 0 ] [ [ dup min count ] dip min2 ] ifte` |
| mbpp-915 | 78 | `sort dup [ 0 gt? ] filter over [ 0 eq? ] filter cat swap [ 0 lt? ] filter cat` |
| mbpp-734 | 90 | `take reverse 0 0 pack2 [ [ unpack2 ] dip swap inc mul dup [ add ] dip pack2 ] fold head` |
| mbpp-145 | 98 | `1 max2 take dup max swap min sub` |
| codearc-498 | 102 | `6 10 mul 0 pack2 [ [ unpack2 ] dip [ over 1 eq? ] [ drop pack2 ] [ swap drop [ over 3 sub gt? ] [ min2 1 pack2 ] [ drop 3 sub 0 pack2 ] ifte ] ifte ] fold head` |
| codearc-740 | 112 | `dec 8 mul inc isqrt inc 2 div` |
| codearc-525 | 114 | `over over gcd [ add ] dip div` |
| codearc-260 | 115 | `swap [ gcd 1 eq? ] swap cons filter` |
| mbpp-232 | 115 | `[ sort reverse ] dip take` |
| codearc-636 | 118 | `[ range1 ] dip [ mul ] swap cons map` |
| mbpp-21 | 118 | `[ range1 ] dip [ mul ] swap cons map` |
| mbpp-856 | 118 | `take reverse uncons neg inc 0 pack2 [ [ unpack2 ] dip [ dup 0 eq? ] [ drop [ inc ] dip ] [ [ dup 1 eq? ] [ drop over add ] [ drop ] ifte ] ifte pack2 ] fold second` |
| codearc-294 | 119 | `dup product swap dup dedup over [ swap count inc ] swap cons map product dec swap min pick2 over div pack4` |
| mbpp-767 | 122 | `[ take ] dip [ [ [ unpack2 ] dip ] dip over sub pick3 swap count rot add [ cons ] dip pack2 ] swap cons nil 0 pack2 swap fold second` |
| codearc-167 | 125 | `[ pow ] swap cons map` |
| mbpp-503 | 127 | `adjacent [ sum ] map` |
| codearc-333 | 128 | `dup [ odd? ] filter sort nil pack2 [ [ unpack2 ] dip [ dup odd? ] [ drop [ uncons ] dip swap ] [ ] ifte cons pack2 ] fold second reverse` |
| mbpp-690 | 128 | `adjacent [ unpack2 mul ] map` |
| codearc-106 | 132 | `popcount` |
| codearc-977 | 132 | `[ 2 swap pow ] map sum` |
| codearc-856 | 134 | `[ sort ] dip dec at` |
| mbpp-345 | 134 | `differences [ neg ] map` |
| codearc-466 | 136 | `swap [ [ over over ge? ] [ add ] [ over swap gcd add ] ifte ] fold` |
| codearc-643 | 136 | `dup len [ swap count ] dip dup rot sub 100 mul swap over over div [ over over mod 2 mul ] dip pack3 swap drop [ unpack3 drop lt? ] [ third inc ] [ [ unpack3 drop eq? ] [ third dup 2 mod add ] [ third ] ifte ] ifte` |
| mbpp-586 | 143 | `[ drop ] dip slice-index over over drop-n [ take ] dip swap cat` |
| codearc-680 | 147 | `adjacent [ unpack2 swap sub 0 max2 ] map sum` |
| codearc-873 | 162 | `0 nil pack2 [ [ unpack2 ] dip [ over len 2 lt? ] [ cons ] [ [ over dup head swap second sub pick2 head pick2 swap sub eq? ] [ cons ] [ [ dup head [ [ dup len 3 lt? ] [ drop 0 ] [ sum ] ifte add ] dip ] dip pack2 reverse ] ifte ] ifte pack2 ] fold unpack2 [ dup len 3 lt? ] [ drop 0 ] [ sum ] ifte add` |
| codearc-90 | 163 | `dup head 0 lt? bool>int 2 mul 1 swap sub [ [ abs decimal-digits ] map flatten from-digits ] dip mul` |
| mbpp-328 | 174 | `[ over swap slice-index drop-n ] dip [ swap ] dip slice-index take cat` |
| mbpp-36 | 181 | `[ dup 1 eq? ] [ drop [ 10 mul ] dip div ] [ dec 10 swap pow rot mul over mod 10 mul swap div ] ifte` |
| mbpp-685 | 198 | `inc primes-below sum` |
| codearc-532 | 202 | `sort adjacent [ unpack2 swap sub ] map max` |
| codearc-59 | 203 | `dup len range [ even? ] filter swap [ swap nth ] swap cons map [ odd? ] filter sum` |
| codearc-53 | 210 | `[ [ dup 0 lt? ] [ neg digits uncons neg swap sum add ] [ digit-sum ] ifte 0 gt? ] filter len` |
| mbpp-702 | 212 | `[ [ sort ] dip ] dip [ dup [ dec take ] dip ] dip swap [ over swap [ [ swap ] dip [ [ sub ] dip [ over 0 ge? ] [ le? ] [ drop drop false ] ifte ] swap cons swap cons filter len ] swap cons swap cons map 1 [ max2 ] fold ] dip swap sub` |
| codearc-676 | 217 | `over len range -rot [ [ swap nth ] dip eq? ] swap cons swap cons filter [ dup len 2 eq? ] [ unpack2 swap sub inc ] [ drop 0 ] ifte` |
| mbpp-448 | 234 | `[ dup 0 eq? ] [ drop 3 ] [ [ dup 1 eq? ] [ drop 3 ] [ 2 sub range 3 0 2 5 pack4 [ drop unpack4 pick3 pick3 add [ rot drop ] dip2 dup [ add ] dip swap pack4 ] fold fourth ] ifte ] ifte` |
| codearc-4 | 238 | `[ dup 0 eq? ] [ drop ] [ [ dup 0 lt? ] [ gcd neg ] [ gcd ] ifte ] ifte` |
| mbpp-943 | 253 | `over len over len add range [ nil pack3 ] dip swap [ drop unpack3 [ pick2 null? ] [ [ uncons ] dip swap cons ] [ [ over null? ] [ [ [ uncons ] dip swap ] dip swap cons ] [ [ pick2 head pick2 head le? ] [ [ [ uncons ] dip swap ] dip swap cons ] [ [ uncons ] dip swap cons ] ifte ] ifte ] ifte pack3 ] fold third reverse` |
| codearc-1032 | 265 | `[ prefix-sums dup len range zip ] dip over len [ [ swap head lt? ] swap cons filter seconds ] dip quote cat head` |
| codearc-1102 | 277 | `dup len range swap nil pack2 [ drop unpack2 over last cons [ differences ] dip pack2 ] fold second` |
| mbpp-412 | 278 | `dup len range swap [ [ over len lt? ] [ over swap nth [ dup odd? ] [ remove1 ] [ drop ] ifte ] [ drop ] ifte ] fold` |
| mbpp-116 | 286 | `[ dup head 0 lt? ] [ [ abs decimal-digits ] map flatten from-digits neg ] [ [ abs decimal-digits ] map flatten from-digits ] ifte` |
| codearc-914 | 305 | `10 3 add div inc range [ 10 3 add mul digits dup 4 contains swap 7 contains add 0 eq? ] filter len` |
| codearc-1042 | 306 | `swap 6 mul sub 5 add [ dup 5 lt? ] [ 5 swap div log2 neg ] [ 5 div log2 ] ifte` |
| codearc-475 | 330 | `dup max dec [ dup 1 lt? ] [ drop drop -1 ] [ swap dup head [ sub ] swap cons map 0 [ gcd ] fold [ dup 0 eq? ] [ drop ] [ [ over over ge? ] [ swap drop ] [ swap range1 swap [ swap divides? ] swap cons filter max ] ifte ] ifte ] ifte` |
| mbpp-802 | 341 | `take dup len range 1 drop-n swap [ swap over over nth [ dec nth ] dip gt? ] swap cons filter 0 quote cat head` |
| codearc-67 | 354 | `adjacent [ unpack2 gt? bool>int ] map dup len range1 zip [ head 1 eq? ] filter seconds -1 quote swap cat last` |
| codearc-734 | 357 | `decimal-digits dup sum swap len dup dec factorial swap range [ drop 1 ] map from-digits mul mul` |
| mbpp-275 | 360 | `[ take ] dip [ ceil-div ] swap cons map dup len range1 zip [ head -1 gt? ] filter [ dup null? ] [ drop 0 ] [ sort-by-head last second ] ifte` |
| codearc-306 | 421 | `[ dup len range dup [ over inc drop-n swap [ swap pack2 ] swap cons map ] swap cons map flatten swap ] dip [ [ over head over swap nth [ swap second nth ] dip add ] dip eq? ] swap cons swap cons filter nil [ [ over over head contains 1 eq? ] [ drop ] [ [ over over second contains 1 eq? ] [ drop ] [ cat ] ifte ] ifte ] fold sum` |
| codearc-26 | 426 | `dup range 2 drop-n swap [ swap divides? ] swap cons filter [ prime? ] filter max` |
| codearc-1106 | 441 | `nil -1 pack3 zeros128 swap [ drop unpack3 [ dup -1 eq? ] [ drop [ over contains 1 eq? ] [ over over swap over len range -rot [ [ swap at ] dip eq? ] swap cons swap cons filter head inc pack3 ] [ over cons swap digits [ sq ] map sum swap -1 pack3 ] ifte ] [ pack3 ] ifte ] fold third` |
| codearc-663 | 451 | `[ dup head swap second dup [ abs pow ] dip [ dup 0 lt? ] [ drop 1 swap ] [ drop 1 ] ifte pack2 ] dip dup head swap second dup [ abs pow ] dip [ dup 0 lt? ] [ drop 1 swap ] [ drop 1 ] ifte pack2 reverse zip [ product ] map unpack2 [ over over lt? ] [ drop drop 1 ] [ [ eq? ] [ drop drop 0 ] [ drop drop -1 ] ifte ] ifte` |
| mbpp-225 | 469 | `over over sub abs inc range [ pack3 ] dip swap [ drop unpack3 [ over over lt? ] [ over over add 2 div pick3 swap at pick3 pick2 at sub [ dup 0 eq? ] [ drop dec ] [ [ dup 0 gt? ] [ drop over over add 2 div inc rot drop swap ] [ drop over over add 2 div swap drop ] ifte ] ifte ] [ ] ifte pack3 ] fold unpack3 swap drop at` |
| mbpp-638 | 570 | `[ dup sq sq 1000 sq sq dup sq sq sq dup sq mul mul mul swap inc 1000 sq sq mul 0 swap pack3 zeros16 dup cat dup cat swap [ drop unpack3 over over add 2 div [ dup dup sq sq sq dup sq mul mul pick4 le? ] [ [ swap drop ] dip swap ] [ [ drop ] dip ] ifte pack3 ] fold second ] dip dup 6 10 mul 2 add 10 mul 1 add 10 mul 5 add mul 1 10 mul 3 add 10 mul 1 add 10 mul 2 add 100 mul add 1000 sq sq mul [ 3 10 mul 9 add 10 mul 6 add 10 mul 5 add mul 1 10 mul 1 add 10 mul 3 add 10 mul 7 add 100 mul sub mul ] dip add 100 sq 1000 sq sq mul over over div -rot over over mod [ swap drop ] dip dup add [ over over lt? ] [ drop drop inc ] [ [ over over eq? ] [ drop drop [ dup odd? ] [ inc ] [ ] ifte ] [ drop drop ] ifte ] ifte` |
| mbpp-229 | 582 | `range [ 0 pack2 ] dip swap [ [ unpack2 ] dip [ pick2 over nth 0 lt? ] [ over [ pick2 over nth pick3 pick3 nth [ swap ] dip [ set-at ] dip2 set-at ] dip inc pack2 ] [ drop pack2 ] ifte ] fold head` |
| codearc-235 | 584 | `decimal-digits dup len range swap 0 pack2 quote [ drop [ dup unpack2 over dedup -rot [ 10 mul pick2 add [ swap remove1 ] dip pack2 ] swap cons swap cons map swap cons ] map flatten dedup ] fold seconds dedup [ dup 0 gt? bool>int swap 3 divides? bool>int mul 1 eq? ] filter dup len swap max pack2` |
| codearc-691 | 716 | `dup isqrt range1 1 drop-n [ [ 4 max2 ] dip range-between ] dip [ over swap swap 0 pack2 [ [ unpack2 ] dip [ pick2 over divides? ] [ dup 1 pack2 quote dup head unpack2 [ sq ] dip dup add pack2 cons dup head unpack2 [ sq ] dip dup add pack2 cons dup head unpack2 [ sq ] dip dup add pack2 cons dup head unpack2 [ sq ] dip dup add pack2 cons swap [ swap unpack2 rot mul [ unpack2 ] dip2 [ pick3 pick2 divides? ] [ [ [ swap ] dip div swap ] dip add pack2 ] [ drop drop pack2 ] ifte ] swap cons [ pack2 ] dip2 [ swap ] dip fold ] [ drop pack2 ] ifte ] fold unpack2 [ over 1 eq? ] [ swap drop ] [ add ] ifte [ over over eq? ] [ drop drop false ] [ divides? ] ifte ] swap cons filter` |
| mbpp-71 | 828 | `dup len dup range -rot pack2 [ drop unpack2 [ dup 1 gt? ] [ 10 mul 10 3 add div dup [ over len over sub range swap [ over add [ pick2 pick2 nth pick3 pick2 nth gt? ] [ pick2 pick2 nth pick3 pick2 nth -rot [ set-at ] dip2 set-at ] [ drop drop ] ifte ] swap cons [ swap ] dip fold ] dip ] [ ] ifte pack2 ] fold head` |
| codearc-788 | 854 | `[ ge? ] [ drop drop drop nil ] [ pick2 sub [ 2 max2 ] dip range-between swap [ [ pick2 null? ] [ over add [ [ over [ dup 2 eq? ] [ drop true ] [ dup 2 gt? bool>int swap odd? bool>int mul 1 eq? ] ifte ] [ [ dup [ dup 2 eq? ] [ drop true ] [ dup 2 gt? bool>int swap odd? bool>int mul 1 eq? ] ifte ] [ [ over prime? ] [ prime? ] [ false ] ifte ] [ false ] ifte ] [ false ] ifte ] [ pack2 swap drop ] [ drop drop ] ifte ] [ drop drop ] ifte ] swap cons nil swap fold ] ifte` |
| mbpp-550 | 927 | `pack3 zeros16 dup cat swap [ drop [ dup len 1 eq? ] [ ] [ unpack3 [ gt? ] [ drop drop head quote ] [ [ eq? ] [ drop at quote ] [ over over add 2 div [ [ over lt? ] [ pick3 over inc at pick4 pick2 at lt? ] [ false ] ifte ] [ [ drop drop ] dip at quote ] [ [ [ pick2 gt? ] [ pick3 over at pick4 pick2 dec at lt? ] [ false ] ifte ] [ dec [ drop drop ] dip at quote ] [ [ pick3 pick3 at pick4 pick2 at gt? ] [ dec swap drop pack3 ] [ inc rot drop swap pack3 ] ifte ] ifte ] ifte ] ifte ] ifte ] ifte ] fold head` |
| codearc-229 | 1198 | `[ gt? ] [ drop drop zeros8 0 cons 0 cons ] [ [ dec [ dup 0 lt? ] [ drop zeros8 0 cons 0 cons ] [ 10 range swap [ over 0 eq? bool>int [ swap 10 powers -rot [ rot pick2 inc over 10 mul div over mul pick3 inc pick2 10 mul mod pick3 pick3 mul sub 0 max2 pick2 min2 add [ pick2 0 eq? ] [ swap sub 0 max2 ] [ [ drop ] dip ] ifte [ drop drop ] dip ] swap cons swap cons map sum ] dip add ] swap cons map ] ifte ] dip [ dup 0 lt? ] [ drop zeros8 0 cons 0 cons ] [ 10 range swap [ over 0 eq? bool>int [ swap 10 powers -rot [ rot pick2 inc over 10 mul div over mul pick3 inc pick2 10 mul mod pick3 pick3 mul sub 0 max2 pick2 min2 add [ pick2 0 eq? ] [ swap sub 0 max2 ] [ [ drop ] dip ] ifte [ drop drop ] dip ] swap cons swap cons map sum ] dip add ] swap cons map ] ifte swap zip [ unpack2 sub ] map ] ifte` |
| mbpp-735 | 1840 | `[ dup 1 eq? ] [ ] [ [ dup 0 lt? ] [ -1 dec sxor ] [ dup bitlen dec 0 max2 pow2 dec 1 bxor sxor ] ifte ] ifte` |
| mbpp-2 | 4136 | `[ dedup ] dip dedup [ over len over len lt? ] [ swap ] [ ] ifte dedup zeros8 [ drop nil ] map [ over over [ dup -1 eq? ] [ dec ] [ ] ifte [ dup 0 lt? ] [ 2 32 pow sq add ] [ ] ifte over len 16 add range -rot over len over swap mod swap pack3 [ drop [ dup len 1 eq? ] [ ] [ unpack3 [ [ over len over sub 10 ge? ] [ over over drop-n 10 take ] [ over over drop-n 1 take ] ifte 0 -1 pack2 [ [ unpack2 ] dip [ over 0 lt? ] [ [ null? ] [ drop drop dup ] [ drop ] ifte ] [ drop ] ifte [ inc ] dip pack2 ] fold second ] dip swap [ dup 0 ge? ] [ [ drop ] dip add swap drop quote ] [ drop 32 div swap 5 mul inc over add pick2 len mod swap pack3 ] ifte ] ifte ] fold head swap quote set-at [ dup flatten len 5 mul over len dec 3 mul ge? ] [ dup len 4 mul range [ drop nil ] map swap flatten swap [ over over [ dup -1 eq? ] [ dec ] [ ] ifte [ dup 0 lt? ] [ 2 32 pow sq add ] [ ] ifte over len 16 add range -rot over len over swap mod swap pack3 [ drop [ dup len 1 eq? ] [ ] [ unpack3 [ [ over len over sub 10 ge? ] [ over over drop-n 10 take ] [ over over drop-n 1 take ] ifte 0 -1 pack2 [ [ unpack2 ] dip [ over 0 lt? ] [ [ null? ] [ drop drop dup ] [ drop ] ifte ] [ drop ] ifte [ inc ] dip pack2 ] fold second ] dip swap [ dup 0 ge? ] [ [ drop ] dip add swap drop quote ] [ drop 32 div swap 5 mul inc over add pick2 len mod swap pack3 ] ifte ] ifte ] fold head swap quote set-at ] fold ] [ ] ifte ] fold flatten over [ swap contains 1 eq? ] swap cons filter swap drop dedup zeros8 [ drop nil ] map [ over over [ dup -1 eq? ] [ dec ] [ ] ifte [ dup 0 lt? ] [ 2 32 pow sq add ] [ ] ifte over len 16 add range -rot over len over swap mod swap pack3 [ drop [ dup len 1 eq? ] [ ] [ unpack3 [ [ over len over sub 10 ge? ] [ over over drop-n 10 take ] [ over over drop-n 1 take ] ifte 0 -1 pack2 [ [ unpack2 ] dip [ over 0 lt? ] [ [ null? ] [ drop drop dup ] [ drop ] ifte ] [ drop ] ifte [ inc ] dip pack2 ] fold second ] dip swap [ dup 0 ge? ] [ [ drop ] dip add swap drop quote ] [ drop 32 div swap 5 mul inc over add pick2 len mod swap pack3 ] ifte ] ifte ] fold head swap quote set-at [ dup flatten len 5 mul over len dec 3 mul ge? ] [ dup len 4 mul range [ drop nil ] map swap flatten swap [ over over [ dup -1 eq? ] [ dec ] [ ] ifte [ dup 0 lt? ] [ 2 32 pow sq add ] [ ] ifte over len 16 add range -rot over len over swap mod swap pack3 [ drop [ dup len 1 eq? ] [ ] [ unpack3 [ [ over len over sub 10 ge? ] [ over over drop-n 10 take ] [ over over drop-n 1 take ] ifte 0 -1 pack2 [ [ unpack2 ] dip [ over 0 lt? ] [ [ null? ] [ drop drop dup ] [ drop ] ifte ] [ drop ] ifte [ inc ] dip pack2 ] fold second ] dip swap [ dup 0 ge? ] [ [ drop ] dip add swap drop quote ] [ drop 32 div swap 5 mul inc over add pick2 len mod swap pack3 ] ifte ] ifte ] fold head swap quote set-at ] fold ] [ ] ifte ] fold flatten` |
