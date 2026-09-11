# world-authoring-20260911-w4 raw (author dialects)

raw author outputs before the merge pass


## Verified

| source | verified | median witness length (`mark-28` ops) | median macro-form length |
|---|---|---|---|
| codearc | 50/50 (100%) | 135.0 | 17.5 |
| mbpp | 49/50 (98%) | 115 | 8 |
| **all** | 99/100 (99%) | 127 | 14 |

Claimed-vs-verified mismatches: 1; missing ids: 0; duplicates: 0.

By author: a 26/26 (100%); b 26/26 (100%); c 23/24 (96%); d 24/24 (100%).

## Witness length against `stack_pbe_v4`

| bucket (ops) | this probe (verified) | v4 behaviours (train + eval, 220) |
|---|---|---|
| 1-5 | 5 | 15 |
| 6-10 | 1 | 14 |
| 11-20 | 14 | 34 |
| 21-40 | 8 | 37 |
| 41+ | 71 | 120 |
| median | 127 | 43.0 |

Exact matches with a v4 witness: 9 (mbpp-404, mbpp-929, mbpp-924, mbpp-443, mbpp-8, mbpp-514, codearc-111, mbpp-250, codearc-128). Same skeleton as a v4 witness: 11.

## Macros

193 macros defined across `macros*.txt`; median expansion ratio (`mark-28` ops per macro-form token) on verified programs: 12.0.

| macro | verified programs using it |
|---|---|
| `len` | 20 |
| `range` | 15 |
| `head` | 13 |
| `unpack2` | 13 |
| `dec` | 12 |
| `inc` | 12 |
| `sum` | 12 |
| `pack2` | 11 |
| `gt?` | 11 |
| `take` | 10 |
| `count` | 10 |
| `max2` | 9 |
| `2` | 9 |
| `-1` | 7 |
| `reverse` | 7 |
| `sort` | 7 |
| `3` | 7 |
| `max` | 7 |
| `neg` | 6 |
| `min` | 6 |

## Unsolved

| id | source | passed | note |
|---|---|---|---|
| mbpp-847 | mbpp | 0/27 | shape: identity; machine lists are immutable |

## Every verified program

| id | len | macro form |
|---|---|---|
| mbpp-127 | 1 | `mul` |
| mbpp-789 | 1 | `mul` |
| mbpp-35 | 4 | `dup inc mul` |
| mbpp-8 | 5 | `[ sq ] map` |
| mbpp-514 | 5 | `sum` |
| mbpp-447 | 7 | `[ dup sq mul ] map` |
| mbpp-404 | 12 | `min2` |
| mbpp-924 | 12 | `max2` |
| codearc-288 | 12 | `[ divides? ] swap cons filter` |
| codearc-128 | 12 | `min2` |
| mbpp-929 | 13 | `count` |
| mbpp-629 | 13 | `[ even? ] filter` |
| codearc-111 | 13 | `count` |
| mbpp-250 | 13 | `count` |
| mbpp-435 | 14 | `10 mod` |
| mbpp-279 | 14 | `dup 4 mul 3 sub mul` |
| mbpp-443 | 16 | `min` |
| codearc-1063 | 16 | `add swap sub 0 max2` |
| mbpp-267 | 18 | `dup sq 4 mul dec mul 3 div` |
| codearc-918 | 18 | `[ even? ] filter sum` |
| codearc-328 | 23 | `pack3 [ kc-triangle ] map product` |
| mbpp-632 | 24 | `dup [ 0 eq? not ] filter swap [ 0 eq? ] filter cat` |
| mbpp-672 | 24 | `max2 max2` |
| codearc-49 | 26 | `2 div 2 mul [ over over gt? ] [ drop drop -1 ] [ swap drop ] ifte` |
| codearc-446 | 26 | `[ dup 0 eq? ] [ ] [ dec 9 mod inc ] ifte` |
| mbpp-42 | 29 | `drop dup [ swap count 1 gt? ] swap cons filter sum` |
| codearc-535 | 30 | `dup [ swap count even? ] swap cons filter` |
| codearc-830 | 36 | `dup [ swap count ] swap cons map 0 [ max2 ] fold` |
| mbpp-236 | 47 | `[ over over lt? ] [ drop drop -1 ] [ over over sub ka-tri [ 2 mul sub ka-tri ] dip add ] ifte` |
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
| mbpp-232 | 115 | `[ sort reverse ] dip take` |
| codearc-260 | 115 | `swap [ gcd 1 eq? ] swap cons filter` |
| codearc-636 | 118 | `[ range1 ] dip [ mul ] swap cons map` |
| mbpp-21 | 118 | `[ range1 ] dip [ mul ] swap cons map` |
| mbpp-856 | 118 | `take reverse uncons neg inc 0 pack2 [ [ unpack2 ] dip [ dup 0 eq? ] [ drop [ inc ] dip ] [ [ dup 1 eq? ] [ drop over add ] [ drop ] ifte ] ifte pack2 ] fold second` |
| codearc-294 | 119 | `dup product swap dup dedup over [ swap count inc ] swap cons map product dec swap min pick2 over div pack4` |
| mbpp-767 | 122 | `[ take ] dip [ kd-pair-step ] swap cons nil 0 pack2 swap fold second` |
| codearc-167 | 125 | `[ pow ] swap cons map` |
| mbpp-503 | 127 | `kc-adjacent [ sum ] map` |
| mbpp-690 | 128 | `kb-adjacent [ unpack2 mul ] map` |
| codearc-333 | 128 | `dup [ odd? ] filter sort nil pack2 [ kc-sort-odd-step ] fold second reverse` |
| codearc-977 | 132 | `[ 2 swap pow ] map sum` |
| codearc-106 | 132 | `popcount` |
| mbpp-345 | 134 | `ka-differences [ neg ] map` |
| codearc-856 | 134 | `[ sort ] dip dec at` |
| codearc-466 | 136 | `swap [ [ over over ge? ] [ add ] [ over swap gcd add ] ifte ] fold` |
| codearc-643 | 136 | `dup len [ swap count ] dip dup rot sub 100 mul swap kd-round-even` |
| mbpp-586 | 143 | `[ drop ] dip slice-index over over drop-n [ take ] dip swap cat` |
| codearc-680 | 147 | `kb-adjacent [ unpack2 swap sub 0 max2 ] map sum` |
| codearc-873 | 162 | `0 nil pack2 [ [ unpack2 ] dip kd-run-step pack2 ] fold unpack2 kd-run-sum add` |
| codearc-90 | 163 | `dup head 0 lt? bool>int 2 mul 1 swap sub [ [ abs decimal-digits ] map flatten from-digits ] dip mul` |
| mbpp-328 | 174 | `[ over swap slice-index drop-n ] dip [ swap ] dip slice-index take cat` |
| mbpp-36 | 181 | `[ dup 1 eq? ] [ drop [ 10 mul ] dip div ] [ dec 10 swap pow rot mul over mod 10 mul swap div ] ifte` |
| mbpp-685 | 198 | `inc primes-below sum` |
| codearc-532 | 202 | `sort kc-adjacent [ unpack2 swap sub ] map max` |
| codearc-59 | 203 | `dup len range [ even? ] filter swap [ swap nth ] swap cons map [ odd? ] filter sum` |
| codearc-53 | 210 | `[ [ dup 0 lt? ] [ neg digits uncons neg swap sum add ] [ digit-sum ] ifte 0 gt? ] filter len` |
| mbpp-702 | 212 | `[ [ sort ] dip ] dip [ dup [ dec take ] dip ] dip swap [ ka-max-window ] dip swap sub` |
| codearc-676 | 217 | `kd-indices [ dup len 2 eq? ] [ unpack2 swap sub inc ] [ drop 0 ] ifte` |
| mbpp-448 | 234 | `[ dup 0 eq? ] [ drop 3 ] [ [ dup 1 eq? ] [ drop 3 ] [ 2 sub range 3 0 2 5 pack4 [ kb-perrin-step ] fold fourth ] ifte ] ifte` |
| codearc-4 | 238 | `[ dup 0 eq? ] [ drop ] [ [ dup 0 lt? ] [ gcd neg ] [ gcd ] ifte ] ifte` |
| mbpp-943 | 253 | `over len over len add range [ nil pack3 ] dip swap [ ka-merge-step ] fold third reverse` |
| codearc-1032 | 265 | `[ prefix-sums dup len range zip ] dip over len [ [ swap head lt? ] swap cons filter seconds ] dip quote cat head` |
| codearc-1102 | 277 | `dup len range swap nil pack2 [ drop unpack2 over last cons [ ka-differences ] dip pack2 ] fold second` |
| mbpp-412 | 278 | `dup len range swap [ kc-remove-odd-step ] fold` |
| mbpp-116 | 286 | `[ dup head 0 lt? ] [ kc-join-ints neg ] [ kc-join-ints ] ifte` |
| codearc-914 | 305 | `10 3 add div inc range [ 10 3 add mul digits dup 4 contains swap 7 contains add 0 eq? ] filter len` |
| codearc-1042 | 306 | `swap 6 mul sub 5 add [ dup 5 lt? ] [ 5 swap div log2 neg ] [ 5 div log2 ] ifte` |
| codearc-475 | 330 | `dup max dec [ dup 1 lt? ] [ drop drop -1 ] [ swap dup head [ sub ] swap cons map 0 [ gcd ] fold [ dup 0 eq? ] [ drop ] [ [ over over ge? ] [ swap drop ] [ swap range1 swap [ swap divides? ] swap cons filter max ] ifte ] ifte ] ifte` |
| mbpp-802 | 341 | `take dup len range 1 drop-n swap [ swap over over nth [ dec nth ] dip gt? ] swap cons filter 0 quote cat head` |
| codearc-67 | 354 | `kc-adjacent [ unpack2 gt? bool>int ] map dup len range1 zip [ head 1 eq? ] filter seconds -1 quote swap cat last` |
| codearc-734 | 357 | `decimal-digits dup sum swap len dup dec factorial swap range [ drop 1 ] map from-digits mul mul` |
| mbpp-275 | 360 | `[ take ] dip [ kc-ceil-div ] swap cons map dup len range1 zip [ head -1 gt? ] filter [ dup null? ] [ drop 0 ] [ sort-by-head last second ] ifte` |
| codearc-306 | 421 | `[ dup len range kb-index-pairs swap ] dip [ [ kb-pair-total ] dip eq? ] swap cons swap cons filter nil [ kb-select-pair ] fold sum` |
| codearc-26 | 426 | `dup range 2 drop-n swap [ swap divides? ] swap cons filter [ prime? ] filter max` |
| codearc-1106 | 441 | `nil -1 pack3 zeros128 swap [ ka-happy-step ] fold third` |
| codearc-663 | 451 | `[ kc-power-ratio ] dip kc-power-ratio reverse zip [ product ] map unpack2 kc-reverse-compare` |
| mbpp-225 | 469 | `over over sub abs inc range [ pack3 ] dip swap [ ka-find-min-step ] fold unpack3 swap drop at` |
| mbpp-638 | 570 | `kb-wind-chill` |
| mbpp-229 | 582 | `range [ 0 pack2 ] dip swap [ kb-partition-step ] fold head` |
| codearc-235 | 584 | `decimal-digits dup len range swap 0 pack2 quote [ drop [ dup kd-perm-children swap cons ] map flatten dedup ] fold seconds dedup [ dup 0 gt? bool>int swap 3 divides? bool>int mul 1 eq? ] filter dup len swap max pack2` |
| codearc-691 | 716 | `dup isqrt range1 1 drop-n [ [ 4 max2 ] dip range-between ] dip [ over swap kb-factor-sum [ over over eq? ] [ drop drop false ] [ divides? ] ifte ] swap cons filter` |
| mbpp-71 | 828 | `dup len dup range -rot pack2 [ ka-comb-step ] fold head` |
| codearc-788 | 854 | `[ ge? ] [ drop drop drop nil ] [ pick2 sub [ 2 max2 ] dip range-between swap [ kd-gap-step ] swap cons nil swap fold ] ifte` |
| mbpp-550 | 927 | `pack3 zeros16 dup cat swap [ drop [ dup len 1 eq? ] [ ] [ kd-max-search ] ifte ] fold head` |
| codearc-229 | 1198 | `[ gt? ] [ drop drop zeros8 0 cons 0 cons ] [ [ dec kd-digit-prefix ] dip kd-digit-prefix swap zip [ unpack2 sub ] map ] ifte` |
| mbpp-735 | 1840 | `[ dup 1 eq? ] [ ] [ [ dup 0 lt? ] [ -1 dec sxor ] [ dup bitlen dec 0 max2 pow2 dec 1 bxor sxor ] ifte ] ifte` |
| mbpp-2 | 4136 | `[ dedup ] dip dedup [ over len over len lt? ] [ swap ] [ ] ifte kd-set-order over [ swap contains 1 eq? ] swap cons filter swap drop kd-set-order` |
