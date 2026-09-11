# world-authoring-20260911-w3 raw (author dialects)

raw author outputs before the merge pass


## Verified

| source | verified | median witness length (`mark-28` ops) | median macro-form length |
|---|---|---|---|
| codearc | 49/50 (98%) | 221 | 17 |
| mbpp | 50/50 (100%) | 121.5 | 8.5 |
| **all** | 99/100 (99%) | 151 | 13 |

Claimed-vs-verified mismatches: 0; missing ids: 0; duplicates: 0.

By author: a 25/26 (96%); b 26/26 (100%); c 24/24 (100%); d 24/24 (100%).

## Witness length against `stack_pbe_v4`

| bucket (ops) | this probe (verified) | v4 behaviours (train + eval, 220) |
|---|---|---|
| 1-5 | 7 | 15 |
| 6-10 | 3 | 14 |
| 11-20 | 5 | 34 |
| 21-40 | 6 | 37 |
| 41+ | 78 | 120 |
| median | 151 | 43.0 |

Exact matches with a v4 witness: 2 (mbpp-972, codearc-22). Same skeleton as a v4 witness: 4.

## Macros

223 macros defined across `macros*.txt`; median expansion ratio (`mark-28` ops per macro-form token) on verified programs: 11.8.

| macro | verified programs using it |
|---|---|
| `range` | 21 |
| `2` | 20 |
| `inc` | 17 |
| `head` | 17 |
| `len` | 17 |
| `sum` | 16 |
| `pack2` | 14 |
| `unpack2` | 13 |
| `take` | 11 |
| `dec` | 10 |
| `second` | 9 |
| `3` | 8 |
| `mod` | 7 |
| `sort` | 7 |
| `gt?` | 7 |
| `pow` | 7 |
| `le?` | 7 |
| `zip` | 7 |
| `pack3` | 7 |
| `max2` | 6 |

## Unsolved

| id | source | passed | note |
|---|---|---|---|
| codearc-92 | codearc | 2/16 | why: factor-pair divisor sums exceed 100,000 fuel, already at limit=100; best 2/16 (nonpositive limits). Includes both m |

## Every verified program

| id | len | macro form |
|---|---|---|
| mbpp-458 | 1 | `mul` |
| codearc-522 | 1 | `mul` |
| mbpp-972 | 1 | `cat` |
| codearc-22 | 1 | `add` |
| codearc-1062 | 4 | `2 div` |
| mbpp-882 | 5 | `mul 2 mul` |
| mbpp-112 | 5 | `add 2 mul` |
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
| mbpp-347 | 43 | `[ over over gt? ] [ swap ] [ ] ifte over 3 mul over sub inc rot drop swap dup inc mul mul 6 div` |
| codearc-180 | 43 | `reverse 0 0 pack2 [ [ unpack2 ] dip swap inc mul dup rot add swap pack2 ] fold head` |
| codearc-290 | 56 | `[ dup len 2 lt? ] [ drop -1 ] [ dup len over max mul swap sum sub ] ifte` |
| mbpp-940 | 57 | `sort` |
| codearc-121 | 60 | `inversions` |
| codearc-1046 | 68 | `dec dup 3 sum-multiples over 5 sum-multiples add swap 3 5 mul sum-multiples sub` |
| codearc-477 | 69 | `abs [ over over mod 2 mul over eq? ] [ drop ] [ over over mod 2 mul over gt? bool>int [ over over div ] dip add mul swap drop ] ifte` |
| mbpp-379 | 71 | `kc-cuboid` |
| mbpp-282 | 76 | `zip [ unpack2 sub ] map` |
| mbpp-825 | 79 | `ka-indices` |
| mbpp-351 | 87 | `[ take dup ] dip [ [ swap count ] dip eq? ] swap cons swap cons filter ka-first-or-minus-one` |
| mbpp-384 | 94 | `swap 1 max2 take dup min count` |
| mbpp-558 | 103 | `sub abs digit-sum` |
| mbpp-687 | 107 | `gcd` |
| mbpp-496 | 110 | `[ sort ] dip take` |
| codearc-166 | 112 | `reverse nil [ kb-div-chain-step ] fold seconds max` |
| mbpp-22 | 114 | `kc-first-duplicate [ dup null? ] [ drop -1 ] [ head ] ifte` |
| codearc-962 | 118 | `over over gcd [ add 2 sub ] dip add` |
| mbpp-93 | 120 | `pow` |
| mbpp-340 | 123 | `[ 0 gt? ] filter sort 3 take sum` |
| codearc-538 | 124 | `over over gcd [ div ] swap cons [ swap pack2 ] dip map` |
| mbpp-527 | 125 | `[ take ] dip [ ka-pair-count-step ] swap cons [ nil 0 pack2 ] dip fold second` |
| codearc-1050 | 129 | `kc-elevator` |
| codearc-459 | 136 | `[ [ dup ] dip pow swap sub ] swap cons map sum` |
| mbpp-389 | 137 | `range 2 1 pack2 [ drop unpack2 dup rot add pack2 ] fold head` |
| mbpp-498 | 138 | `[ over over mod 0 eq? ] [ swap drop ] [ [ dup 0 lt? ] [ drop drop 1 ] [ gcd ] ifte ] ifte` |
| codearc-1019 | 144 | `pack3 sort unpack3 [ [ add ] dip le? ] [ drop drop drop 0 ] [ sq [ sq [ sq ] dip add ] dip [ eq? ] [ drop drop 2 ] [ [ gt? ] [ drop drop 1 ] [ drop drop 3 ] ifte ] ifte ] ifte` |
| mbpp-308 | 151 | `[ [ swap [ mul ] swap cons map ] swap cons map flatten sort reverse ] dip ka-py-take` |
| codearc-642 | 151 | `[ div ] swap cons map dup -1 [ max2 ] fold inc range swap [ swap count ] swap cons map` |
| codearc-402 | 154 | `[ over 0 le? ] [ drop drop drop 1 ] [ [ pow ] dip mod ] ifte` |
| codearc-150 | 167 | `take max-subarray-sum 0 max2` |
| mbpp-559 | 167 | `take max-subarray-sum 0 max2` |
| codearc-270 | 178 | `[ dup 0 eq? ] [ drop drop 1 ] [ dec 4 mod inc [ 10 mod ] dip pow 10 mod ] ifte` |
| mbpp-295 | 179 | `dup range 2 drop-n swap [ swap divides? ] swap cons filter sum inc` |
| mbpp-516 | 186 | `dup max ka-radix-modulus [ over swap mod swap pack2 ] swap cons map ka-sort-key seconds` |
| mbpp-256 | 193 | `kd-primes-below len` |
| mbpp-649 | 199 | `kc-irange swap [ swap at ] swap cons map sum` |
| mbpp-483 | 207 | `kc-factorial-divisible` |
| codearc-736 | 210 | `[ over null? ] [ drop drop 0 ] [ over over at pack3 dup head over third [ lt? ] swap cons filter sum swap dup head over second inc take over third count swap third mul add ] ifte` |
| mbpp-346 | 221 | `[ range nil 1 cons [ drop reverse prefix-sums 0 cons ] fold ] dip nth` |
| codearc-1082 | 221 | `[ dup 0 le? ] [ drop drop nil ] [ [ 0 max2 ] dip kc-distribute-positive ] ifte` |
| codearc-302 | 225 | `dup [ even? ] filter sort reverse over [ odd? ] filter sort nil pack3 [ [ unpack3 ] dip [ even? ] [ drop [ [ uncons ] dip ] dip rot cons ] [ drop [ uncons ] dip swap cons ] ifte pack3 ] fold third reverse` |
| codearc-752 | 230 | `dup len [ [ 0 eq? not ] filter nil 0 pack2 [ kb-merge-step ] fold unpack2 [ dup 0 eq? ] [ drop ] [ cons ] ifte reverse ] dip over len sub range [ drop 0 ] map cat` |
| codearc-28 | 239 | `dup len range zip [ unpack2 mul ] map 1 drop-n` |
| codearc-828 | 241 | `dup digits len dup 10 swap pow dec 9 div [ swap inc mul ] dip sub` |
| codearc-342 | 249 | `[ inc range ] dip [ kc-power-sum ] swap cons map sum` |
| codearc-719 | 251 | `dup len range zip 1 drop-n [ unpack2 divides? ] filter [ head ] map` |
| mbpp-348 | 261 | `2 div [ dup 0 lt? ] [ inc 1 swap div ] [ dup range1 over [ add ] swap cons map product over range1 product div swap inc div ] ifte` |
| codearc-630 | 277 | `[ range-between ] dip [ [ over 9 mod over 9 mod eq? ] [ [ kd-digit-sum ] dip eq? ] [ drop drop false ] ifte ] swap cons filter dup head swap last pack2` |
| codearc-871 | 297 | `[ dup len range swap zip ] dip over -rot [ kb-twosum-row ] swap cons swap cons map [ null? not ] filter head` |
| codearc-699 | 299 | `[ dup 0 lt? ] [ ka-odometer neg ] [ ka-odometer ] ifte` |
| codearc-557 | 315 | `dup len range zip dup [ head ] map dedup swap [ swap [ [ head ] dip eq? ] swap cons filter seconds dup max swap min sub ] swap cons map max` |
| codearc-471 | 316 | `kd-primes-below swap [ ge? ] swap cons filter [ kd-happy-prime-end ] filter len` |
| codearc-843 | 319 | `[ dup 0 eq? ] [ drop 4 ] [ dup pows2 swap [ swap div odd? ] swap cons filter head over over add pick2 over bxor pick2 div 4 div add [ drop drop ] dip ] ifte` |
| mbpp-903 | 376 | `range1 [ dup bitlen swap popcount sub ] map sum` |
| codearc-696 | 427 | `swap range1 10 1 add drop-n swap [ [ decimal-digits dup tail zip ] dip [ [ unpack2 swap sub ] dip over 0 gt? bool>int [ le? bool>int ] dip mul 1 eq? not ] swap cons filter null? ] swap cons filter len` |
| mbpp-661 | 434 | `take [ dup len 2 le? ] [ sum ] [ dup 3 drop-n swap dup head over 2 take sum pick2 3 take dup sum swap min sub pick3 third pack4 swap drop [ kb-dp3-step ] fold third ] ifte` |
| codearc-980 | 482 | `decimal-digits dup len range swap 0 pack2 quote nil swap pack2 [ ka-perm-step ] fold head dedup sum` |
| codearc-451 | 489 | `[ over null? bool>int over null? bool>int add 0 gt? ] [ drop drop nil ] [ over len over len add dec range -rot [ ka-poly-coeff ] swap cons swap cons map ] ifte` |
| codearc-10 | 525 | `dup len range swap dup ka-every-third sort [ ka-third-element ] swap cons swap cons map` |
| codearc-711 | 547 | `kc-first-repeat` |
| codearc-483 | 554 | `kc-riders` |
| mbpp-155 | 569 | `[ dup 0 gt? ] [ dup bitlen range [ odd? ] filter [ pow2 ] map sum bxor ] [ ] ifte` |
| mbpp-926 | 571 | `kc-rencontres` |
| mbpp-531 | 581 | `[ take ] dip range swap nil 0 cons pack2 [ drop unpack2 over over kc-coin-next cons pack2 ] fold second head` |
| codearc-921 | 613 | `dup 2 div range1 [ 2 ge? ] filter [ prime? ] filter [ range1 3 drop-n ] dip rot [ kb-factor-match? ] swap cons swap cons filter` |
| codearc-175 | 630 | `[ ka-wind-root ] dip dup ka-6215 mul ka-131200 add ka-scale mul [ ka-3965 mul ka-113700 sub mul ] dip add ka-scale 100 100 mul mul ka-round-ratio` |
| mbpp-103 | 659 | `[ over 0 eq? ] [ drop drop 0 ] [ [ over over le? ] [ drop drop 0 ] [ pack2 dup second 2 add range swap unpack2 [ pack3 dup third inc over head sub over second pow over second inc pick2 head kd-binomial mul swap head even? bool>int 2 mul dec mul ] swap cons swap cons map sum ] ifte ] ifte` |
| codearc-292 | 726 | `decimal-digits dup len 2 sub range1 swap [ swap kc-split-inner ] swap cons map max` |
| mbpp-843 | 735 | `[ over 1 le? ] [ drop drop 1 ] [ over over len mul range -rot dup len range swap zip [ unpack2 dup quote pack3 ] map 1 1 pack3 swap [ kb-ugly-step ] swap cons fold second ] ifte` |
| codearc-578 | 740 | `pack2 dup dup second inc range swap unpack2 [ pack3 dup head over second pow over third pick2 head kd-binomial mul swap dup third swap head sub even? bool>int 2 mul dec mul ] swap cons swap cons map sum swap second kd-factorial div` |
| codearc-323 | 778 | `pack2 dup second 9 div inc range swap unpack2 [ kd-comfort-group ] swap cons swap cons map sum` |
| mbpp-633 | 786 | `take nil 0 pack2 [ ka-pair-xor-step ] fold second` |
| mbpp-203 | 827 | `sxor 0 max2 popcount` |
| mbpp-934 | 912 | `pack2 dup unpack2 min2 inc range swap unpack2 [ pack3 dup second over head kd-binomial over third pick2 head kd-binomial mul swap head pow2 mul ] swap cons swap cons map sum` |
| codearc-420 | 964 | `[ dup 1000 lt? ] [ range-between [ dup kb-digit-under1000 eq? ] filter ] [ range-between [ dup kb-digit-fast-sum eq? ] filter ] ifte` |
| mbpp-815 | 1316 | `dup range [ dec 0 0 rot pack4 ] dip swap [ drop kc-dnf-step ] fold head` |
| mbpp-560 | 2578 | `kc-set-union` |
