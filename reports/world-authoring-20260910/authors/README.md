# Wave 1 raw (author dialects)

raw author outputs before the merge pass


## Verified

| source | verified | median witness length (`mark-28` ops) | median macro-form length |
|---|---|---|---|
| codearc | 34/34 (100%) | 175.5 | 15.0 |
| leetcode-hard | 2/5 (40%) | 1164.5 | 37.0 |
| leetcode-medium | 21/25 (84%) | 232 | 27 |
| mbpp | 35/36 (97%) | 143 | 14 |
| **all** | 92/100 (92%) | 171.5 | 16.0 |

Claimed-vs-verified mismatches: 0; missing ids: 0; duplicates: 0.

By author: a 25/26 (96%); b 24/26 (92%); c 20/24 (83%); d 23/24 (96%).

## Witness length against `stack_pbe_v4`

| bucket (ops) | this probe (verified) | v4 behaviours (train + eval, 220) |
|---|---|---|
| 1-5 | 3 | 15 |
| 6-10 | 5 | 14 |
| 11-20 | 3 | 34 |
| 21-40 | 3 | 37 |
| 41+ | 78 | 120 |
| median | 171.5 | 43.0 |

Exact matches with a v4 witness: 1 (mbpp-517). Same skeleton as a v4 witness: 4.

## Macros

222 macros defined across `macros*.txt`; median expansion ratio (`mark-28` ops per macro-form token) on verified programs: 11.0.

| macro | verified programs using it |
|---|---|
| `2` | 27 |
| `unpack2` | 19 |
| `range` | 19 |
| `pack2` | 18 |
| `inc` | 15 |
| `len` | 14 |
| `mod` | 13 |
| `sum` | 13 |
| `le?` | 12 |
| `max2` | 11 |
| `pick2` | 10 |
| `take` | 10 |
| `dec` | 10 |
| `range1` | 10 |
| `bool>int` | 10 |
| `pack3` | 9 |
| `4` | 8 |
| `head` | 8 |
| `zip` | 7 |
| `3` | 7 |

## Unsolved

| id | source | passed | note |
|---|---|---|---|
| leetcode-beautiful-arrangement | leetcode | 11/18 | why: descending-position assignment enumeration exceeds 100,000 fuel for n=9..15; best 11/18, including n<=8 and nonposi |
| leetcode-magical-string | leetcode | 22/64 | why: two-list queue generation exceeds 100,000 fuel on large n; best 22/64, preserving Python prefix slicing for nonposi |
| leetcode-minimum-increments-for-target-multiples-in-an-array | leetcode | 47/64 | best 47/64; why: exhaustive LCM allocations exceed 100,000 fuel as the number of targets grows; redundant target divisor |
| mbpp-769 | mbpp | 2/27 | why: computes both set differences, but preserves first-occurrence order rather than CPython hash-table iteration; exact |
| leetcode-closest-dessert-cost | leetcode | 56/64 | why: ternary DFS with attainable-interval pruning still exceeds 100,000 fuel on eight tests; best verified 56/64 |
| leetcode-number-of-squareful-arrays | leetcode | 41/43 | why: precomputed adjacency bitmasks and labelled permutation frontiers still exceed 100,000 fuel on two tests; best veri |
| leetcode-race-car | leetcode | 21/64 | why: reference dynamic programming exceeds 100,000 fuel on 43 tests; exact all-ones and power-of-two cases plus smaller  |
| leetcode-guess-number-higher-or-lower-ii | leetcode | 5/52 | best 5/52; shape: exact interval minimax DP; cubic candidate enumeration and immutable-list lookups exceed 100,000 fuel  |

## Every verified program

| id | len | macro form |
|---|---|---|
| mbpp-292 | 1 | `div` |
| codearc-95 | 1 | `mul` |
| mbpp-816 | 2 | `drop nil` |
| mbpp-257 | 6 | `swap pack2` |
| leetcode-most-expensive-item-that-can-not-be-bought | 8 | `dec swap dec mul dec` |
| mbpp-266 | 8 | `sq 4 mul` |
| codearc-421 | 8 | `2 mul inc swap sub` |
| codearc-997 | 9 | `neg over div neg mul` |
| mbpp-268 | 14 | `dup dec mul 6 mul inc` |
| codearc-305 | 15 | `[ [ dup ] dip mod add ] swap cons map` |
| mbpp-517 | 16 | `max` |
| mbpp-66 | 23 | `[ 0 ge? ] filter len` |
| codearc-146 | 35 | `dup max swap min add` |
| leetcode-construct-the-longest-new-string | 36 | `[ [ over eq? ] [ add ] [ min2 2 mul inc ] ifte ] dip add 2 mul` |
| mbpp-318 | 42 | `0 max2 dup 3 div over inc 3 div mul swap 2 add 3 div mul` |
| mbpp-375 | 45 | `over over div over mul [ pick2 over sub 2 mul pick2 gt? ] [ add swap drop ] [ -rot drop drop ] ifte` |
| leetcode-count-sorted-vowel-strings | 51 | `0 max2 inc dup inc over 2 add pick2 3 add mul mul mul 6 4 mul div` |
| mbpp-635 | 57 | `sort` |
| codearc-102 | 59 | `dup [ even? ] filter len kb-choose2 swap [ odd? ] filter len kb-choose2 add` |
| codearc-1069 | 60 | `ka-inversions` |
| codearc-56 | 61 | `kb-min-subarray` |
| leetcode-maximum-score-from-removing-stones | 66 | `pack3 [ 0 max2 ] map dup sum swap max over swap sub swap 2 div min2` |
| mbpp-273 | 76 | `zip [ unpack2 sub ] map` |
| mbpp-616 | 80 | `zip [ unpack2 mod ] map` |
| mbpp-489 | 94 | `swap 1 max2 take dup max count` |
| leetcode-max-chunks-to-make-sorted | 95 | `0 0 0 pack3 [ kd-chunk-step ] fold unpack3 drop swap drop` |
| codearc-147 | 97 | `[ dup 0 eq? ] [ ] [ digits head ] ifte` |
| codearc-950 | 107 | `[ dup 0 eq? ] [ ] [ digits dup product swap sum sub ] ifte` |
| mbpp-10 | 110 | `[ sort ] dip take` |
| mbpp-296 | 110 | `take ka-inversions` |
| mbpp-506 | 117 | `range swap [ swap sub ] swap cons map product` |
| leetcode-clumsy-factorial | 120 | `[ dup 2 le? ] [ ] [ [ dup 4 le? ] [ dup 2 mul swap 4 eq? bool>int sub ] [ dup 4 mod [ dup 0 eq? ] [ drop inc ] [ [ dup 3 eq? ] [ drop dec ] [ drop 2 add ] ifte ] ifte ] ifte ] ifte` |
| leetcode-sum-of-two-integers | 124 | `add kb-signed32` |
| leetcode-find-the-winner-of-the-circular-game | 131 | `[ range1 ] dip [ swap [ add ] dip mod ] swap cons 0 swap fold inc` |
| codearc-780 | 134 | `zip 2 take [ unpack2 2 mul swap sub ] map` |
| mbpp-158 | 135 | `[ over max [ take ] dip [ swap sub ] swap cons map ] dip [ over over [ mod 0 eq? bool>int ] swap cons map product 1 eq? ] [ [ div ] swap cons map sum ] [ drop drop -1 ] ifte` |
| codearc-368 | 137 | `[ over 0 lt? ] [ 2 add over div swap ] [ over div swap ] ifte [ mul ] swap cons swap range1 swap map` |
| leetcode-bulb-switcher-ii | 138 | `[ dup 0 lt? ] [ 0 ] [ [ dup 0 eq? ] [ 1 ] [ [ over 0 le? ] [ 1 ] [ [ over 1 eq? ] [ 2 ] [ [ over 2 eq? ] [ [ dup 1 eq? ] [ 3 ] [ 4 ] ifte ] [ [ dup 1 eq? ] [ 4 ] [ [ dup 2 eq? ] [ 7 ] [ 8 ] ifte ] ifte ] ifte ] ifte ] ifte ] ifte ] ifte [ drop drop ] dip` |
| mbpp-149 | 143 | `take nil [ kd-diff-one-best inc pack2 cons ] fold seconds 1 [ max2 ] fold` |
| codearc-868 | 146 | `[ ka-diff-pairs-step ] swap cons [ nil 0 pack2 ] dip fold unpack2 swap drop` |
| mbpp-60 | 158 | `take nil [ over over [ [ unpack2 drop ] dip sub abs 1 le? ] swap cons filter seconds 0 [ max2 ] fold inc pack2 cons ] fold seconds 0 [ max2 ] fold` |
| codearc-597 | 161 | `div dup range1 swap [ [ kb-sum-squares ] dip le? ] swap cons filter len` |
| codearc-960 | 163 | `[ dup odd? bool>int swap abs digits 3 contains mul 1 eq? ] filter sum` |
| mbpp-5 | 165 | `[ dup odd? ] [ drop 0 ] [ 2 div range 1 1 pack2 [ drop unpack2 over 4 mul swap sub swap pack2 ] fold unpack2 drop ] ifte` |
| codearc-196 | 167 | `dup len dup range swap [ over inc -rot swap sub mul inc 2 div ] swap cons map dot` |
| leetcode-moving-stones-until-consecutive | 169 | `pack3 sort unpack3 over sub -rot swap sub swap [ over add 2 le? ] [ drop drop 0 0 pack2 ] [ over over add 2 sub [ min2 3 lt? bool>int 2 swap sub ] dip pack2 ] ifte` |
| codearc-888 | 174 | `ka-complement` |
| codearc-935 | 177 | `swap kb-py-slice-start over swap drop-n zip [ unpack2 eq? bool>int ] map sum` |
| codearc-493 | 184 | `nil pack2 zeros16 swap [ drop unpack2 [ drop 10 ge? ] [ swap digit-product dup rot swap cons pack2 ] [ pack2 ] ifte ] fold unpack2 swap drop reverse` |
| mbpp-890 | 187 | `[ dup 0 lt? ] [ [ drop drop ] dip ] [ [ zip ] dip take [ unpack2 eq? bool>int ] map 0 1 pack2 [ [ unpack2 ] dip mul dup [ add ] dip pack2 ] fold head ] ifte` |
| codearc-94 | 188 | `abs swap abs gcd dup range1 swap [ swap mod 0 eq? ] swap cons filter sum` |
| mbpp-648 | 196 | `dup len range swap [ [ [ even? ] [ inc ] [ dec ] ifte ] dip swap nth ] swap cons map` |
| mbpp-144 | 200 | `dup [ take ] dip dup range swap [ swap 2 mul swap dec sub ] swap cons map dot` |
| codearc-488 | 202 | `[ dup 0 lt? ] [ drop drop 0 ] [ [ dup 0 eq? ] [ drop dup dedup swap [ swap count 1 gt? ] swap cons filter len ] [ [ dedup dup ] dip [ rot add contains ] swap cons swap cons map sum ] ifte ] ifte` |
| codearc-769 | 203 | `dup len dec range swap [ drop dup tail zip [ unpack2 add ] map ] fold sum` |
| codearc-850 | 203 | `[ dup null? ] [ drop 1 ] [ dup min 1 max2 range1 tail swap [ swap [ mod 0 eq? bool>int ] swap cons map product 1 eq? ] swap cons filter [ dup null? ] [ drop 1 ] [ head ] ifte ] ifte` |
| mbpp-782 | 209 | `dup len dup range -rot [ [ over nth ] dip swap [ kc-odd-weight ] dip mul ] swap cons swap cons map sum` |
| mbpp-845 | 217 | `[ dup 0 lt? ] [ drop 0 ] [ range1 product digits len ] ifte` |
| mbpp-566 | 221 | `[ dup 0 lt? ] [ abs digits [ neg 10 mod ] map sum ] [ digit-sum ] ifte` |
| leetcode-the-kth-factor-of-n | 232 | `[ dup range1 swap [ swap divides? ] swap cons filter ] dip [ dup 0 le? ] [ drop drop -1 ] [ dec [ over len over gt? ] [ nth ] [ drop drop -1 ] ifte ] ifte` |
| mbpp-485 | 234 | `[ sort ] dip take [ kc-palindrome? ] filter -1 [ max2 ] fold` |
| leetcode-sum-of-numbers-with-units-digit-k | 239 | `[ over 0 eq? ] [ drop drop 0 ] [ over 10 min2 range1 -rot [ rot mul sub dup 0 ge? bool>int swap 10 mod 0 eq? bool>int mul 1 eq? ] swap cons swap cons filter [ null? ] [ drop -1 ] [ head ] ifte ] ifte` |
| mbpp-48 | 248 | `dup 4 powers swap [ [ over ge? ] [ swap div 2 mod 0 eq? ] [ drop drop false ] ifte ] swap cons filter sum add` |
| mbpp-909 | 248 | `dup range swap 0 pack2 [ kd-prev-pal-step ] fold unpack2 drop` |
| mbpp-765 | 258 | `inc dup dup kc-log2 add kc-log2 add` |
| leetcode-water-bottles-ii | 265 | `over pick2 pick2 pack3 pick2 abs pick2 abs add 2 add 2 mul pack2 [ drop drop ] dip unpack2 range swap [ ka-water-step ] fold head` |
| codearc-509 | 265 | `digits dup len range swap [ swap drop-n uncons [ add ] swap cons map ] swap cons map nil [ cat ] fold` |
| codearc-1054 | 305 | `kc-decimal-digits dup len dec range swap [ drop dup tail zip [ unpack2 add inc 2 div ] map ] fold head` |
| leetcode-apply-operations-to-make-sum-of-array-greater-than-or-equal-to-k | 319 | `[ dup 0 le? ] [ ] [ 4 mul dup ka-isqrt dup sq pick2 lt? bool>int add swap drop 2 sub ] ifte` |
| leetcode-egg-drop-with-2-eggs-and-n-floors | 321 | `dup 8 mul 1 add ka-isqrt 1 sub 2 div dup over pick2 1 add mul 2 div pick3 lt? bool>int add [ drop drop ] dip` |
| codearc-143 | 330 | `uncons dup dup pack2 dup pack3 [ ka-prod-step ] fold head` |
| mbpp-416 | 334 | `range1 nil 0 cons [ over over 2 div nth pick2 pick2 3 div nth add pick2 pick2 4 div nth add max2 quote cat ] fold last` |
| codearc-461 | 358 | `[ dup 9 le? ] [ abs 10 mod ] [ 1 1 pack3 zeros16 swap [ drop unpack3 kc-digit-block pack3 ] fold unpack3 pick2 dec over mod [ rot dec swap div add kc-decimal-digits ] dip nth ] ifte` |
| mbpp-707 | 358 | `[ dup 0 lt? ] [ inc 2 div ] [ range1 [ 2 base-digit-sum ] map sum ] ifte` |
| mbpp-469 | 377 | `over len min2 range swap dup [ drop 0 ] map pack2 [ kb-stock-round ] fold kb-second last` |
| leetcode-mirror-reflection | 382 | `over over kc-gcd dup [ div 2 mod ] dip rot swap div 2 mod [ dup 1 eq? ] [ drop ] [ drop drop 2 ] ifte` |
| codearc-543 | 427 | `[ kd-transpose37 ] map sort [ kd-transpose37 ] map` |
| codearc-654 | 452 | `dup 2 mul range swap 0 pack2 [ kc-nearest-pal-step ] fold unpack2 swap drop` |
| leetcode-rotated-digits | 472 | `0 max2 digits dup nil 9 cons 8 cons 6 cons 5 cons 2 cons 1 cons 0 cons kb-rot-count swap nil 8 cons 1 cons 0 cons kb-rot-count sub` |
| codearc-610 | 495 | `dec range swap quote [ drop dup head kb-derivative cons ] fold reverse` |
| codearc-943 | 516 | `ka-signed-xor pows2 [ [ swap div 2 mod ] swap cons ] dip swap map sum` |
| leetcode-minimum-number-of-operations-to-make-x-and-y-equal | 521 | `[ over over le? ] [ swap sub ] [ over over sub swap [ 0 pack2 quote ] dip2 pack3 kc-op-search unpack3 drop swap drop ] ifte` |
| leetcode-2-keys-keyboard | 593 | `[ dup 0 le? ] [ ] [ dup kc-log2 range swap 0 pack2 [ drop kc-factor-sum-step ] fold unpack2 swap drop ] ifte` |
| mbpp-195 | 649 | `pick2 pick2 0 pick3 1 sub 0 1 sub pack3 pack3 over pack2 [ drop drop drop ] dip unpack2 range swap [ ka-first-step ] fold last last` |
| codearc-990 | 677 | `[ 10 inc max2 nil pack2 ] dip range swap [ ka-pal-step ] fold unpack2 swap drop reverse` |
| codearc-569 | 811 | `[ over null? ] [ swap drop ] [ [ dup null? ] [ drop ] [ [ kc-concat-number ] dip kc-concat-number add kc-signed-digits ] ifte ] ifte` |
| leetcode-super-egg-drop | 1019 | `[ dup 1 lt? ] [ drop drop 0 ] [ [ over 1 eq? ] [ swap drop ] [ [ over 0 le? ] [ swap drop ka-bitlen ] [ dup pick2 pick2 ka-bitlen min2 1 add range [ drop 0 ] map 0 pack3 over 2 mul ka-isqrt 2 add pack2 [ drop drop ] dip unpack2 range swap [ ka-egg-step ] fold last ] ifte ] ifte ] ifte` |
| leetcode-the-earliest-and-latest-rounds-where-players-compete | 1310 | `kd-tournament` |
| leetcode-number-of-self-divisible-permutations | 1463 | `[ dup 0 le? ] [ drop 1 ] [ dup kd-perm-table swap [ odd? ] [ drop kd-perm-odd ] [ drop unpack2 swap drop unpack2 1000 div div sq ] ifte ] ifte` |
| codearc-1022 | 1562 | `[ over over gt? ] [ 0 ] [ [ dup 0 lt? ] [ over abs ka-nofive over abs 1 sub ka-nofive sub ] [ [ over 0 lt? ] [ over abs ka-nofive over ka-nofive add 1 sub ] [ [ over 0 eq? ] [ dup ka-nofive ] [ dup ka-nofive pick2 1 sub ka-nofive sub ] ifte ] ifte ] ifte ] ifte [ drop drop ] dip` |
| leetcode-maximum-points-in-an-archery-competition | 1762 | `kd-archery` |
| mbpp-854 | 1967 | `dup len 2 div range reverse swap [ ka-heap-sift ] fold` |
