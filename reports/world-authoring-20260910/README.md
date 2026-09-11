# world-authoring-500 (#127), wave 1: 100 MBPP / CodeARC / LeetCode-medium+hard behaviours authored by codex exec

Behaviours sampled by `experiments/world_behaviours/extract.py` (seed 17, wider pool, hypothesis-breaking sampler on every source); programs authored by four `codex exec` agents (gpt-6-astra, high effort) in macro form and migrated to the canonical library; every program re-verified on every test by `report.py`.


## Verified

| source | verified | median witness length (`mark-28` ops) | median macro-form length |
|---|---|---|---|
| codearc | 34/34 (100%) | 175.5 | 24.5 |
| leetcode-hard | 2/5 (40%) | 1080.0 | 242.0 |
| leetcode-medium | 20/25 (80%) | 147.0 | 37.0 |
| mbpp | 35/36 (97%) | 143 | 16 |
| **all** | 91/100 (91%) | 163 | 24 |

Claimed-vs-verified mismatches: 0; missing ids: 0; duplicates: 0.

By author: canonical 91/100 (91%).

## Witness length against `stack_pbe_v4`

| bucket (ops) | this probe (verified) | v4 behaviours (train + eval, 220) |
|---|---|---|
| 1-5 | 3 | 15 |
| 6-10 | 5 | 14 |
| 11-20 | 3 | 34 |
| 21-40 | 3 | 37 |
| 41+ | 77 | 120 |
| median | 163 | 43.0 |

Exact matches with a v4 witness: 1 (mbpp-517). Same skeleton as a v4 witness: 4.

## Macros

110 macros defined across `macros*.txt`; median expansion ratio (`mark-28` ops per macro-form token) on verified programs: 5.5.

| macro | verified programs using it |
|---|---|
| `2` | 35 |
| `unpack2` | 25 |
| `inc` | 24 |
| `pack2` | 23 |
| `range` | 21 |
| `len` | 20 |
| `pick2` | 20 |
| `max2` | 19 |
| `mod` | 18 |
| `bool>int` | 17 |
| `head` | 17 |
| `dec` | 15 |
| `le?` | 15 |
| `sum` | 15 |
| `abs` | 14 |
| `pick3` | 14 |
| `range1` | 14 |
| `gt?` | 13 |
| `pack3` | 12 |
| `take` | 12 |

## Unsolved

| id | source | passed | note |
|---|---|---|---|
| leetcode-beautiful-arrangement | leetcode | 11/18 | why: descending-position assignment enumeration exceeds 100,000 fuel for n=9..15; best 11/18, including n<=8 and nonposi |
| leetcode-closest-dessert-cost | leetcode | 56/64 | why: ternary DFS with attainable-interval pruning still exceeds 100,000 fuel on eight tests; best verified 56/64 |
| leetcode-guess-number-higher-or-lower-ii | leetcode | 5/52 | best 5/52; shape: exact interval minimax DP; cubic candidate enumeration and immutable-list lookups exceed 100,000 fuel  |
| leetcode-magical-string | leetcode | 22/64 | why: two-list queue generation exceeds 100,000 fuel on large n; best 22/64, preserving Python prefix slicing for nonposi |
| leetcode-minimum-increments-for-target-multiples-in-an-array | leetcode | 47/64 | best 47/64; why: exhaustive LCM allocations exceed 100,000 fuel as the number of targets grows; redundant target divisor |
| leetcode-number-of-self-divisible-permutations | leetcode | 12/14 | verified 14/14 with the probe-era recursive gcd (105 fuel, valid only for Euclid chains of <= 7 steps); the canonical re |
| leetcode-number-of-squareful-arrays | leetcode | 41/43 | why: precomputed adjacency bitmasks and labelled permutation frontiers still exceed 100,000 fuel on two tests; best veri |
| leetcode-race-car | leetcode | 21/64 | why: reference dynamic programming exceeds 100,000 fuel on 43 tests; exact all-ones and power-of-two cases plus smaller  |
| mbpp-769 | mbpp | 2/27 | why: computes both set differences, but preserves first-occurrence order rather than CPython hash-table iteration; exact |

## Every verified program

| id | len | macro form |
|---|---|---|
| codearc-95 | 1 | `mul` |
| mbpp-292 | 1 | `div` |
| mbpp-816 | 2 | `drop nil` |
| mbpp-257 | 6 | `swap pack2` |
| codearc-421 | 8 | `2 mul inc swap sub` |
| leetcode-most-expensive-item-that-can-not-be-bought | 8 | `dec swap dec mul dec` |
| mbpp-266 | 8 | `sq 4 mul` |
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
| codearc-102 | 59 | `dup [ even? ] filter len dup dec mul 2 div swap [ odd? ] filter len dup dec mul 2 div add` |
| codearc-1069 | 60 | `inversions` |
| codearc-56 | 61 | `dup head 0 pack2 [ [ unpack2 ] dip add swap over min2 swap 0 min2 pack2 ] fold unpack2 drop` |
| leetcode-maximum-score-from-removing-stones | 66 | `pack3 [ 0 max2 ] map dup sum swap max over swap sub swap 2 div min2` |
| mbpp-273 | 76 | `zip [ unpack2 sub ] map` |
| mbpp-616 | 80 | `zip [ unpack2 mod ] map` |
| mbpp-489 | 94 | `swap 1 max2 take dup max count` |
| leetcode-max-chunks-to-make-sorted | 95 | `0 0 0 pack3 [ [ unpack3 ] dip max2 dup pick3 eq? bool>int [ swap ] dip add rot inc rot [ swap ] dip pack3 ] fold unpack3 drop swap drop` |
| codearc-147 | 97 | `[ dup 0 eq? ] [ ] [ digits head ] ifte` |
| codearc-950 | 107 | `[ dup 0 eq? ] [ ] [ digits dup product swap sum sub ] ifte` |
| mbpp-10 | 110 | `[ sort ] dip take` |
| mbpp-296 | 110 | `take inversions` |
| mbpp-506 | 117 | `range swap [ swap sub ] swap cons map product` |
| leetcode-clumsy-factorial | 120 | `[ dup 2 le? ] [ ] [ [ dup 4 le? ] [ dup 2 mul swap 4 eq? bool>int sub ] [ dup 4 mod [ dup 0 eq? ] [ drop inc ] [ [ dup 3 eq? ] [ drop dec ] [ drop 2 add ] ifte ] ifte ] ifte ] ifte` |
| leetcode-sum-of-two-integers | 124 | `add pows2 last dup rot add over 2 mul mod swap sub` |
| leetcode-find-the-winner-of-the-circular-game | 131 | `[ range1 ] dip [ swap [ add ] dip mod ] swap cons 0 swap fold inc` |
| codearc-780 | 134 | `zip 2 take [ unpack2 2 mul swap sub ] map` |
| mbpp-158 | 135 | `[ over max [ take ] dip [ swap sub ] swap cons map ] dip [ over over [ mod 0 eq? bool>int ] swap cons map product 1 eq? ] [ [ div ] swap cons map sum ] [ drop drop -1 ] ifte` |
| codearc-368 | 137 | `[ over 0 lt? ] [ 2 add over div swap ] [ over div swap ] ifte [ mul ] swap cons swap range1 swap map` |
| leetcode-bulb-switcher-ii | 138 | `[ dup 0 lt? ] [ 0 ] [ [ dup 0 eq? ] [ 1 ] [ [ over 0 le? ] [ 1 ] [ [ over 1 eq? ] [ 2 ] [ [ over 2 eq? ] [ [ dup 1 eq? ] [ 3 ] [ 4 ] ifte ] [ [ dup 1 eq? ] [ 4 ] [ [ dup 2 eq? ] [ 7 ] [ 8 ] ifte ] ifte ] ifte ] ifte ] ifte ] ifte ] ifte [ drop drop ] dip` |
| mbpp-149 | 143 | `take nil [ over over [ [ head ] dip sub abs 1 eq? ] swap cons filter seconds 0 [ max2 ] fold inc pack2 cons ] fold seconds 1 [ max2 ] fold` |
| leetcode-mirror-reflection | 144 | `over over gcd dup [ div 2 mod ] dip rot swap div 2 mod [ dup 1 eq? ] [ drop ] [ drop drop 2 ] ifte` |
| codearc-868 | 146 | `[ [ unpack2 ] dip2 pick3 pick2 cons pick3 [ pick4 ] dip swap pick4 pick4 [ [ sub abs ] dip eq? ] swap cons swap cons filter len add pack2 [ drop drop drop drop ] dip ] swap cons [ nil 0 pack2 ] dip fold unpack2 swap drop` |
| leetcode-apply-operations-to-make-sum-of-array-greater-than-or-equal-to-k | 150 | `[ dup 0 le? ] [ ] [ 4 mul dup isqrt dup sq pick2 lt? bool>int add swap drop 2 sub ] ifte` |
| leetcode-egg-drop-with-2-eggs-and-n-floors | 152 | `dup 8 mul 1 add isqrt 1 sub 2 div dup over pick2 1 add mul 2 div pick3 lt? bool>int add [ drop drop ] dip` |
| mbpp-60 | 158 | `take nil [ over over [ [ unpack2 drop ] dip sub abs 1 le? ] swap cons filter seconds 0 [ max2 ] fold inc pack2 cons ] fold seconds 0 [ max2 ] fold` |
| codearc-597 | 161 | `div dup range1 swap [ [ dup dup inc mul swap 2 mul inc mul 6 div ] dip le? ] swap cons filter len` |
| codearc-960 | 163 | `[ dup odd? bool>int swap abs digits 3 contains mul 1 eq? ] filter sum` |
| mbpp-5 | 165 | `[ dup odd? ] [ drop 0 ] [ 2 div range 1 1 pack2 [ drop unpack2 over 4 mul swap sub swap pack2 ] fold unpack2 drop ] ifte` |
| codearc-196 | 167 | `dup len dup range swap [ over inc -rot swap sub mul inc 2 div ] swap cons map dot` |
| leetcode-moving-stones-until-consecutive | 169 | `pack3 sort unpack3 over sub -rot swap sub swap [ over add 2 le? ] [ drop drop 0 0 pack2 ] [ over over add 2 sub [ min2 3 lt? bool>int 2 swap sub ] dip pack2 ] ifte` |
| codearc-888 | 174 | `dup abs pows2 swap [ le? ] swap cons filter last 2 mul 2 max2 [ over 0 lt? ] [ 2 mul ] [ ] ifte swap sub dec` |
| codearc-935 | 177 | `swap [ dup 0 lt? ] [ over len add 0 max2 ] [ ] ifte over swap drop-n zip [ unpack2 eq? bool>int ] map sum` |
| codearc-493 | 184 | `nil pack2 zeros16 swap [ drop unpack2 [ drop 10 ge? ] [ swap digit-product dup rot swap cons pack2 ] [ pack2 ] ifte ] fold unpack2 swap drop reverse` |
| mbpp-890 | 187 | `[ dup 0 lt? ] [ [ drop drop ] dip ] [ [ zip ] dip take [ unpack2 eq? bool>int ] map 0 1 pack2 [ [ unpack2 ] dip mul dup [ add ] dip pack2 ] fold head ] ifte` |
| mbpp-648 | 196 | `dup len range swap [ [ [ even? ] [ inc ] [ dec ] ifte ] dip swap nth ] swap cons map` |
| mbpp-144 | 200 | `dup [ take ] dip dup range swap [ swap 2 mul swap dec sub ] swap cons map dot` |
| codearc-488 | 202 | `[ dup 0 lt? ] [ drop drop 0 ] [ [ dup 0 eq? ] [ drop dup dedup swap [ swap count 1 gt? ] swap cons filter len ] [ [ dedup dup ] dip [ rot add contains ] swap cons swap cons map sum ] ifte ] ifte` |
| codearc-769 | 203 | `dup len dec range swap [ drop dup tail zip [ unpack2 add ] map ] fold sum` |
| codearc-850 | 203 | `[ dup null? ] [ drop 1 ] [ dup min 1 max2 range1 tail swap [ swap [ mod 0 eq? bool>int ] swap cons map product 1 eq? ] swap cons filter [ dup null? ] [ drop 1 ] [ head ] ifte ] ifte` |
| mbpp-782 | 209 | `dup len dup range -rot [ [ over nth ] dip swap [ over sub swap inc mul inc 2 div ] dip mul ] swap cons swap cons map sum` |
| mbpp-845 | 217 | `[ dup 0 lt? ] [ drop 0 ] [ range1 product digits len ] ifte` |
| mbpp-566 | 221 | `[ dup 0 lt? ] [ abs digits [ neg 10 mod ] map sum ] [ digit-sum ] ifte` |
| leetcode-the-kth-factor-of-n | 232 | `[ dup range1 swap [ swap divides? ] swap cons filter ] dip [ dup 0 le? ] [ drop drop -1 ] [ dec [ over len over gt? ] [ nth ] [ drop drop -1 ] ifte ] ifte` |
| mbpp-485 | 234 | `[ sort ] dip take [ palindrome? ] filter -1 [ max2 ] fold` |
| leetcode-sum-of-numbers-with-units-digit-k | 239 | `[ over 0 eq? ] [ drop drop 0 ] [ over 10 min2 range1 -rot [ rot mul sub dup 0 ge? bool>int swap 10 mod 0 eq? bool>int mul 1 eq? ] swap cons swap cons filter [ null? ] [ drop -1 ] [ head ] ifte ] ifte` |
| mbpp-48 | 248 | `dup 4 powers swap [ [ over ge? ] [ swap div 2 mod 0 eq? ] [ drop drop false ] ifte ] swap cons filter sum add` |
| codearc-94 | 254 | `abs swap abs gcd dup range1 swap [ swap mod 0 eq? ] swap cons filter sum` |
| mbpp-765 | 258 | `inc dup dup log2 add log2 add` |
| mbpp-909 | 260 | `dup range swap 0 pack2 [ drop unpack2 [ dup 1 eq? ] [ ] [ drop dec dup palindrome? bool>int ] ifte pack2 ] fold unpack2 drop` |
| codearc-509 | 265 | `digits dup len range swap [ swap drop-n uncons [ add ] swap cons map ] swap cons map nil [ cat ] fold` |
| leetcode-water-bottles-ii | 265 | `over pick2 pick2 pack3 pick2 abs pick2 abs add 2 add 2 mul pack2 [ drop drop ] dip unpack2 range swap [ drop unpack3 [ over over ge? ] [ pick2 1 add pick2 pick2 sub 1 add pick2 1 add pack3 [ drop drop drop ] dip ] [ pack3 ] ifte ] fold head` |
| codearc-1054 | 305 | `decimal-digits dup len dec range swap [ drop dup tail zip [ unpack2 add inc 2 div ] map ] fold head` |
| codearc-143 | 330 | `uncons dup dup pack2 dup pack3 [ [ unpack3 ] dip pick2 over [ unpack2 ] dip [ dup pick2 gt? ] [ [ pick2 over mul pick3 ge? ] [ pick2 over mul over pack2 ] [ pick2 pick2 pack2 ] ifte ] [ dup over pack2 ] ifte [ drop drop drop ] dip pick2 pick2 [ unpack2 ] dip [ dup pick2 gt? ] [ [ pick2 over mul pick3 le? ] [ pick2 over mul over pack2 ] [ pick2 pick2 pack2 ] ifte ] [ dup over pack2 ] ifte [ drop drop drop ] dip [ pick4 ] dip swap pick2 head pick2 head max2 max2 pick2 pick2 pack3 [ drop drop drop drop drop drop ] dip ] fold head` |
| mbpp-416 | 334 | `range1 nil 0 cons [ over over 2 div nth pick2 pick2 3 div nth add pick2 pick2 4 div nth add max2 quote cat ] fold last` |
| codearc-461 | 358 | `[ dup 9 le? ] [ abs 10 mod ] [ 1 1 pack3 zeros16 swap [ drop unpack3 [ over over mul 9 mul pick3 lt? ] [ over over mul 9 mul -rot [ sub ] dip2 [ 10 mul ] dip inc ] [ ] ifte pack3 ] fold unpack3 pick2 dec over mod [ rot dec swap div add decimal-digits ] dip nth ] ifte` |
| mbpp-707 | 358 | `[ dup 0 lt? ] [ inc 2 div ] [ range1 [ 2 base-digit-sum ] map sum ] ifte` |
| mbpp-469 | 379 | `over len min2 range swap dup [ drop 0 ] map pack2 [ drop unpack2 over zip dup head second neg 0 nil pack3 [ pack2 dup head head over second unpack2 sub max2 over head second pick2 second second pick2 add max2 pick2 head third over cons [ drop ] dip3 pack3 ] fold third reverse pack2 ] fold second last` |
| codearc-543 | 427 | `[ dup 0 lt? [ abs digits [ [ dup 3 eq? ] [ drop 7 ] [ [ dup 7 eq? ] [ drop 3 ] [ ] ifte ] ifte ] map 0 [ swap 10 mul add ] fold ] dip [ ] [ drop neg ] [ drop ] ifte ] map sort [ dup 0 lt? [ abs digits [ [ dup 3 eq? ] [ drop 7 ] [ [ dup 7 eq? ] [ drop 3 ] [ ] ifte ] ifte ] map 0 [ swap 10 mul add ] fold ] dip [ ] [ drop neg ] [ drop ] ifte ] map` |
| codearc-654 | 452 | `dup 2 mul range swap 0 pack2 [ [ unpack2 ] dip [ over 0 gt? ] [ drop pack2 ] [ [ drop ] dip over over add [ dup [ dup 10 gt? ] [ palindrome? ] [ drop false ] ifte ] [ [ drop ] dip pack2 ] [ drop over over sub [ dup [ dup 10 gt? ] [ palindrome? ] [ drop false ] ifte ] [ [ drop ] dip pack2 ] [ drop drop 0 pack2 ] ifte ] ifte ] ifte ] fold unpack2 swap drop` |
| leetcode-rotated-digits | 472 | `0 max2 digits dup nil 9 cons 8 cons 6 cons 5 cons 2 cons 1 cons 0 cons [ 1 0 pack2 ] dip [ pack3 dup dup head head swap dup third swap second contains mul swap dup head second over third len mul over head head pick2 third pick3 second [ lt? ] swap cons filter len mul add swap drop pack2 ] swap cons fold sum swap nil 8 cons 1 cons 0 cons [ 1 0 pack2 ] dip [ pack3 dup dup head head swap dup third swap second contains mul swap dup head second over third len mul over head head pick2 third pick3 second [ lt? ] swap cons filter len mul add swap drop pack2 ] swap cons fold sum sub` |
| codearc-943 | 516 | `[ over 0 lt? over 0 lt? eq? ] [ [ [ dup 0 lt? ] [ neg dec ] [ ] ifte ] dip [ dup 0 lt? ] [ neg dec ] [ ] ifte bxor ] [ [ [ dup 0 lt? ] [ neg dec ] [ ] ifte ] dip [ dup 0 lt? ] [ neg dec ] [ ] ifte bxor inc ] ifte pows2 [ [ swap div 2 mod ] swap cons ] dip swap map sum` |
| leetcode-minimum-number-of-operations-to-make-x-and-y-equal | 521 | `[ over over le? ] [ swap sub ] [ over over sub swap [ 0 pack2 quote ] dip2 pack3 [ [ drop head null? ] [ drop ] [ [ zeros1024 swap [ drop [ dup head null? ] [ ] [ unpack3 [ uncons unpack2 ] dip2 pick3 over sub abs pick3 add [ swap ] dip min2 swap [ pick3 over le? bool>int pick3 pick3 ge? bool>int add 0 gt? ] [ [ drop drop ] dip2 pack3 ] [ [ nil 5 cons 10 inc cons -rot [ over pick3 div over pick3 [ pick4 ] dip swap mod inc add pack2 pick2 pick4 div inc pick2 [ pick4 ] dip swap [ pick4 ] dip swap [ [ [ pick4 ] dip swap ] dip swap ] dip swap mod sub inc add pack2 pack2 [ drop drop drop ] dip ] swap cons swap cons map nil [ cat ] fold cat ] dip2 pack3 ] ifte ] ifte ] fold ] dip dup i ] ifte ] dup i unpack3 drop swap drop ] ifte` |
| codearc-610 | 523 | `dec range swap quote [ drop dup head dup isqrt range1 tail over 0 pack2 pick2 [ [ drop over head swap mod 0 eq? ] [ [ [ drop ] dip2 [ drop over head swap mod 0 eq? ] [ pack3 dup head head over second div swap dup head second swap dup third swap second div add pack2 ] [ drop drop ] ifte ] swap cons swap cons zeros16 -rot fold ] [ drop drop ] ifte ] swap cons fold unpack2 [ [ dup 1 gt? ] [ div ] [ drop drop 0 ] ifte ] dip add 1 max2 cons ] fold reverse` |
| leetcode-2-keys-keyboard | 538 | `[ dup 0 le? ] [ ] [ dup log2 range swap 0 pack2 [ drop unpack2 [ over 1 gt? ] [ swap dup dup dup isqrt range1 tail swap [ swap divides? ] swap cons filter [ dup null? ] [ drop ] [ head swap drop ] ifte swap over div [ add ] dip swap ] [ ] ifte pack2 ] fold unpack2 swap drop ] ifte` |
| mbpp-195 | 649 | `pick2 pick2 0 pick3 1 sub 0 1 sub pack3 pack3 over pack2 [ drop drop drop ] dip unpack2 range swap [ drop unpack3 unpack3 [ pick2 pick2 le? ] [ pick2 pick2 add 2 div [ pick4 ] dip swap over nth [ [ pick4 ] dip swap ] dip swap [ [ pick4 ] dip swap ] dip swap [ pick2 [ [ [ [ pick4 ] dip swap ] dip swap ] dip swap ] dip swap eq? ] [ [ [ pick4 ] dip swap ] dip swap pick4 1 sub [ pick4 ] dip swap pack3 ] [ [ pick2 [ [ [ [ pick4 ] dip swap ] dip swap ] dip swap ] dip swap gt? ] [ [ [ pick4 ] dip swap ] dip swap pick4 1 sub [ [ pick4 ] dip swap ] dip swap pack3 ] [ pick3 1 add [ [ pick4 ] dip swap ] dip swap [ [ pick4 ] dip swap ] dip swap pack3 ] ifte ] ifte pack3 [ drop drop drop drop drop drop drop ] dip ] [ pack3 pack3 ] ifte ] fold last last` |
| codearc-990 | 677 | `[ 10 inc max2 nil pack2 ] dip range swap [ drop unpack2 over dup digits dup len 2 mod over pick2 len 1 add 2 div take from-digits dup pick2 [ digits dup reverse ] dip [ dup 1 eq? ] [ drop tail ] [ drop ] ifte cat from-digits over over [ [ pick4 ] dip swap ] dip swap lt? bool>int add pick3 [ digits dup reverse ] dip [ dup 1 eq? ] [ drop tail ] [ drop ] ifte cat from-digits [ drop drop drop drop drop ] dip dup 1 add pick2 pick2 cons pack2 [ drop drop drop ] dip ] fold unpack2 swap drop reverse` |
| codearc-569 | 811 | `[ over null? ] [ swap drop ] [ [ dup null? ] [ drop ] [ [ [ dup head 0 lt? ] [ [ abs decimal-digits ] map nil [ cat ] fold 0 [ swap 10 mul add ] fold neg ] [ [ abs decimal-digits ] map nil [ cat ] fold 0 [ swap 10 mul add ] fold ] ifte ] dip [ dup head 0 lt? ] [ [ abs decimal-digits ] map nil [ cat ] fold 0 [ swap 10 mul add ] fold neg ] [ [ abs decimal-digits ] map nil [ cat ] fold 0 [ swap 10 mul add ] fold ] ifte add [ dup 0 lt? ] [ neg decimal-digits uncons neg cons ] [ decimal-digits ] ifte ] ifte ] ifte` |
| leetcode-super-egg-drop | 850 | `[ dup 1 lt? ] [ drop drop 0 ] [ [ over 1 eq? ] [ swap drop ] [ [ over 0 le? ] [ swap drop bitlen ] [ dup pick2 pick2 bitlen min2 1 add range [ drop 0 ] map 0 pack3 over 2 mul isqrt 2 add pack2 [ drop drop ] dip unpack2 range swap [ drop unpack3 [ over last pick3 ge? ] [ pick2 pick2 pick2 pack3 ] [ pick2 pick2 dup tail swap zip [ unpack2 add inc ] map 0 cons pick2 1 add pack3 ] ifte [ drop drop drop ] dip ] fold last ] ifte ] ifte ] ifte` |
| leetcode-the-earliest-and-latest-rounds-where-players-compete | 1310 | `pick2 over sub inc pick2 over over over min2 32 mul [ max2 ] dip add quote dup [ pick4 ] dip swap 1 0 0 pack5 [ pick4 ] dip swap range dup pick2 [ drop dup unpack5 pick4 [ dup 32 div swap 32 mod eq? ] filter len 0 gt? bool>int pick2 pick4 pick2 [ dup 1 eq? pick3 0 eq? bool>int swap bool>int mul 1 eq? ] [ drop swap drop ] [ drop drop ] ifte pick2 [ pick4 ] dip swap pick3 [ dup 1 eq? ] [ drop swap drop ] [ drop drop ] ifte [ [ [ pick4 ] dip swap ] dip swap ] dip swap [ dup 32 div swap 32 mod eq? not ] filter dup [ [ [ [ pick4 ] dip swap ] dip swap ] dip swap ] dip swap [ over 32 div pick2 32 mod over inc pick2 pick2 add pick4 2 div over over sub pick3 over max2 [ [ [ pick4 ] dip swap ] dip swap ] dip swap inc 2 div [ [ pick4 ] dip swap ] dip swap over min2 [ [ [ [ pick4 ] dip swap ] dip swap ] dip swap ] dip swap range1 pick3 pick2 range-between dup pick2 [ swap [ over sub over over min2 32 mul [ max2 ] dip add ] swap cons map ] swap cons map flatten dup [ drop drop drop drop drop drop drop drop drop drop drop drop drop drop ] dip ] swap cons map flatten dedup dup [ [ [ [ [ pick4 ] dip swap ] dip swap ] dip swap ] dip swap ] dip swap inc 2 div [ [ [ [ [ pick4 ] dip swap ] dip swap ] dip swap ] dip swap ] dip swap inc [ [ pick4 ] dip swap ] dip swap [ [ pick4 ] dip swap ] dip swap pack5 dup [ drop drop drop drop drop drop drop drop drop drop drop drop ] dip ] fold dup unpack5 over over pack2 dup [ drop drop drop drop drop drop drop drop drop drop drop drop drop drop ] dip` |
| codearc-1022 | 1562 | `[ over over gt? ] [ 0 ] [ [ dup 0 lt? ] [ over abs digits 0 1 pack2 [ [ unpack2 ] dip pick2 9 mul pick2 pick2 pick3 5 gt? bool>int sub mul add pick2 pick2 5 eq? 0 1 eq? eq? bool>int mul pack2 [ drop drop drop ] dip ] fold unpack2 add over abs 1 sub digits 0 1 pack2 [ [ unpack2 ] dip pick2 9 mul pick2 pick2 pick3 5 gt? bool>int sub mul add pick2 pick2 5 eq? 0 1 eq? eq? bool>int mul pack2 [ drop drop drop ] dip ] fold unpack2 add sub ] [ [ over 0 lt? ] [ over abs digits 0 1 pack2 [ [ unpack2 ] dip pick2 9 mul pick2 pick2 pick3 5 gt? bool>int sub mul add pick2 pick2 5 eq? 0 1 eq? eq? bool>int mul pack2 [ drop drop drop ] dip ] fold unpack2 add over digits 0 1 pack2 [ [ unpack2 ] dip pick2 9 mul pick2 pick2 pick3 5 gt? bool>int sub mul add pick2 pick2 5 eq? 0 1 eq? eq? bool>int mul pack2 [ drop drop drop ] dip ] fold unpack2 add add 1 sub ] [ [ over 0 eq? ] [ dup digits 0 1 pack2 [ [ unpack2 ] dip pick2 9 mul pick2 pick2 pick3 5 gt? bool>int sub mul add pick2 pick2 5 eq? 0 1 eq? eq? bool>int mul pack2 [ drop drop drop ] dip ] fold unpack2 add ] [ dup digits 0 1 pack2 [ [ unpack2 ] dip pick2 9 mul pick2 pick2 pick3 5 gt? bool>int sub mul add pick2 pick2 5 eq? 0 1 eq? eq? bool>int mul pack2 [ drop drop drop ] dip ] fold unpack2 add pick2 1 sub digits 0 1 pack2 [ [ unpack2 ] dip pick2 9 mul pick2 pick2 pick3 5 gt? bool>int sub mul add pick2 pick2 5 eq? 0 1 eq? eq? bool>int mul pack2 [ drop drop drop ] dip ] fold unpack2 add sub ] ifte ] ifte ] ifte ] ifte [ drop drop ] dip` |
| leetcode-maximum-points-in-an-archery-competition | 1762 | `dup [ inc ] map [ drop ] dip dup head 0 min2 pick2 over sub [ drop ] dip over [ abs ] map sum pick3 abs over over add inc [ drop ] dip [ drop ] dip pick2 len dec range1 pick3 tail [ [ [ [ drop ] dip ] dip ] dip ] dip over over zip [ drop ] dip [ drop ] dip 0 quote quote over over pick4 [ pick2 head pick2 unpack2 [ [ [ [ drop ] dip ] dip ] dip ] dip over range [ [ drop ] dip ] dip dup pick4 [ swap drop ] swap cons map [ drop ] dip [ [ [ drop ] dip ] dip ] dip pick2 pick2 [ add ] swap cons map [ [ drop ] dip ] dip pick2 pick2 cat [ [ [ drop ] dip ] dip ] dip pick2 pick2 cat [ [ drop ] dip ] dip [ [ drop ] dip ] dip over over zip [ unpack2 min2 ] map [ drop ] dip [ drop ] dip over over cons [ drop ] dip [ drop ] dip ] swap cons fold [ drop ] dip dup head dup len range dup pick2 zip [ drop ] dip [ drop ] dip dup [ pick4 ] dip swap [ [ unpack2 swap drop ] dip le? ] swap cons filter [ head ] map 0 [ max2 ] fold [ drop ] dip nil pick2 [ [ pick4 ] dip swap ] dip swap pick3 pick3 pack4 [ drop ] dip [ [ drop ] dip ] dip [ [ [ [ drop ] dip ] dip ] dip ] dip pick2 reverse [ [ [ drop ] dip ] dip ] dip dup pick2 [ pick4 ] dip swap [ pick2 unpack4 [ [ [ [ [ [ drop ] dip ] dip ] dip ] dip ] dip ] dip pick3 tail [ [ [ [ drop ] dip ] dip ] dip ] dip dup head [ [ pick4 ] dip swap ] dip swap unpack2 [ [ [ [ [ [ [ [ drop ] dip ] dip ] dip ] dip ] dip ] dip ] dip ] dip [ [ [ pick4 ] dip swap ] dip swap ] dip swap pick3 [ [ [ pick4 ] dip swap ] dip swap ] dip swap [ over len over le? ] [ drop drop ] [ rot drop nth ] ifte [ [ [ drop ] dip ] dip ] dip [ [ [ [ [ [ [ drop ] dip ] dip ] dip ] dip ] dip ] dip ] dip dup [ [ [ pick4 ] dip swap ] dip swap ] dip swap gt? bool>int [ drop ] dip over over mul [ [ drop ] dip ] dip [ [ pick4 ] dip swap ] dip swap over sub [ [ [ [ [ [ [ drop ] dip ] dip ] dip ] dip ] dip ] dip ] dip pick3 pick3 mul [ [ [ drop ] dip ] dip ] dip [ [ [ drop ] dip ] dip ] dip [ pick4 ] dip swap over sub [ drop ] dip [ [ [ [ [ drop ] dip ] dip ] dip ] dip ] dip pick4 pick3 cons [ [ [ drop ] dip ] dip ] dip [ [ [ [ drop ] dip ] dip ] dip ] dip pick3 pick3 pick3 pick3 pack4 [ drop ] dip [ drop ] dip [ drop ] dip [ drop ] dip ] swap cons fold [ drop ] dip [ drop ] dip [ [ drop ] dip ] dip dup unpack4 [ drop drop drop ] dip [ drop ] dip over over [ over 0 eq? ] [ swap drop [ drop 0 ] map ] [ swap drop ] ifte [ drop ] dip [ drop ] dip over over sum sub [ [ drop ] dip ] dip over over cons [ drop ] dip [ drop ] dip` |
| mbpp-854 | 1967 | `dup len 2 div range reverse swap [ over over pick2 pick4 pick4 nth pack4 [ drop drop ] dip dup head len bitlen range swap [ drop unpack4 [ pick2 2 mul 1 add pick4 len lt? [ drop drop drop drop ] dip ] [ pick2 2 mul 1 add [ dup 1 add [ pick4 ] dip swap len lt? ] [ [ pick4 over nth [ pick4 ] dip swap pick2 1 add nth ge? ] [ dup 1 add ] [ dup ] ifte ] [ dup ] ifte [ pick4 ] dip swap [ pick4 ] dip swap [ [ [ pick4 ] dip swap ] dip swap ] dip swap pick3 nth pick2 pick2 take over quote cat pick3 pick3 1 add drop-n cat [ drop drop drop ] dip over [ pick4 ] dip swap [ pick4 ] dip swap pack4 [ drop drop drop drop drop drop ] dip ] [ pack4 ] ifte ] fold dup head len bitlen range swap [ drop unpack4 [ pick2 pick2 gt? [ drop drop drop drop ] dip ] [ [ dup pick4 pick4 1 sub 2 div nth lt? ] [ pick3 pick3 [ pick4 ] dip swap [ pick4 ] dip swap 1 sub 2 div nth pick2 pick2 take over quote cat pick3 pick3 1 add drop-n cat [ drop drop drop ] dip pick3 1 sub 2 div pick3 pick3 pack4 ] [ pick3 pick3 pick3 pick3 pack4 ] ifte [ drop drop drop drop ] dip ] [ pack4 ] ifte ] fold unpack4 pick3 pick3 pick2 pick2 pick2 take over quote cat pick3 pick3 1 add drop-n cat [ drop drop drop ] dip [ drop drop drop drop ] dip ] fold` |
