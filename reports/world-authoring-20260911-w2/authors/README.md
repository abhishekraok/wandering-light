# world-authoring-20260911-w2 raw (author dialects)

raw author outputs before the merge pass


## Verified

| source | verified | median witness length (`mark-28` ops) | median macro-form length |
|---|---|---|---|
| codearc | 47/50 (94%) | 157 | 11 |
| mbpp | 49/50 (98%) | 123 | 9 |
| **all** | 96/100 (96%) | 144.5 | 9.0 |

Claimed-vs-verified mismatches: 1; missing ids: 0; duplicates: 0.

By author: a 26/26 (100%); b 26/26 (100%); c 21/24 (88%); d 23/24 (96%).

## Witness length against `stack_pbe_v4`

| bucket (ops) | this probe (verified) | v4 behaviours (train + eval, 220) |
|---|---|---|
| 1-5 | 1 | 15 |
| 6-10 | 3 | 14 |
| 11-20 | 7 | 34 |
| 21-40 | 10 | 37 |
| 41+ | 75 | 120 |
| median | 144.5 | 43.0 |

Exact matches with a v4 witness: 0 (none). Same skeleton as a v4 witness: 1.

## Macros

200 macros defined across `macros*.txt`; median expansion ratio (`mark-28` ops per macro-form token) on verified programs: 13.7.

| macro | verified programs using it |
|---|---|
| `2` | 13 |
| `pack2` | 13 |
| `len` | 11 |
| `dec` | 10 |
| `inc` | 10 |
| `range` | 10 |
| `head` | 10 |
| `le?` | 9 |
| `max2` | 8 |
| `take` | 8 |
| `second` | 8 |
| `mod` | 8 |
| `rot` | 7 |
| `3` | 6 |
| `last` | 6 |
| `unpack2` | 6 |
| `4` | 5 |
| `range1` | 5 |
| `reverse` | 5 |
| `neg` | 5 |

## Unsolved

| id | source | passed | note |
|---|---|---|---|
| mbpp-587 | mbpp | 0/27 | shape: identity; tuple output is normalized to a flat list |
| codearc-462 | codearc | 13/34 | why: fuel exhausted on larger ranges; current 13/34, best 14/34 before generalizing prefix-window validation; final-digi |
| codearc-376 | codearc | 29/34 | 29/34; why: breadth-first nondecreasing factorizations exceed fuel on large products; includes reference bounds and orig |
| codearc-721 | codearc | 5/34 | why: direct ant simulation with the reference periodic reduction passes 5/34; fuel is exhausted at n=500 and throughout  |

## Every verified program

| id | len | macro form |
|---|---|---|
| mbpp-89 | 2 | `dec` |
| mbpp-664 | 6 | `2 div inc` |
| mbpp-581 | 7 | `2 mul over add mul` |
| mbpp-135 | 8 | `dup 2 mul dec mul` |
| mbpp-86 | 12 | `dup dec mul 3 mul inc` |
| codearc-893 | 13 | `[ lt? ] swap cons filter len` |
| codearc-130 | 13 | `[ even? ] filter` |
| mbpp-426 | 13 | `[ odd? ] filter` |
| codearc-587 | 16 | `abs neg` |
| mbpp-710 | 16 | `dup head swap last pack2` |
| mbpp-797 | 16 | `inc 2 div sq swap 2 div sq sub` |
| mbpp-609 | 21 | `over dec max2 swap [ mul ] dip div` |
| codearc-7 | 22 | `dup [ swap count 1 eq? ] swap cons filter` |
| codearc-154 | 22 | `dup [ swap count 1 eq? ] swap cons filter` |
| codearc-314 | 23 | `[ 2 mod 2 mul 1 swap sub ] map sum` |
| mbpp-968 | 23 | `over dec min2 rot mul swap div` |
| mbpp-641 | 24 | `dup 7 mul 5 sub mul 2 div` |
| mbpp-329 | 24 | `[ 0 le? ] filter len` |
| mbpp-221 | 28 | `[ even? ] filter [ null? ] [ drop -1 ] [ head ] ifte` |
| codearc-1109 | 35 | `dup rot swap sub [ sub ] dip ka-ceildiv 1 max2` |
| codearc-718 | 35 | `[ swap contains 0 eq? ] swap cons filter` |
| codearc-738 | 45 | `dup min swap max over over swap sub pack3` |
| codearc-801 | 54 | `dedup sum` |
| codearc-155 | 54 | `dedup product` |
| mbpp-362 | 55 | `dup [ over count pack2 ] swap cons map uncons [ [ over second over second lt? ] [ swap drop ] [ drop ] ifte ] fold head` |
| mbpp-817 | 56 | `[ kb-divides-either ] swap cons swap cons filter` |
| mbpp-289 | 56 | `dup 4 div over 100 div sub over 100 4 mul div add add 7 mod` |
| mbpp-141 | 57 | `sort` |
| mbpp-594 | 59 | `dup [ even? ] filter kc-first-or-minus1 swap [ odd? ] filter kc-first-or-minus1 sub` |
| codearc-512 | 60 | `inversions` |
| mbpp-895 | 67 | `uncons 0 swap pack2 [ [ unpack2 ] dip pick2 0 max2 add over max2 rot drop pack2 ] fold second` |
| mbpp-264 | 70 | `[ dup 2 le? ] [ 10 2 mul inc mul 2 div ] [ 2 sub 4 mul 10 2 mul inc add ] ifte` |
| mbpp-324 | 74 | `0 0 0 pack3 [ [ unpack3 ] dip [ pick3 odd? ] [ [ swap ] dip add swap ] [ add ] ifte [ inc ] dip2 pack3 ] fold tail` |
| mbpp-801 | 86 | `pack3 dedup len [ dup 3 eq? ] [ drop 0 ] [ 4 swap sub ] ifte` |
| codearc-1008 | 87 | `reverse uncons swap uncons rot over over swap sub [ max2 ] dip swap pack2 [ [ unpack2 ] dip over over max2 [ sub max2 ] dip pack2 ] fold head` |
| codearc-153 | 89 | `digit-sum` |
| codearc-580 | 96 | `0 0 pack2 quote [ ka-subsets-step ] fold swap [ swap head eq? ] swap cons filter seconds swap [ le? ] swap cons filter max` |
| mbpp-479 | 105 | `[ dup 10 lt? ] [ ] [ digits head ] ifte` |
| codearc-11 | 106 | `dedup sort` |
| mbpp-831 | 107 | `take kc-equal-pairs` |
| codearc-814 | 115 | `over sub inc range swap [ add ] swap cons map` |
| codearc-677 | 116 | `[ dup 0 eq? ] [ drop 1 ] [ dec isqrt inc 2 div inc ] ifte` |
| codearc-481 | 116 | `over swap [ kb-unswallowable? ] swap cons swap cons filter len` |
| codearc-1012 | 117 | `max-subarray-sum 0 max2` |
| mbpp-199 | 123 | `pows2 swap [ le? ] swap cons filter last` |
| codearc-142 | 125 | `kc-max-product` |
| mbpp-583 | 131 | `kc-catalan` |
| mbpp-57 | 144 | `[ sort reverse ] dip 1 max2 take from-digits` |
| codearc-544 | 145 | `dup [ ka-interval-power ] dip over div mul` |
| codearc-472 | 145 | `kc-fibonacci-pair` |
| mbpp-463 | 148 | `kc-mbpp-product` |
| mbpp-183 | 148 | `[ take ] dip abs kc-difference-pairs` |
| mbpp-385 | 152 | `kc-perrin` |
| codearc-594 | 157 | `0 pack2 zeros16 swap [ kb-persistence-step ] fold second` |
| mbpp-620 | 162 | `take reverse nil [ ka-divchain-step ] fold seconds max` |
| codearc-715 | 162 | `[ dup ka-oddity-key swap pack2 ] map uncons [ [ over head over head lt? ] [ swap drop ] [ drop ] ifte ] fold second` |
| mbpp-520 | 163 | `uncons [ ka-lcm ] fold` |
| mbpp-169 | 171 | `[ dup 2 le? ] [ ] [ 2 sub range 1 2 pack2 [ drop unpack2 dup 2 mul rot add pack2 ] fold second ] ifte` |
| codearc-622 | 178 | `[ dup 0 le? ] [ drop drop -1 ] [ [ abs ] dip dec range swap [ drop 10 div ] fold 10 mod ] ifte` |
| mbpp-786 | 185 | `kc-bisect-right` |
| codearc-198 | 185 | `kc-bisect-right` |
| mbpp-344 | 192 | `isqrt swap dec isqrt sub` |
| mbpp-911 | 203 | `sort dup reverse 3 take product swap dup last swap 2 take product mul max2` |
| mbpp-189 | 208 | `take dup len inc range1 swap [ swap contains 0 eq? ] swap cons filter head` |
| codearc-119 | 225 | `[ drop ] dip range swap [ drop ] swap quote cat map flatten max-subarray-sum` |
| mbpp-194 | 225 | `[ dup 0 lt? ] [ abs digits [ neg 10 mod ] map ] [ digits ] ifte 0 [ swap 8 mul add ] fold` |
| codearc-629 | 238 | `[ dup 0 eq? ] [ drop ] [ [ dup 0 lt? ] [ gcd neg ] [ gcd ] ifte ] ifte` |
| mbpp-501 | 238 | `gcd dup range1 swap [ swap divides? ] swap cons filter len` |
| codearc-221 | 239 | `dup len range zip [ unpack2 mul ] map 1 drop-n` |
| codearc-74 | 251 | `[ drop drop kc-prime? ] [ drop swap drop ] [ swap drop swap drop ] ifte` |
| codearc-565 | 260 | `[ dup 0 lt? ] [ [ reverse ] dip 3 swap sub ] [ ] ifte [ dup 4 lt? ] [ nth ] [ 3 sub range swap [ drop kb-sequence-step ] fold last ] ifte` |
| mbpp-736 | 270 | `over len dup range [ 0 swap pack4 ] dip swap [ ka-bisect-step ] fold third` |
| codearc-287 | 272 | `kd-pop-root-step kd-pop-root-step` |
| mbpp-671 | 273 | `kb-set-unset` |
| codearc-507 | 276 | `kc-reverse-binary` |
| codearc-391 | 285 | `[ dup 0 le? ] [ drop 0 ] [ dup log2 swap ka-popcount add ] ifte` |
| codearc-937 | 285 | `[ dup ka-popcount swap pack2 ] map nil [ ka-bitpair-insert ] fold seconds` |
| mbpp-402 | 286 | `[ over 0 eq? ] [ drop drop drop 1 ] [ [ kd-choose ] dip mod ] ifte` |
| codearc-72 | 302 | `[ dup kb-digit-weight swap pack2 ] map nil [ kb-key-insert ] fold seconds` |
| codearc-233 | 317 | `[ sort prefix-sums dup len range zip ] dip [ [ head ] dip ge? ] swap cons filter head second` |
| codearc-473 | 333 | `dup len over dup max swap min sub 2 add mul range swap 0 pack2 [ ka-equalize-step ] fold unpack2 swap last pack2` |
| codearc-172 | 350 | `take nil 0 pack2 [ kb-xor-pairs-step ] fold second` |
| mbpp-28 | 364 | `[ lt? ] [ drop drop 0 ] [ over over sub ka-factorial [ ka-factorial ] dip mul [ ka-factorial ] dip div ] ifte` |
| mbpp-849 | 367 | `dup range1 [ kc-prime? ] filter swap [ swap mod 0 eq? ] swap cons filter sum` |
| codearc-954 | 408 | `[ dup 1 le? ] [ 1 eq? bool>int 2 mul dec ] [ dup isqrt range1 1 drop-n swap 1 pack2 [ kd-mobius-step ] fold unpack2 [ over 1 eq? ] [ swap drop ] [ swap drop neg ] ifte ] ifte` |
| codearc-381 | 414 | `1 10 inc pack3 zeros16 swap [ kb-digit-sequence-step ] fold unpack3 rot swap sub over over swap mod pick2 dec swap sub 10 swap pow rot mul div 10 mod` |
| codearc-810 | 445 | `[ dup 0 lt? ] [ [ over len neg lt? ] [ drop drop -1 ] [ dup [ at ] dip neg pow 1 swap div ] ifte ] [ [ over len ge? ] [ drop drop -1 ] [ dup [ nth ] dip pow ] ifte ] ifte` |
| mbpp-325 | 451 | `[ dup 3 le? ] [ ] [ range1 nil 0 cons [ kb-min-squares-step ] fold last ] ifte` |
| codearc-1053 | 501 | `dup 0 max2 kd-65 2 mul div isqrt inc 4 mul range [ kd-sierpinski-increment ] map prefix-sums uncons rot [ kd-nearest-step ] swap cons fold` |
| codearc-407 | 527 | `kd-chameleon-order [ dup head 0 eq? bool>int over third 0 eq? bool>int mul 1 eq? ] [ drop -1 ] [ dup second over head sub 3 mod 0 eq? [ ] [ drop second ] [ drop drop -1 ] ifte ] ifte` |
| mbpp-245 | 541 | `kc-bitonic-sum` |
| mbpp-119 | 736 | `take 0 [ ka-xor ] fold` |
| mbpp-100 | 917 | `kd-next-pal` |
| codearc-289 | 917 | `kd-next-pal` |
| mbpp-209 | 1741 | `[ kb-heapify ] dip 0 swap kb-set-at 0 kb-sift` |
| codearc-585 | 1866 | `kc-emirps` |
