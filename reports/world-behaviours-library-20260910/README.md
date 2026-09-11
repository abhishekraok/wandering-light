# World-behaviours probe: can MBPP and LeetCode-easy behaviours get `mark-28` witnesses?

training-data-sources (#118). Behaviours sampled by `experiments/world_behaviours/extract.py`
(seed 17, ints and flat int lists only, ≤ 3 arguments); programs authored by Claude in macro
form and expanded to `mark-28`; every program re-verified on every test by `report.py`.


## Verified

| source | verified | median witness length (`mark-28` ops) | median macro-form length |
|---|---|---|---|
| leetcode | 42/45 (93%) | 105.0 | 25.0 |
| mbpp | 49/50 (98%) | 123 | 20 |
| **all** | 91/95 (96%) | 111 | 24 |

Claimed-vs-verified mismatches: 0; missing ids: 0; duplicates: 0.

By author: canonical 91/95 (96%).

## Witness length against `stack_pbe_v4`

| bucket (ops) | this probe (verified) | v4 behaviours (train + eval, 220) |
|---|---|---|
| 1-5 | 6 | 15 |
| 6-10 | 3 | 14 |
| 11-20 | 8 | 34 |
| 21-40 | 6 | 37 |
| 41+ | 68 | 120 |
| median | 111 | 43.0 |

Exact matches with a v4 witness: 5 (leetcode-add-two-integers, mbpp-309, mbpp-588, mbpp-925, mbpp-970). Same skeleton as a v4 witness: 5.

## Macros

110 macros defined across `macros*.txt`; median expansion ratio (`mark-28` ops per macro-form token) on verified programs: 4.0.

| macro | verified programs using it |
|---|---|
| `2` | 23 |
| `inc` | 22 |
| `mod` | 20 |
| `dec` | 18 |
| `range` | 17 |
| `max2` | 16 |
| `len` | 15 |
| `head` | 14 |
| `10` | 12 |
| `sum` | 11 |
| `le?` | 11 |
| `bool>int` | 10 |
| `min2` | 9 |
| `rot` | 9 |
| `7` | 8 |
| `take` | 8 |
| `100` | 7 |
| `4` | 7 |
| `range1` | 7 |
| `-1` | 7 |

## Unsolved

| id | source | passed | note |
|---|---|---|---|
| leetcode-convert-integer-to-the-sum-of-two-no-zero-integers | leetcode | 0/40 | unsolved on fuel: linear search needs >=611 candidates (test 10606) x two zero-free checks; k4-digit-product is ~380 fue |
| leetcode-count-largest-group | leetcode | 0/40 | unsolved: n up to 10000 needs a digit-sum histogram over every 1..n; k4-digit-sum is ~350 fuel each (3.5M total vs 100k) |
| leetcode-maximum-number-of-balls-in-a-box | leetcode | 0/40 | fuel: ranges hold up to 90002 numbers (100k step cap); histogram over 46 digit sums has no closed form the basis affords |
| mbpp-579 | mbpp | 0/27 | output order is CPython set iteration order (tuple(set(a)^set(b))), not sorted; reproducing it means simulating the hash |

## Every verified program

| id | len | macro form |
|---|---|---|
| leetcode-add-two-integers | 1 | `add` |
| leetcode-count-of-matches-in-tournament | 2 | `dec` |
| mbpp-176 | 2 | `add add` |
| mbpp-373 | 2 | `mul mul` |
| leetcode-find-the-maximum-achievable-number | 5 | `2 mul add` |
| mbpp-925 | 5 | `product` |
| mbpp-646 | 7 | `sub inc dup sq mul` |
| leetcode-water-bottles | 8 | `dec over dec swap div add` |
| mbpp-171 | 8 | `5 mul` |
| mbpp-309 | 12 | `max2` |
| mbpp-970 | 12 | `min2` |
| mbpp-41 | 13 | `[ even? ] filter` |
| leetcode-count-distinct-numbers-on-board | 15 | `dec 1 max2` |
| leetcode-calculate-delayed-arrival-time | 19 | `add 6 4 mul mod` |
| leetcode-smallest-even-multiple | 19 | `[ even? ] [ ] [ dup add ] ifte` |
| mbpp-313 | 19 | `[ 0 ge? ] filter head` |
| leetcode-divisible-and-non-divisible-sums-difference | 20 | `over over div dup inc mul mul [ dup inc mul 2 div ] dip sub` |
| mbpp-572 | 22 | `dup [ swap count 1 eq? ] swap cons filter` |
| mbpp-807 | 29 | `[ odd? ] filter [ dup null? ] [ drop -1 ] [ head ] ifte` |
| leetcode-minimum-cuts-to-divide-a-circle | 32 | `[ 1 eq? ] [ drop 0 ] [ [ odd? ] [ ] [ 2 div ] ifte ] ifte` |
| leetcode-pass-the-pillow | 34 | `over dec 2 mul mod [ over lt? ] [ swap drop inc ] [ swap 2 mul dec swap sub ] ifte` |
| mbpp-588 | 35 | `dup max swap min sub` |
| mbpp-249 | 36 | `swap [ swap contains 1 eq? ] swap cons filter` |
| leetcode-account-balance-after-rounded-purchase | 41 | `5 add 10 div 10 mul 100 swap sub` |
| leetcode-find-the-child-who-has-the-ball-after-k-seconds | 41 | `over dec over over mod [ div ] dip [ drop odd? ] [ swap drop sub dec ] [ swap drop swap drop ] ifte` |
| leetcode-minimum-amount-of-time-to-fill-cups | 41 | `dup max swap sum inc 2 div max2` |
| mbpp-591 | 42 | `[ tail null? ] [ ] [ dup last swap dup head swap tail reverse tail reverse swap quote cat swap cons ] ifte` |
| leetcode-total-distance-traveled | 45 | `over dec 4 div 0 max2 min2 add 10 mul` |
| leetcode-smallest-number-with-all-set-bits | 50 | `[ swap drop [ over dec swap lt? ] [ drop dup add ] [ drop ] ifte ] swap cons zeros16 1 rot fold dec` |
| mbpp-152 | 57 | `sort` |
| mbpp-366 | 57 | `uncons 0 pack2 [ [ unpack2 ] dip dup [ [ over ] dip mul max2 ] dip swap [ swap drop ] dip pack2 ] fold unpack2 swap drop` |
| mbpp-824 | 58 | `nil 0 cons [ [ uncons ] dip [ drop 1 eq? ] [ swap drop cons 0 cons ] [ [ odd? ] [ swap drop cons 0 cons ] [ drop drop 1 cons ] ifte ] ifte ] fold uncons drop reverse` |
| mbpp-962 | 63 | `[ dec [ dup 0 lt? ] [ neg 2 div neg ] [ 2 div ] ifte dup inc mul ] dip [ dup 0 lt? ] [ neg 2 div neg ] [ 2 div ] ifte dup inc mul swap sub` |
| leetcode-find-the-distance-value-between-two-arrays | 64 | `[ [ swap ] dip [ [ sub abs ] dip le? ] swap cons swap cons filter len 0 eq? ] swap cons swap cons filter len` |
| leetcode-calculate-money-in-leetcode-bank | 72 | `dup 7 div swap 7 mod over 2 mul over add inc over mul 2 div [ drop dup 7 add mul 7 mul 2 div ] dip add` |
| leetcode-distribute-money-to-maximum-children | 73 | `[ lt? ] [ drop drop -1 ] [ [ 8 mul swap lt? ] [ swap drop dec ] [ [ 8 mul 4 sub eq? ] [ swap drop 2 sub ] [ sub 7 div ] ifte ] ifte ] ifte` |
| mbpp-663 | 77 | `[ dup pick2 mod eq? ] [ [ abs ] dip over mod [ [ over ] dip lt? ] [ drop drop drop -1 ] [ [ over ] dip sub swap mod sub ] ifte ] [ drop drop drop -1 ] ifte` |
| mbpp-548 | 81 | `nil [ lis-best inc pack2 cons ] fold seconds 0 [ max2 ] fold` |
| leetcode-number-of-students-doing-homework-at-a-given-time | 92 | `dup [ le? bool>int ] swap cons swap [ ge? bool>int ] swap cons [ swap ] dip map [ map ] dip dot` |
| mbpp-54 | 93 | `dup 0 [ max2 ] fold inc [ [ over 0 lt? ] [ add ] [ drop ] ifte ] swap cons map sort` |
| leetcode-form-smallest-number-from-two-digit-arrays | 102 | `[ swap [ [ over over eq? ] [ drop ] [ over over min2 10 mul [ max2 ] dip add ] ifte ] swap cons map min min2 ] swap cons 100 swap fold` |
| leetcode-count-integers-with-even-digit-sum | 103 | `dup digit-sum 2 mod sub 2 div` |
| mbpp-218 | 105 | `over over gcd div swap drop dec` |
| leetcode-find-the-key-of-the-numbers | 107 | `[ nil swap cons swap cons swap cons over [ div 10 mod ] swap cons map min mul ] swap cons swap cons swap cons nil 1000 cons 100 cons 10 cons 1 cons swap map sum` |
| leetcode-minimum-element-after-replacement-with-digit-sum | 108 | `[ digit-sum ] map min` |
| leetcode-harshad-number | 111 | `dup digit-sum [ mod 0 eq? ] [ swap drop ] [ drop drop -1 ] ifte` |
| mbpp-841 | 115 | `take 0 nil pack2 [ [ unpack2 ] dip over over [ gt? ] swap cons filter len -rot cons -rot add swap pack2 ] fold head` |
| mbpp-564 | 118 | `take nil 0 pack2 [ [ unpack2 ] dip dup [ [ over ] dip [ eq? not ] swap cons filter len add ] dip swap [ cons ] dip pack2 ] fold unpack2 swap drop` |
| mbpp-777 | 123 | `[ dup 0 le? ] [ drop min ] [ over dedup [ [ [ swap [ lt? ] swap cons filter len ] dip lt? ] swap cons swap cons ] dip swap filter sum ] ifte` |
| mbpp-623 | 125 | `[ pow ] swap cons map` |
| leetcode-fibonacci-number | 129 | `range nil 1 cons 0 cons [ drop uncons swap uncons swap drop swap over add nil swap cons swap cons ] fold head` |
| mbpp-524 | 131 | `take nil [ lis-best over add pack2 cons ] fold seconds 0 [ max2 ] fold` |
| mbpp-96 | 131 | `dup range1 swap [ swap mod 0 eq? ] swap cons filter len` |
| mbpp-577 | 138 | `[ dup 0 lt? ] [ ] [ range1 product 10 mod ] ifte` |
| leetcode-find-the-pivot-integer | 141 | `dup dup inc mul 2 div swap inc range swap [ swap sq eq? ] swap cons filter [ null? ] [ drop -1 ] [ head ] ifte` |
| mbpp-589 | 152 | `dup range1 [ sq ] map swap [ le? ] swap cons filter swap [ ge? ] swap cons filter` |
| leetcode-number-of-even-and-odd-bits | 160 | `0 0 rot nil swap cons swap cons swap cons zeros8 swap [ drop uncons swap uncons swap head dup 4 mod dup 2 mod swap 2 div [ rot ] dip add [ rot ] dip [ add ] dip rot 4 div nil swap cons swap cons swap cons ] fold 2 take` |
| mbpp-126 | 168 | `over over min2 range [ 0 gt? ] filter -rot [ [ over mod ] dip [ swap ] dip swap mod add 0 eq? ] swap cons swap cons filter sum` |
| mbpp-644 | 169 | `[ dup 0 eq? ] [ drop dup reverse swap cat ] [ [ dup 0 lt? ] [ over len add 0 max2 ] [ ] ifte over over take reverse -rot drop-n cat ] ifte` |
| mbpp-836 | 180 | `take 1 1000 1000 mul neg 0 0 pack4 [ [ unpack4 ] dip [ inc ] dip swap [ add ] dip [ drop lt? ] [ [ [ drop drop ] dip ] dip swap over over swap ] [ ] ifte [ drop 0 lt? ] [ drop drop 0 0 ] [ ] ifte pack4 ] fold head` |
| leetcode-sum-multiples | 186 | `dup 3 sum-multiples over 5 sum-multiples add over 7 sum-multiples add over 3 5 mul sum-multiples sub over 3 7 mul sum-multiples sub over 5 7 mul sum-multiples sub swap 3 5 mul 7 mul sum-multiples add` |
| leetcode-distribute-candies-among-children-i | 188 | `dup inc range rot rot [ [ swap sub ] dip over over min2 rot rot sub 0 max2 sub inc 0 max2 ] swap cons swap cons map sum` |
| mbpp-142 | 188 | `[ zip ] dip zip [ unpack2 [ unpack2 over ] dip eq? [ eq? ] dip bool>int swap bool>int mul ] map sum` |
| leetcode-number-of-common-factors | 189 | `over over min2 range1 [ [ divides-both ] swap cons swap cons ] dip swap filter len` |
| leetcode-minimum-sum-of-four-digit-number-after-splitting-digits | 191 | `dup 10 mod swap dup 10 div 10 mod swap dup 100 div 10 mod swap 1000 div over over add [ min2 ] dip [ [ over over add ] dip ] dip [ [ [ min2 ] dip ] dip ] dip [ swap ] dip [ [ add ] dip ] dip over over add [ min2 min2 ] dip swap 9 mul add` |
| mbpp-539 | 195 | `zip [ unpack2 pow ] map` |
| mbpp-536 | 206 | `[ dup 0 lt? ] [ neg [ reverse ] dip ] [ ] ifte over len over add dec over div range swap [ mul ] swap cons map swap [ swap nth ] swap cons map` |
| mbpp-429 | 220 | `zip [ unpack2 band ] map` |
| leetcode-sum-of-digits-in-base-k | 222 | `base-digit-sum` |
| leetcode-smallest-divisible-digit-product-i | 228 | `swap 10 range swap [ add ] swap cons map swap [ swap digit-product swap divides? ] swap cons filter head` |
| leetcode-number-of-days-in-a-month | 233 | `[ 2 eq? ] [ drop [ leap? ] [ drop 7 4 mul inc ] [ drop 7 4 mul ] ifte ] [ swap drop [ dup 8 lt? bool>int swap even? bool>int eq? ] [ drop 10 3 mul ] [ drop 10 3 mul inc ] ifte ] ifte` |
| leetcode-two-out-of-three | 250 | `[ [ [ [ dup ] dip swap contains ] dip ] dip [ swap [ [ dup ] dip swap contains ] dip add ] dip swap [ [ dup ] dip swap contains ] dip add swap drop 1 gt? ] swap cons swap cons swap cons 100 range1 swap filter` |
| mbpp-107 | 253 | `range-between [ 0 [ [ drop drop 0 eq? ] [ drop swap drop ] [ [ [ dup 16 mod 10 ge? bool>int ] dip add [ 16 div ] dip ] dip dup i ] ifte ] dup i ] map sum` |
| mbpp-4 | 264 | `over len min2 0 max2 nlargest` |
| mbpp-286 | 276 | `[ take ] dip swap quote [ drop ] swap cat swap range swap map nil [ cat ] fold max-subarray-sum` |
| leetcode-xor-operation-in-an-array | 298 | `over over 2 mod swap 2 mod mul [ 2 div dup [ add dec ] dip dec dup 4 mod [ 0 eq? ] [ drop ] [ [ 1 eq? ] [ drop drop 1 ] [ [ 2 eq? ] [ drop inc ] [ drop drop 0 ] ifte ] ifte ] ifte swap dup 4 mod [ 0 eq? ] [ drop ] [ [ 1 eq? ] [ drop drop 1 ] [ [ 2 eq? ] [ drop inc ] [ drop drop 0 ] ifte ] ifte ] ifte bxor 2 mul ] dip add` |
| mbpp-219 | 311 | `[ sort ] dip over len over sub [ dup len range ] dip2 [ [ over over lt? bool>int ] dip [ swap drop ] dip rot swap ge? bool>int add 0 gt? ] swap cons swap cons filter swap [ swap nth ] swap cons map` |
| mbpp-235 | 351 | `[ dup 0 lt? ] [ ] [ 6 range swap [ over odd? bool>int -rot swap 2 swap pow over over div 2 mod [ over over swap le? bool>int mul swap drop ] dip rot max2 mul ] swap cons map sum ] ifte` |
| mbpp-733 | 401 | `over len dec 0 swap -1 pack3 -rot [ [ [ drop ] dip ] dip [ unpack3 ] dip2 [ drop drop drop le? ] [ [ over over add 2 div ] dip3 rot [ [ dup ] dip2 rot [ swap ] dip nth over over eq? [ ] [ drop drop drop 0 ] [ drop lt? [ ] [ drop 1 ] [ drop 2 ] ifte ] ifte ] dip [ drop 0 eq? ] [ drop drop dup dec [ drop ] dip2 swap ] [ [ drop 1 eq? ] [ [ drop dec swap drop ] dip ] [ [ drop inc swap [ swap drop ] dip ] dip ] ifte ] ifte ] [ drop drop ] ifte pack3 ] swap cons swap cons 7 range -rot fold tail tail head` |
| leetcode-get-maximum-in-generated-array | 424 | `nil 1 cons 0 cons uncons dup quote nil swap cons swap cons [ [ uncons swap head ] dip rot over add rot swap cons over cons nil swap cons swap cons ] fold uncons drop head reverse uncons dup quote nil swap cons swap cons [ [ uncons swap head ] dip rot over add rot swap cons over cons nil swap cons swap cons ] fold uncons drop head reverse uncons dup quote nil swap cons swap cons [ [ uncons swap head ] dip rot over add rot swap cons over cons nil swap cons swap cons ] fold uncons drop head reverse uncons dup quote nil swap cons swap cons [ [ uncons swap head ] dip rot over add rot swap cons over cons nil swap cons swap cons ] fold uncons drop head reverse uncons dup quote nil swap cons swap cons [ [ uncons swap head ] dip rot over add rot swap cons over cons nil swap cons swap cons ] fold uncons drop head reverse uncons dup quote nil swap cons swap cons [ [ uncons swap head ] dip rot over add rot swap cons over cons nil swap cons swap cons ] fold uncons drop head reverse uncons dup quote nil swap cons swap cons [ [ uncons swap head ] dip rot over add rot swap cons over cons nil swap cons swap cons ] fold uncons drop head reverse swap inc take max` |
| mbpp-239 | 459 | `[ dup 0 eq? ] [ drop drop 0 ] [ dec range swap inc range [ drop dup len range swap [ swap 2 div nth ] swap cons map prefix-sums ] fold last ] ifte` |
| mbpp-34 | 459 | `0 swap dec -1 pack3 swap [ swap drop [ unpack3 ] dip [ drop drop le? ] [ [ over over add 2 div ] dip2 swap [ over over swap over over at over inc eq? [ ] [ drop drop drop 0 ] [ drop over over dec at eq? swap drop [ ] [ drop 1 ] [ drop 2 ] ifte ] ifte swap drop ] dip [ drop 0 eq? ] [ [ drop inc swap [ swap drop ] dip ] dip ] [ [ drop 1 eq? ] [ drop drop inc [ swap drop ] dip over inc -rot ] [ [ drop dec swap drop ] dip ] ifte ] ifte ] [ drop ] ifte pack3 ] swap cons 7 range -rot fold tail tail head` |
| leetcode-maximum-height-of-a-triangle | 480 | `over over [ [ [ dup inc 2 div sq ] dip ] dip [ le? ] dip [ swap ] dip [ 2 div dup inc mul ] dip le? [ bool>int ] dip bool>int mul 1 eq? ] swap cons swap cons 10 dup add range1 swap filter 0 [ max2 ] fold [ swap [ [ [ dup inc 2 div sq ] dip ] dip [ le? ] dip [ swap ] dip [ 2 div dup inc mul ] dip le? [ bool>int ] dip bool>int mul 1 eq? ] swap cons swap cons 10 dup add range1 swap filter 0 [ max2 ] fold ] dip max2` |
| leetcode-count-symmetric-integers | 512 | `over over 10 quote [ over over mod [ over over div swap ] dip [ over over mod ] dip [ [ div inc ] dip ] dip [ add ] dip sub ] cat 10 quote cat [ over over div [ [ swap ] dip ] dip [ dup dup dup mul mul [ mul ] dip add ] dip [ add ] dip swap dup ] cat [ swap ] dip swap quote cat [ sub swap ] cat swap quote cat [ swap sub mul ] cat 1000 dup mul 100 mul quote cat [ [ swap dup mul ] dip mul sub inc 0 swap lt? ] cat 9 100 mul range swap filter len [ [ [ [ 10 inc mul ] dip ] dip between ] swap cons swap cons 9 range1 swap filter len ] dip add` |
| leetcode-count-square-sum-triples | 529 | `dup [ [ 4 4 mul over over mod [ div ] dip ] dip [ over over sq swap sq add ] dip le? bool>int [ over over sub odd? bool>int ] dip mul [ over over gt? bool>int ] dip mul [ 1 ge? bool>int ] dip mul swap drop 1 eq? ] swap cons 2 dup mul dup mul dup mul range swap filter 7 dup add range [ 2 add ] map [ [ 4 4 mul over over mod [ div ] dip ] dip [ [ divides-both ] swap cons swap cons ] dip swap filter len 0 eq? ] swap cons filter over [ [ 4 4 mul over over mod [ div ] dip sq swap sq add ] dip swap div ] swap cons map sum 2 mul swap drop` |
| leetcode-count-numbers-with-unique-digits-ii | 536 | `over 10 div over 10 div over sub inc range swap [ add ] swap cons map [ [ [ over 10 mul sub 0 max2 ] dip [ over ] dip swap 10 mul sub 9 min2 [ [ dup ] dip ] dip [ swap ] dip swap dup 100 lt? bool>int over 100 div swap sub over 10 lt? bool>int [ over 10 div 10 mod ] dip 2 mul sub [ [ dup 0 eq? bool>int 3 mul over 10 mod swap sub ] dip ] dip [ [ [ drop ] dip ] dip ] dip nil swap cons swap cons swap cons dup uncons [ uncons ] dip [ [ head ] dip ] dip over over sub [ [ over ] dip swap over sub ] dip mul [ drop ] dip [ sub ] dip mul 0 eq? bool>int [ [ over over ] dip [ [ between ] swap cons swap cons ] dip swap filter len ] dip [ [ swap sub inc ] dip ] dip [ sub ] dip 1 swap sub mul swap drop ] swap cons swap cons ] dip swap map sum` |
| mbpp-846 | 558 | `[ [ sort ] dip sort ] dip dup dup add range [ [ [ [ [ drop ] dip ] dip ] dip [ unpack4 ] dip3 [ [ drop drop ] dip [ drop drop ] dip dup [ lt? ] dip swap [ lt? ] dip bool>int swap bool>int add 2 eq? ] [ drop [ pick4 ] dip [ nth ] dip pick4 nth le? [ ] [ drop [ [ inc ] dip2 inc ] dip ] [ drop [ [ inc ] dip dec ] dip ] ifte over max2 ] [ drop drop drop ] ifte pack4 ] swap cons swap cons swap cons ] dip 1 0 1 1 pack4 rot fold last` |
| mbpp-656 | 564 | `dup [ nsmallest ] dip [ swap ] dip nsmallest zip [ unpack2 sub abs ] map sum` |
| mbpp-799 | 595 | `over over 2 swap pow mul [ 16 2 mul swap sub 2 swap pow div ] dip [ dup 0 lt? ] [ [ neg dec ] dip neg dec band neg dec ] [ over over band [ add ] dip sub ] ifte` |
| mbpp-466 | 712 | `0 over dec nil pack3 -rot [ [ [ drop ] dip ] dip [ unpack3 ] dip2 [ drop drop nil eq? ] [ [ over over add [ dup 0 lt? ] [ neg 2 div neg ] [ 2 div ] ifte ] dip3 rot [ [ dup ] dip2 rot dup [ [ over ] dip [ 0 eq? ] [ drop drop true ] [ over over dec at [ at ] dip swap le? ] ifte ] dip swap [ ] [ drop [ swap dec eq? ] [ drop drop drop true ] [ swap drop over over inc at [ at ] dip swap le? ] ifte [ ] [ drop 0 ] [ drop 2 ] ifte ] [ drop [ dup 0 gt? ] [ drop drop drop 1 ] [ drop drop drop 2 ] ifte ] ifte ] dip [ drop 0 eq? ] [ drop drop ] [ [ drop 1 eq? ] [ [ drop dec swap drop ] dip ] [ [ drop inc swap [ swap drop ] dip ] dip ] ifte ] ifte ] [ drop drop ] ifte pack3 ] swap cons swap cons 7 range -rot fold tail tail head` |
| leetcode-count-operations-to-obtain-zero | 916 | `0 [ drop 0 eq? ] [ ] [ [ over over div ] dip add [ over over mod ] dip [ [ swap drop ] dip ] dip ] ifte [ drop 0 eq? ] [ ] [ [ over over div ] dip add [ over over mod ] dip [ [ swap drop ] dip ] dip ] ifte [ drop 0 eq? ] [ ] [ [ over over div ] dip add [ over over mod ] dip [ [ swap drop ] dip ] dip ] ifte [ drop 0 eq? ] [ ] [ [ over over div ] dip add [ over over mod ] dip [ [ swap drop ] dip ] dip ] ifte [ drop 0 eq? ] [ ] [ [ over over div ] dip add [ over over mod ] dip [ [ swap drop ] dip ] dip ] ifte [ drop 0 eq? ] [ ] [ [ over over div ] dip add [ over over mod ] dip [ [ swap drop ] dip ] dip ] ifte [ drop 0 eq? ] [ ] [ [ over over div ] dip add [ over over mod ] dip [ [ swap drop ] dip ] dip ] ifte [ drop 0 eq? ] [ ] [ [ over over div ] dip add [ over over mod ] dip [ [ swap drop ] dip ] dip ] ifte [ drop 0 eq? ] [ ] [ [ over over div ] dip add [ over over mod ] dip [ [ swap drop ] dip ] dip ] ifte [ drop 0 eq? ] [ ] [ [ over over div ] dip add [ over over mod ] dip [ [ swap drop ] dip ] dip ] ifte [ drop 0 eq? ] [ ] [ [ over over div ] dip add [ over over mod ] dip [ [ swap drop ] dip ] dip ] ifte [ drop 0 eq? ] [ ] [ [ over over div ] dip add [ over over mod ] dip [ [ swap drop ] dip ] dip ] ifte [ drop 0 eq? ] [ ] [ [ over over div ] dip add [ over over mod ] dip [ [ swap drop ] dip ] dip ] ifte [ drop 0 eq? ] [ ] [ [ over over div ] dip add [ over over mod ] dip [ [ swap drop ] dip ] dip ] ifte [ drop 0 eq? ] [ ] [ [ over over div ] dip add [ over over mod ] dip [ [ swap drop ] dip ] dip ] ifte [ drop 0 eq? ] [ ] [ [ over over div ] dip add [ over over mod ] dip [ [ swap drop ] dip ] dip ] ifte [ drop 0 eq? ] [ ] [ [ over over div ] dip add [ over over mod ] dip [ [ swap drop ] dip ] dip ] ifte [ drop 0 eq? ] [ ] [ [ over over div ] dip add [ over over mod ] dip [ [ swap drop ] dip ] dip ] ifte [ drop 0 eq? ] [ ] [ [ over over div ] dip add [ over over mod ] dip [ [ swap drop ] dip ] dip ] ifte [ drop 0 eq? ] [ ] [ [ over over div ] dip add [ over over mod ] dip [ [ swap drop ] dip ] dip ] ifte [ drop 0 eq? ] [ ] [ [ over over div ] dip add [ over over mod ] dip [ [ swap drop ] dip ] dip ] ifte [ drop 0 eq? ] [ ] [ [ over over div ] dip add [ over over mod ] dip [ [ swap drop ] dip ] dip ] ifte [ drop 0 eq? ] [ ] [ [ over over div ] dip add [ over over mod ] dip [ [ swap drop ] dip ] dip ] ifte [ drop 0 eq? ] [ ] [ [ over over div ] dip add [ over over mod ] dip [ [ swap drop ] dip ] dip ] ifte [ drop 0 eq? ] [ ] [ [ over over div ] dip add [ over over mod ] dip [ [ swap drop ] dip ] dip ] ifte [ drop 0 eq? ] [ ] [ [ over over div ] dip add [ over over mod ] dip [ [ swap drop ] dip ] dip ] ifte [ drop drop ] dip` |
