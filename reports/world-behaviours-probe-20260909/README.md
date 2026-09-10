# World-behaviours probe: can MBPP and LeetCode-easy behaviours get `mark-28` witnesses?

training-data-sources (#118). Behaviours sampled by `experiments/world_behaviours/extract.py`
(seed 17, ints and flat int lists only, ≤ 3 arguments); programs authored by Claude in macro
form and expanded to `mark-28`; every program re-verified on every test by `report.py`.

## Findings (written with the 2026-09-09 run; the tables below regenerate)

- **Feasibility is not the gap.** 91 of 95 verified. The four misses are one
  behaviour that is CPython set-iteration order rather than a function, and
  three LeetCode problems whose tests span up to 90,002 numbers against the
  100,000-step fuel cap. Nothing failed for want of an op.
- **Raw witnesses are longer than v4's, not shorter.** Median 102 `mark-28`
  ops against v4's 44, and 66 of 91 sit in the 41+ bucket. The same programs
  are a median 17 tokens in macro form (4.6 ops per token), so the length is
  the basis's, not the behaviours': every author ended up with a library of
  stack shuffles, packed accumulators, index-list builders and digit helpers.
  That library is the object highways (#96) proposes to mine; here it was
  written top-down in an afternoon.
- **Loops are capped by nesting, not fuel.** `MAX_QUOTATION_DEPTH = 32` and a
  self-applying quotation nests ~4 levels per iteration, so the recursive
  `range` in the shared macros refuses past n = 7. All four authors found this
  independently and rebuilt ranges as recursion-free doubling folds.
- **Six solves replay the reference's quirks, not the stated behaviour**
  (`mbpp-34`, `mbpp-733`, `mbpp-466` binary searches on unsorted input,
  `mbpp-824` mutate-while-iterating, `mbpp-846` merge replay, `mbpp-799`
  bit-width bound). They pass every test because the tests were drawn from
  that reference. A corpus built this way inherits the reference as the
  behaviour, which is the right reading of world-authored data but the wrong
  one of the description.
- **Overlap with v4 is small**: 4 of 91 expanded programs equal a v4 witness
  (`add`, `product`, `max2`, and the range of a list), and none of the rest
  share a skeleton.

## Verified

| source | verified | median witness length (`mark-28` ops) | median macro-form length |
|---|---|---|---|
| leetcode | 42/45 (93%) | 99.5 | 23.5 |
| mbpp | 49/50 (98%) | 115 | 13 |
| **all** | 91/95 (96%) | 102 | 17 |

Claimed-vs-verified mismatches: 0; missing ids: 0; duplicates: 0.

## Witness length against `stack_pbe_v4`

| bucket (ops) | this probe (verified) | v4 behaviours |
|---|---|---|
| 1-5 | 6 | 9 |
| 6-10 | 3 | 11 |
| 11-20 | 8 | 28 |
| 21-40 | 8 | 28 |
| 41+ | 66 | 99 |
| median | 102 | 44 |

Exact matches with a v4 witness: 4 (mbpp-925, mbpp-309, mbpp-588, leetcode-add-two-integers). Same skeleton as a v4 witness: 4.

## Macros

129 macros defined across `macros*.txt`; median expansion ratio (`mark-28` ops per macro-form token) on verified programs: 4.6.

| macro | verified programs using it |
|---|---|
| `2` | 22 |
| `inc` | 18 |
| `mod` | 18 |
| `dec` | 17 |
| `head` | 13 |
| `len` | 13 |
| `max2` | 13 |
| `sum` | 11 |
| `10` | 10 |
| `le?` | 9 |
| `-1` | 8 |
| `7` | 8 |
| `bool>int` | 8 |
| `take` | 8 |
| `min2` | 8 |
| `k1--rot` | 7 |
| `100` | 7 |
| `odd?` | 6 |
| `k2-unpair` | 6 |
| `4` | 6 |

## Unsolved

| id | source | passed | note |
|---|---|---|---|
| mbpp-579 | mbpp | 0/27 | output order is CPython set iteration order (tuple(set(a)^set(b))), not sorted; reproducing it means simulating the hash |
| leetcode-maximum-number-of-balls-in-a-box | leetcode | 0/40 | fuel: ranges hold up to 90002 numbers (100k step cap); histogram over 46 digit sums has no closed form the basis affords |
| leetcode-count-largest-group | leetcode | 0/40 | unsolved: n up to 10000 needs a digit-sum histogram over every 1..n; k4-digit-sum is ~350 fuel each (3.5M total vs 100k) |
| leetcode-convert-integer-to-the-sum-of-two-no-zero-integers | leetcode | 0/40 | unsolved on fuel: linear search needs >=611 candidates (test 10606) x two zero-free checks; k4-digit-product is ~380 fue |

## Every verified program

| id | len | macro form |
|---|---|---|
| leetcode-add-two-integers | 1 | `add` |
| mbpp-176 | 2 | `add add` |
| mbpp-373 | 2 | `mul mul` |
| leetcode-count-of-matches-in-tournament | 2 | `dec` |
| mbpp-925 | 5 | `product` |
| leetcode-find-the-maximum-achievable-number | 5 | `2 mul add` |
| mbpp-646 | 7 | `sub inc dup sq mul` |
| mbpp-171 | 8 | `5 mul` |
| leetcode-water-bottles | 8 | `dec over dec swap div add` |
| mbpp-970 | 12 | `min2` |
| mbpp-309 | 12 | `max2` |
| mbpp-41 | 13 | `[ even? ] filter` |
| leetcode-count-distinct-numbers-on-board | 15 | `dec 1 max2` |
| mbpp-313 | 19 | `[ 0 ge? ] filter head` |
| leetcode-calculate-delayed-arrival-time | 19 | `add 6 4 mul mod` |
| leetcode-smallest-even-multiple | 19 | `[ even? ] [ ] [ dup add ] ifte` |
| leetcode-divisible-and-non-divisible-sums-difference | 20 | `over over div dup inc mul mul [ dup inc mul 2 div ] dip sub` |
| mbpp-572 | 22 | `dup [ swap k2-count 1 eq? ] swap cons filter` |
| mbpp-962 | 23 | `[ dec 2 div k1-sumnat ] dip 2 div k1-sumnat swap sub` |
| mbpp-807 | 29 | `[ odd? ] filter [ dup null? ] [ drop -1 ] [ head ] ifte` |
| leetcode-minimum-cuts-to-divide-a-circle | 32 | `[ 1 eq? ] [ drop 0 ] [ [ odd? ] [ ] [ 2 div ] ifte ] ifte` |
| leetcode-pass-the-pillow | 34 | `over dec 2 mul mod [ over lt? ] [ swap drop inc ] [ swap 2 mul dec swap sub ] ifte` |
| mbpp-588 | 35 | `dup max swap min sub` |
| mbpp-249 | 36 | `swap [ swap contains 1 eq? ] swap cons filter` |
| mbpp-218 | 39 | `over over k1-gcd div swap drop dec` |
| leetcode-find-the-child-who-has-the-ball-after-k-seconds | 41 | `over dec over over mod [ div ] dip [ drop odd? ] [ swap drop sub dec ] [ swap drop swap drop ] ifte` |
| leetcode-account-balance-after-rounded-purchase | 41 | `5 add 10 div 10 mul 100 swap sub` |
| leetcode-minimum-amount-of-time-to-fill-cups | 41 | `dup max swap sum inc 2 div max2` |
| mbpp-591 | 42 | `[ tail null? ] [ ] [ dup last swap dup head swap tail reverse tail reverse swap quote cat swap cons ] ifte` |
| leetcode-total-distance-traveled | 45 | `over dec 4 div 0 max2 min2 add 10 mul` |
| leetcode-smallest-number-with-all-set-bits | 50 | `[ swap drop [ over dec swap lt? ] [ drop dup add ] [ drop ] ifte ] swap cons k4-junk16 1 k4-rot fold dec` |
| mbpp-623 | 51 | `[ k1-pow ] swap cons map` |
| mbpp-152 | 57 | `k1-sort` |
| mbpp-54 | 57 | `k1-sort` |
| mbpp-366 | 57 | `uncons 0 k2-pair [ [ k2-unpair ] dip dup [ [ over ] dip mul max2 ] dip swap [ swap drop ] dip k2-pair ] fold k2-unpair swap drop` |
| mbpp-663 | 58 | `[ over over le? ] [ drop drop drop -1 ] [ [ [ over ] dip lt? ] [ drop drop drop -1 ] [ [ over ] dip sub swap mod sub ] ifte ] ifte` |
| mbpp-824 | 58 | `nil 0 cons [ [ uncons ] dip [ drop 1 eq? ] [ swap drop cons 0 cons ] [ [ odd? ] [ swap drop cons 0 cons ] [ drop drop 1 cons ] ifte ] ifte ] fold uncons drop reverse` |
| leetcode-find-the-distance-value-between-two-arrays | 64 | `[ [ swap ] dip [ [ sub abs ] dip le? ] swap cons swap cons filter len 0 eq? ] swap cons swap cons filter len` |
| leetcode-calculate-money-in-leetcode-bank | 72 | `dup 7 div swap 7 mod over 2 mul over add inc over mul 2 div [ drop dup 7 add mul 7 mul 2 div ] dip add` |
| leetcode-distribute-money-to-maximum-children | 73 | `[ lt? ] [ drop drop -1 ] [ [ 8 mul swap lt? ] [ swap drop dec ] [ [ 8 mul 4 sub eq? ] [ swap drop 2 sub ] [ sub 7 div ] ifte ] ifte ] ifte` |
| mbpp-548 | 81 | `nil [ k2-lis-best inc k2-pair cons ] fold k2-seconds 0 [ max2 ] fold` |
| leetcode-number-of-students-doing-homework-at-a-given-time | 92 | `dup [ le? bool>int ] swap cons swap [ ge? bool>int ] swap cons [ swap ] dip map [ map ] dip k3-dot` |
| leetcode-sum-of-digits-in-base-k | 92 | `k3-basesum` |
| leetcode-fibonacci-number | 98 | `k4-iota128 nil 1 cons 0 cons [ drop uncons swap uncons swap drop swap over add nil swap cons swap cons ] fold head` |
| leetcode-count-integers-with-even-digit-sum | 101 | `dup k4-digit-sum 2 mod sub 2 div` |
| leetcode-form-smallest-number-from-two-digit-arrays | 102 | `[ swap [ [ over over eq? ] [ drop ] [ over over min2 10 mul [ max2 ] dip add ] ifte ] swap cons map min min2 ] swap cons 100 swap fold` |
| leetcode-find-the-key-of-the-numbers | 107 | `[ nil swap cons swap cons swap cons over [ div 10 mod ] swap cons map min mul ] swap cons swap cons swap cons nil 1000 cons 100 cons 10 cons 1 cons swap map sum` |
| mbpp-777 | 111 | `[ dup 0 eq? ] [ drop min ] [ over k2-dedup [ [ [ swap [ lt? ] swap cons filter len ] dip lt? ] swap cons swap cons ] dip swap filter sum ] ifte` |
| mbpp-841 | 115 | `take 0 nil k1-pack2 [ [ k1-unpack2 ] dip over over [ gt? ] swap cons filter len k1--rot cons k1--rot add swap k1-pack2 ] fold head` |
| leetcode-find-the-pivot-integer | 116 | `dup dup inc mul 2 div swap inc k4-iota swap [ swap sq eq? ] swap cons filter [ null? ] [ drop -1 ] [ head ] ifte` |
| mbpp-644 | 118 | `over over take reverse k1--rot drop-n cat` |
| mbpp-564 | 118 | `take nil 0 k2-pair [ [ k2-unpair ] dip dup [ [ over ] dip [ eq? not ] swap cons filter len add ] dip swap [ cons ] dip k2-pair ] fold k2-unpair swap drop` |
| leetcode-minimum-element-after-replacement-with-digit-sum | 120 | `[ k3-digitsum ] map min` |
| leetcode-harshad-number | 123 | `dup k3-digitsum [ mod 0 eq? ] [ swap drop ] [ drop drop -1 ] ifte` |
| mbpp-524 | 131 | `take nil [ k2-lis-best over add k2-pair cons ] fold k2-seconds 0 [ max2 ] fold` |
| leetcode-distribute-candies-among-children-i | 157 | `dup inc k4-iota128 k4-rot k4-rot [ [ swap sub ] dip over over min2 k4-rot k4-rot sub 0 max2 sub inc 0 max2 ] swap cons swap cons map sum` |
| leetcode-number-of-even-and-odd-bits | 160 | `0 0 k4-rot nil swap cons swap cons swap cons k4-junk8 swap [ drop uncons swap uncons swap head dup 4 mod dup 2 mod swap 2 div [ k4-rot ] dip add [ k4-rot ] dip [ add ] dip k4-rot 4 div nil swap cons swap cons swap cons ] fold 2 take` |
| mbpp-577 | 161 | `k1-range1 product 10 mod` |
| mbpp-836 | 180 | `take 1 1000 1000 mul neg 0 0 k1-pack4 [ [ k1-unpack4 ] dip [ inc ] dip swap [ add ] dip [ drop lt? ] [ [ [ drop drop ] dip ] dip swap over over swap ] [ ] ifte [ drop 0 lt? ] [ drop drop 0 0 ] [ ] ifte k1-pack4 ] fold head` |
| mbpp-589 | 185 | `dup k1-range1 [ sq ] map swap [ le? ] swap cons filter swap [ ge? ] swap cons filter` |
| leetcode-sum-multiples | 186 | `dup 3 k4-sum-multiples-of over 5 k4-sum-multiples-of add over 7 k4-sum-multiples-of add over 3 5 mul k4-sum-multiples-of sub over 3 7 mul k4-sum-multiples-of sub over 5 7 mul k4-sum-multiples-of sub swap 3 5 mul 7 mul k4-sum-multiples-of add` |
| mbpp-142 | 188 | `[ k2-zip ] dip k2-zip [ k2-unpair [ k2-unpair over ] dip eq? [ eq? ] dip bool>int swap bool>int mul ] map sum` |
| leetcode-smallest-divisible-digit-product-i | 189 | `swap 10 k4-iota16 swap [ add ] swap cons map swap [ swap k4-digit-product swap k4-divisible? ] swap cons filter head` |
| leetcode-minimum-sum-of-four-digit-number-after-splitting-digits | 191 | `dup 10 mod swap dup 10 div 10 mod swap dup 100 div 10 mod swap 1000 div over over add [ min2 ] dip [ [ over over add ] dip ] dip [ [ [ min2 ] dip ] dip ] dip [ swap ] dip [ [ add ] dip ] dip over over add [ min2 min2 ] dip swap 9 mul add` |
| mbpp-235 | 193 | `6 range swap [ over odd? bool>int k1--rot swap 2 swap k1-pow over over div 2 mod [ over over swap le? bool>int mul swap drop ] dip k1-rot max2 mul ] swap cons map sum` |
| mbpp-429 | 194 | `k2-zip [ k2-unpair k2-band ] map` |
| mbpp-126 | 201 | `over over min2 k1-range [ 0 gt? ] filter k1--rot [ [ over mod ] dip [ swap ] dip swap mod add 0 eq? ] swap cons swap cons filter sum` |
| mbpp-536 | 218 | `over len over add dec over div k1-range swap [ mul ] swap cons map swap [ swap nth ] swap cons map` |
| mbpp-799 | 222 | `2 swap k2-pow mul` |
| mbpp-96 | 229 | `dup k2-range1 swap [ swap mod 0 eq? ] swap cons filter len` |
| leetcode-number-of-days-in-a-month | 233 | `[ 2 eq? ] [ drop [ k4-leap? ] [ drop 7 4 mul inc ] [ drop 7 4 mul ] ifte ] [ swap drop [ dup 8 lt? bool>int swap even? bool>int eq? ] [ drop 10 3 mul ] [ drop 10 3 mul inc ] ifte ] ifte` |
| leetcode-number-of-common-factors | 255 | `over over min2 k3-range1 [ [ k3-divides-both ] swap cons swap cons ] dip swap filter len` |
| leetcode-xor-operation-in-an-array | 279 | `over over 2 mod swap 2 mod mul [ 2 div dup [ add dec ] dip dec k3-xorrange swap k3-xorrange k3-xor 2 mul ] dip add` |
| mbpp-539 | 292 | `k2-zip [ k2-unpair k2-pow ] map` |
| leetcode-two-out-of-three | 316 | `[ [ [ k3-has ] dip ] dip [ swap [ k3-has ] dip add ] dip swap [ k3-has ] dip add swap drop 1 gt? ] swap cons swap cons swap cons 100 k3-range1 swap filter` |
| mbpp-733 | 327 | `over len dec 0 swap -1 k1-pack3 k1--rot [ [ [ drop ] dip ] dip [ k1-unpack3 ] k1-dip2 [ drop drop drop le? ] [ [ over over add 2 div ] k1-dip3 k1-rot [ [ dup ] k1-dip2 k1-rot k1-code733 ] dip [ drop 0 eq? ] [ drop drop dup dec [ drop ] k1-dip2 swap ] [ [ drop 1 eq? ] [ [ drop dec swap drop ] dip ] [ [ drop inc swap [ swap drop ] dip ] dip ] ifte ] ifte ] [ drop drop ] ifte k1-pack3 ] swap cons swap cons 7 range k1--rot fold tail tail head` |
| mbpp-4 | 329 | `k2-nlargest` |
| mbpp-219 | 344 | `[ k1-sort ] dip over len over sub [ dup len k1-range ] k1-dip2 [ [ over over lt? bool>int ] dip [ swap drop ] dip k1-rot swap ge? bool>int add 0 gt? ] swap cons swap cons filter swap [ swap nth ] swap cons map` |
| mbpp-107 | 351 | `k2-range-ab [ k2-hexcount ] map sum` |
| mbpp-286 | 374 | `[ take ] dip swap quote [ drop ] swap cat swap k2-range swap map nil [ cat ] fold k2-kadane` |
| mbpp-34 | 385 | `0 swap dec -1 k1-pack3 swap [ swap drop [ k1-unpack3 ] dip [ drop drop le? ] [ [ over over add 2 div ] k1-dip2 swap [ over over swap k1-code34 swap drop ] dip [ drop 0 eq? ] [ [ drop inc swap [ swap drop ] dip ] dip ] [ [ drop 1 eq? ] [ drop drop inc [ swap drop ] dip over inc k1--rot ] [ [ drop dec swap drop ] dip ] ifte ] ifte ] [ drop ] ifte k1-pack3 ] swap cons 7 range k1--rot fold tail tail head` |
| leetcode-get-maximum-in-generated-array | 424 | `nil 1 cons 0 cons k4-stern-double k4-stern-double k4-stern-double k4-stern-double k4-stern-double k4-stern-double k4-stern-double swap inc take max` |
| mbpp-466 | 529 | `0 over dec -1 k1-pack3 k1--rot [ [ [ drop ] dip ] dip [ k1-unpack3 ] k1-dip2 [ drop drop -1 eq? ] [ [ over over add 2 div ] k1-dip3 k1-rot [ [ dup ] k1-dip2 k1-rot k1-code466 ] dip [ drop 0 eq? ] [ drop drop ] [ [ drop 1 eq? ] [ [ drop dec swap drop ] dip ] [ [ drop inc swap [ swap drop ] dip ] dip ] ifte ] ifte ] [ drop drop ] ifte k1-pack3 ] swap cons swap cons 7 range k1--rot fold tail tail head` |
| mbpp-239 | 558 | `[ dup 0 eq? ] [ drop drop 0 ] [ dec k1-range swap inc k1-range [ drop dup len k1-range swap [ swap 2 div nth ] swap cons map k1-prefix ] fold last ] ifte` |
| mbpp-846 | 591 | `[ [ k1-sort ] dip k1-sort ] dip dup dup add k1-range [ [ [ [ [ drop ] dip ] dip ] dip [ k1-unpack4 ] k1-dip3 [ [ drop drop ] dip [ drop drop ] dip dup [ lt? ] dip swap [ lt? ] dip bool>int swap bool>int add 2 eq? ] [ drop [ k1-pick4 ] dip [ nth ] dip k1-pick4 nth le? [ ] [ drop [ [ inc ] k1-dip2 inc ] dip ] [ drop [ [ inc ] dip dec ] dip ] ifte over max2 ] [ drop drop drop ] ifte k1-pack4 ] swap cons swap cons swap cons ] dip 1 0 1 1 k1-pack4 k1-rot fold last` |
| leetcode-count-numbers-with-unique-digits-ii | 602 | `over 10 div over 10 div over sub inc k3-range swap [ add ] swap cons map [ [ [ over 10 mul sub 0 max2 ] dip [ over ] dip swap 10 mul sub 9 min2 [ [ dup ] dip ] dip [ swap ] dip swap dup 100 lt? bool>int over 100 div swap sub over 10 lt? bool>int [ over 10 div 10 mod ] dip 2 mul sub [ [ dup 0 eq? bool>int 3 mul over 10 mod swap sub ] dip ] dip [ [ [ drop ] dip ] dip ] dip nil swap cons swap cons swap cons dup uncons [ uncons ] dip [ [ head ] dip ] dip k3-pdiff3 0 eq? bool>int [ [ over over ] dip [ [ k3-between ] swap cons swap cons ] dip swap filter len ] dip [ [ swap sub inc ] dip ] dip [ sub ] dip 1 swap sub mul swap drop ] swap cons swap cons ] dip swap map sum` |
| leetcode-maximum-height-of-a-triangle | 612 | `over over k3-trih [ swap k3-trih ] dip max2` |
| leetcode-count-symmetric-integers | 644 | `over over 10 quote [ over over mod [ over over div swap ] dip [ over over mod ] dip [ [ div inc ] dip ] dip [ add ] dip sub ] cat 10 quote cat [ over over div [ [ swap ] dip ] dip [ dup dup dup mul mul [ mul ] dip add ] dip [ add ] dip swap dup ] cat [ swap ] dip swap quote cat [ sub swap ] cat swap quote cat [ swap sub mul ] cat 1000 dup mul 100 mul quote cat [ [ swap dup mul ] dip mul sub inc 0 swap lt? ] cat 9 100 mul k3-range swap filter len [ [ [ [ 10 inc mul ] dip ] dip k3-between ] swap cons swap cons 9 k3-range1 swap filter len ] dip add` |
| leetcode-count-square-sum-triples | 661 | `dup [ [ 4 4 mul over over mod [ div ] dip ] dip [ over over sq swap sq add ] dip le? bool>int [ over over sub odd? bool>int ] dip mul [ over over gt? bool>int ] dip mul [ 1 ge? bool>int ] dip mul swap drop 1 eq? ] swap cons 2 dup mul dup mul dup mul k3-range swap filter 7 dup add k3-range [ 2 add ] map [ [ 4 4 mul over over mod [ div ] dip ] dip [ [ k3-divides-both ] swap cons swap cons ] dip swap filter len 0 eq? ] swap cons filter over [ [ 4 4 mul over over mod [ div ] dip sq swap sq add ] dip swap div ] swap cons map sum 2 mul swap drop` |
| mbpp-656 | 760 | `dup [ k2-nsmallest ] dip [ swap ] dip k2-nsmallest k2-zip [ k2-unpair sub abs ] map sum` |
| leetcode-count-operations-to-obtain-zero | 916 | `0 k3-euclid-step k3-euclid-step k3-euclid-step k3-euclid-step k3-euclid-step k3-euclid-step k3-euclid-step k3-euclid-step k3-euclid-step k3-euclid-step k3-euclid-step k3-euclid-step k3-euclid-step k3-euclid-step k3-euclid-step k3-euclid-step k3-euclid-step k3-euclid-step k3-euclid-step k3-euclid-step k3-euclid-step k3-euclid-step k3-euclid-step k3-euclid-step k3-euclid-step k3-euclid-step [ drop drop ] dip` |
