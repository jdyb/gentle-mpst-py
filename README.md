
# Gentle-MPST-Py

This is an educational implementation of
"A Very Gentle Introduction to Multiparty Session Types" by Y. Yoshida and L. Gheri.
The implementation has been tested with Python 3.6 on Ubuntu 18.04.5 and
mypy 0.79. It probably works with later Python versions as well. Mypy is
not required to run the program.

This is not a really intended to be a tool or library, it is just a single-file
program that contains implementations of the algorithms and the examples
provided in the paper.

If you are going to study this program you should do so together with the
paper, studying only this program will probably not be very productive. 

# Current status

There are still large parts of the paper that have not been implemented yet.

## Work in progress:

* "3 Synchronous Multiparty Session Calculus"

Partial implementation of the operational semantics of multiparty sessions.
Example 2 is implemented.

* "4.1 Types and Projections"

Merging with projection is implemented and the examples produce expected
results.

## Remaining sections:

* "4.2 Subtyping": Not started.
* "4.3 Type System": Not started.

## Examples:

1. No reductions, so not implemented.
2. Implemented in `example_2()`
3. Not implemented (skipped non-merging projection).
4. Implemented in `example_4()`
5. Implemented in `section_4_1_example_5().
6. Implemented in `example\_6\_1()` and `example\_6\_2()`
7. Not started.
8. Not started.
9. Not started.

## Exercises:

1. Not started.
2. Not started.
3. Not started.

# Contributing

Bug reports, fixes and additional examples are most welcome! Please open an
issue, pull request or send me an email (jon \<at\> dybeck \<dot\> eu).
Contributions that add language features are not likely to be merged since
this project is intended to closely match the paper. 

# License

This implementation is provided under the MIT license. Please see the LICENSE
file for the full text.

