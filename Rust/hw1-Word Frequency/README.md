# Homework 1: Word Frequencies

[![Travis CI status](https://travis-ci.com/nu-rust-course/hw1-tristan_yishan.svg?token=Ase2AYhgcqMDkdsyq9m2&branch=master)](https://travis-ci.com/nu-rust-course/hw1-tristan_yishan) [![Codecov.io status](https://codecov.io/gh/nu-rust-course/hw1-tristan_yishan/branch/master/graph/badge.svg?token=2ogw6qJhPC)](https://codecov.io/gh/nu-rust-course/hw1-tristan_yishan/branch/master)

For this homework you will design and implement a word frequency counting program, `freq`, in Rust. The objective of this homework is to get you writing Rust code, and to help you gain familiarity with Rust’s iterators, traits, and data structures.

Your freq program consumes text from standard input. It prints a list of word-frequency counts, in descending order of frequency, to the standard output.

```
$ cat test.txt
hello world,
bye world

$ ./freq < test.txt
world: 2
bye: 1
hello: 1
```

You may wish to run your freq program on [some larger inputs](http://www.gutenberg.org/files/11/11.txt).

The specification is intentionally incomplete, so you will have to make some assumptions in your design. Please list your assumptions in a doc comment giving usage information at the top of `main.rs`.

### Evaluation

Your grade will be based on:
 - correctness (how closely your program adheres to its specification),
 - style (not expecting the most idiomatic Rust at this point, but I’ll be looking for good factoring—don’t put everything in main),
 - testing (make sure you have good test coverage), and
 - efficiency (no need to benchmark or profile, but do choose sensible data structures and avoid needless copying).
