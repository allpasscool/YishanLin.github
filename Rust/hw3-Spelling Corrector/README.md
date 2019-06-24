# Homework 2: Spelling Corrector

[![Travis CI status](https://travis-ci.com/nu-rust-course/hw3-david_yishan.svg?token=Ase2AYhgcqMDkdsyq9m2&branch=develop)](https://travis-ci.com/nu-rust-course/hw3-david_yishan) [![Codecov.io status](https://codecov.io/gh/nu-rust-course/hw3-david_yishan/branch/develop/graph/badge.svg?token=8j9xKXzo0p)](https://codecov.io/gh/nu-rust-course/hw3-david_yishan/branch/develop)

For this homework, you will design and implement a statistical spelling correction program, `correct`, based on [an idea of Peter Norvig’s](http://norvig.com/spell-correct.html). The objective is to deepen your knowledge of Rust further, and in particular to start thinking about data structures and performance.


## The concept

The purpose of `correct` is to find possible corrections for misspelled words. It consists of two phases: The first phase is training, which consumes a corpus of words presumed to be correctly spelled and builds a model (as in, it counts the number of occurrences of each word). The second phase uses the model to check individual words. Specifically, it checks whether each word is spelled correctly according to the training and, if not, whether “small edits” can reach a variant that is correctly spelled.

Given a word, an edit action is one of the following:

  - the deletion of one letter;

  - the transposition of two neighboring letters;

  - the replacement of one letter with another letter; and

  - the insertion of a letter at any position.

In this context, Norvig suggests that “small edits” means the application of one edit action possibly followed by the application of a second one to the result of the first.

Once the second part has considered all possible candidate for a potentially misspelled word, it picks the most frequently used one from the training corpus. If none of the candidates is a correct word, `correct` reports a failure.


## Usage details

The `correct` program takes the name of the training file on the command line and then reads words to correct—one per line—from standard input. For each word from standard in, `correct` prints one line. The line consists of just the word if it is spelled correctly. If the word is not correctly spelled, `correct` prints the word and its proposed improvement. If no improvement is found then it prints the misspelled word and “-” in place of a suggestion.

For example:

````
$ cat corpus.txt
hello world hello word hello world
$ cat input.txt
hello
hell
word
wordl
wor
wo
w
$ cargo run corpus.txt < input.txt
hello
hell, hello
word
wordl, world
wor, world
wo, word
w, -
````

The program must work interactively, in the sense that it prints the response to each line immediately after reading that line.


## Internal requirements

Your spelling corrector must be factored into a library implementing the model and a client program that depends on it. The library must encapsulate the model represention and offer operations for both the training and correction phases.

Here is the kind of API I expect to see:

```rust
pub struct CorrectorModel { … }

pub enum Correction {
    Correct,
    Incorrect,
    Suggestion(String)
}

impl CorrectorModel {
    pub fn new() -> Self;
    pub fn learn(&mut self, word: &str);
    pub suggest(&self, candidate: &str) -> Correction;
}
```

The above API assumes that:

 1. The `CorrectorModel` itself does not store the strings as `String`s, but some other way.

 2. The `CorrectorModel` supports adding words incrementally as opposed to having to build it all at once

If the assumption 1 is untrue and it does hold actual `String`s, then `CorrectorModel::learn` should take a `String` and `Correction` should borrow instead:

```rust
pub enum Correction<'a> {
    Correct,
    Incorrect,
    Suggestion(&'a str)
}
```

If assumption 2 doesn't hold then you should get rid of the `CorrectorModel::learn` method and change `CorrectorModel::new` to take the corpus like this:

```rust
impl CorrectorModel {
    pub fn new<I>(corpus: I) -> Self
        where I: IntoIterator<Item = ¯\_(ツ)_/¯>;
}
```

where `¯\_(ツ)_/¯` is `String` or `&str` depending on assumption 1.


## Performance

The combinatorics of this problem, considered naïvely, may be prohibitive. In particular, consider a *k*-letter word. The number of possible letter replacements is *kd*, where *d* is the size of the alphabet. And because we are applying two edits, there are *kd* words one step further away. Ignoring redundancy, this suggests that we have to check ~*k²d²*, and that’s not considering the other three kinds of edits. Thus, actually generating each edited word and then checking it in a hash table will result in poor performance. Similarly, measuring the distance between the word and each word in the corpus will take a long time.

So, is there a better:way? (Yes, much better.) But how? Two possibilities are the [trie](https://en.wikipedia.org/wiki/Trie) and the [BK-tree](https://en.wikipedia.org/wiki/BK-tree).


## Evaluation

Your grade will be based on:

  - correctness (how closely your program adheres to its specification),
  - style (expecting somewhat idiomatic Rust now, and also good factoring),
  - testing (make sure you have good test coverage), and
  - efficiency (choose sensible data structures, avoid needless copying, and you may want to benchmark or profile).
