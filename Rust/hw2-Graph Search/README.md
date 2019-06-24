# Homework 2: Graph Search

{{travis_badge}} {{codecov_badge}}

For this homework, you will implement a graph data structure and graph
search algorithm. The purpose is for you to gain experience in error
handling and implementing data structures in Rust.

## The deliverable

The purpose of `graph` is to find paths in graphs. It reads a graph
specification from a file, and then answers routing queries read from
the standard input.

A graph specification file represents an undirected graph as an
association list of nodes, written as tokens. In particular, each line
is a list of words, where the first word names some node in the graph
and the remaining words enumerate its neighbors. Every node mentioned as
a neighbor must start a line as well, and no node may start more than
one line. And while the input format may specify an edge in only one
direction, the graph it encodes is [undirected nonetheless][symmetric
closure].

The user enters queries on stdin one at a time. A query consists of two
node names, a starting node and an ending node. The program then prints
out a path between the nodes or a message that no such path exists.

### Example

Here’s a simple example. First, let’s see what’s in the graph
description file:

```bash
$ cat graph.dat
foo bar qux
bar foo qux
baz
qux baz
```

We run the program, passing it the graph description file on the
command line. Then we type queries at the prompt and the program
responds to each:

```bash
$ cargo run graph.dat

>>> foo qux
foo bar qux

>>> foo bar
foo bar

>>> foo quux
Unknown node: quux

>>> foo baz
foo bar qux baz

>>> ^D
$
```

As you can see at the end of the transcript, the program exits cleanly
on EOF.

### Other requirements

Your program should deal with user errors gracefully. Nothing in the
graph specification nor the standard input should be able to cause a
panic. It should not be possible for a large graph to cause a stack
overflow.

Your program should be factored into (at least) a graph library and a
binary. The graph library API should be sufficient to write the graph
search efficiently as a client of the graph library.

The graph type (or types) provided by the graph library API must be
abstract data types that protect their own invariants. In particular,
this means that the graph type should be a `struct` with private fields,
forcing the client to access it via public methods. It should not be
possible for a client of the graph library to construct an invalid graph
representation that, for example, thinks vertex *v* is a successor of
vertex *u* but not vice versa, or contains edges that mention vertices
that don’t exist.

List any assumptions that you need to make where this specification is
incomplete, and be sure to test thoroughly.

## Design suggestions

Here are some suggestions to make your design cleaner, more idiomatic,
and more efficient:

  - Don’t store graph search state in the graph object itself. In some
    algorithms textbooks the graph contains the mutable state that graph
    algorithms need to traverse it. But this is wrong conceptually,
    because the DFS state isn’t part of the graph. And it’s wrong
    pragmatically, because it means that you cannot, for example, do
    multiple searches on the same graph in parallel.

  - Think carefully about which functions need to take ownership of
    parameters and which can merely borrow. For example, a function that
    takes a string in order to store it in some data structure usually
    needs to take a `String`, but a function that merely needs to look
    at a string, perhaps to use it for some kind of lookup, should
    usually take a `&str`.

  - The vertices of your graph will be labeled by strings, but that
    doesn’t mean that you should represent vertices as strings
    internally. That is, you may be tempted to define your graph type
    like this:

    ```rust
    pub struct Graph(HashMap<String, Vec<String>>);
    ```

    Or maybe even like this:

    ```rust
    pub struct Graph(HashMap<String, HashSet<String>>);
    ```

    But consider what those representations mean for graph search
    efficiency. Every time you want to know the successors of a vertex,
    which happens a lot, you have to do a hash table lookup, which
    entails hashing the string and then potentially multiple string
    comparisons. And this isn’t true just internal to the graph ADT.
    Clients that want to keep track of some property for each
    vertex—such as whether it’s been visited or how we got there—need to
    use hash tables as well.

    It’s probably better to start with a traditional graph
    representation such as an [adjacency list] or [adjacency matrix],
    which represent vertices as consecutive natural numbers. In
    particular, if a graph has `n` vertices then they are numbered from
    `0` to `n - 1`. Representing vertices as `usize`s means that a
    mapping from vertices to any type `T` can be stored as as a simple
    `Vec<T>` of length `n`, rather than as a hash table.

    Then what about vertex names? You can layer a bidirectional
    vertex–name mapping on top of the basic `usize`-based graph. For
    example, you might end up with something like this:

    ```rust
    /// An undirected graph whose vertices are consecutive `usize`s
    /// starting at `0`.
    pub struct IntGraph {
        …
    }

    /// A bijection between vertex numbers and vertex names.
    pub struct LabelMap {
        forward:  HashMap<String, usize>,
        backward: Vec<String>,
    }

    /// An undirected graph whose vertices are labeled by strings.
    pub struct LabelGraph {
        base:   IntGraph,
        labels: LabelMap,
    }
    ```

    With this approach, clients may have to be aware of the name–number
    correspondence in order to map back and forth. For example, you’d
    first write a depth-first search function for `IntGraph`s that takes
    two vertices as `usize`s and returns the path as a `Vec<usize>`. On
    top of that, you’d layer a depth-first search function for
    `LabelGraph`s that takes two vertices as `&str`s, maps them to
    `usize`, calls the `IntGraph` DFS, and then translates the resulting
    path from `Vec<usize>` to `Vec<String>`.

  - The easiest way to write a DFS is recursively, but that risks
    blowing the control stack, so you have to do it iteratively,
    maintaining your own stack in a `Vec`. This means that the control
    flow of the function is no longer sufficient to construct the path
    that you want to return, so you’ll need to keep track of more path
    information along the way.

    The easy-but-wrong solution is to store paths in your stack instead
    of single vertices, but the problem with this is that allocating and
    copying paths changes your asymptotic complexity from linear to
    quadratic, and that’s no good at all. Instead, the trick is to store
    just vertices (as `usize`s in stack), and for each vertex that you
    reach, you should record the predecessor vertex from which you reached
    it elsewhere. (This might mean that the state tracking which vertices
    we’ve seen will not be just a `Vec<bool>` indicating where we’ve been,
    but `Vec<Option<usize>>` where `None` means we haven’t been there yet,
    and `Some(u)` means we’ve been there and `u` was the predecessor.)
    Then once you reach the destination vertex, you can extract the path
    in reverse by following each vertex to its precessor until reaching the
    source vertex.

## Evaluation

Your grade will be based on:

  - correctness (how closely your program adheres to its specification),
  - style (not expecting the most idiomatic Rust at this point, but I’ll
    be looking for good factoring—don’t put everything in main),
  - testing (make sure you have good test coverage), and
  - efficiency (no need to benchmark or profile, but do choose sensible
    data structures and avoid needless cloning).


[symmetric closure]:
    https://en.wikipedia.org/wiki/Symmetric_closure

[adjacency list]:
    https://en.wikipedia.org/wiki/Adjacency_list

[adjacency matrix]:
    https://en.wikipedia.org/wiki/Adjacency_matrix
