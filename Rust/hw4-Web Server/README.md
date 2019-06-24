# Homework 4: Web Server

[![Travis CI status](https://travis-ci.com/nu-rust-course/hw4-yishan-and.svg?token=Ase2AYhgcqMDkdsyq9m2&branch=develop)](https://travis-ci.com/nu-rust-course/hw4-yishan-and) [![Codecov.io status](https://codecov.io/gh/nu-rust-course/hw4-yishan-and/branch/develop/graph/badge.svg?token=mqUpqjHt7J)](https://codecov.io/gh/nu-rust-course/hw4-yishan-and/branch/develop)

For this homework, you will implement a rudimentary web server. The purpose is for you to begin taking advantage of concurrency in Rust.

## The deliverable

The purpose of `webserver` is to respond to the single command of HTTP 0.9, the GET method, which has the following shape:

```
GET /path/to/file HTTP
```

That is, it is the literal word `GET`, followed by a blank space, followed by a Unix-style absolute path to a file, followed by another blank space and the literal token `HTTP`. The following line is a blank line. For forward compatibility, you should also accept all newer HTTP versions, which will end their request with a token that includes the version, e.g., HTTP/1.1. And you should skip over any header lines following the request but preceding the blank line.

In return to a valid GET request, the web server spawns a thread that retrieves the request, records it to a log file, and generates a response. For this assignment, the following five response statuses are appropriate:

- `200 OK`, which starts a reply that serves the specified file;
- `400 Bad Request`, which indicates that the command is not a properly formatted GET command;
- `403 Forbidden`, which rejects a command because it specifies a file that is off-limits;
- `404 Not Found`, which informs the client that the specified file does not exist; and
- `500 Internal Server Error`, which is issued when the server could not complete the request due to an unspecified error.

Each response is preceded by `HTTP/1.0` and blank space.

The complete header of a `200 OK` response is formatted as follows:

```
HTTP/1.0 200 OK
Server: {server-name}
Content-type: text/{plain-or-html}
Content-length: {number-of-bytes-sent}

```

including a blank line afterward. To keep things simple, the `{plain-or-html}` property is either `html` for files whose suffix is `.html` or `plain` for all others. The remainder of a `200 OK` message is the content of the specified file.

A path specification `{path-to-file}` must start with / and is interpreted after concatenating it with the server’s root path:

  - If the resulting path points to a file, the file is served with a `200 OK` response unless its permissions do not allow so.
  - If the resulting path points to a directory, it is interpreted as pointing to one of these files: `index.html`, `index.shtml`, and `index.txt`. The first file found is served assuming it is accessible. Otherwise the path triggers a 404.
  - Otherwise the server responds with an error message.

Your web server should listen on localhost (127.0.0.1), port 8080. To explore its workings, point your web browser to http:// localhost:8080/src/main.rs (assuming you are running it out of the Cargo directory) or use netcat:

```
$ nc 127.0.0.1 8080
GET /src/main.rs HTTP

HTTP/1.0 200 OK
Server: jat489/0.1
Content-type: text/plain
Content-length: 2034

// This module implements a rudimentary web server.

use std::io;
use std::sync::Mutex;
```

Keep it simple — do not try to handle additional HTTP methods.

### Tips

- Your web server should be able to serve any accessible file, regardless of size or encoding.
- A web server talks to untrusted clients over a network. These clients should not be able to crash the server, prevent it from serving other clients, or access files outside the server root.
- You will want to open the log only once per server run. To share it between threads, you can use a `Mutex<File>` or have a dedicated logging thread that you send log messages to over a channel.

## Hard mode

Instead of using system threads to serve multiple clients concurrently, use asynchronous I/O via [futures](https://aturon.github.io/blog/2016/08/11/futures/) and the [tokio](https://docs.rs/tokio/0.1.19/tokio/) async runtime.

Jonathan got it working using these crate versions:

```toml
futures = "0.1.20"
tokio = "0.1.18"
```

Newer `"0.1.*"` releases also might work, but there are higher-sounding versions of `futures` that aren’t quite ready yet.

## Evaluation

Your grade will be based on:

- correctness (how closely your program adheres to its specification),
- style (idiomatic Rust and good factoring),
- testing (to the extent that you reasonably can), and
- efficiency (no need to benchmark or profile, but do choose sensible data structures and avoid needless copying).
