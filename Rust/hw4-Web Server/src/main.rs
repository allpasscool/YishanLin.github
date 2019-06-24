extern crate chrono;
extern crate mockstream;

use std::{io, net};
use std::io::prelude::*;
use std::net::TcpListener;
use std::fs;
use std::thread;
use std::fs::metadata;
use std::sync::mpsc::channel;
use chrono::{DateTime, Utc};
use std::path::Path;
use std::time::{Duration, Instant};

struct Logstruct{
    timestamp: DateTime<Utc>,
    client_ip: String,
    uri: String,
    status_response: String,
}

pub trait Stream: io::Read + io::Write{

}

enum Response{
    // OK200,
    BadRequest400,
    ForBidden403,
    NotFound404,
    InternalServerError500,    
}

impl Stream for net::TcpStream{
}

fn main() -> io::Result<()>{
    
    let listener = TcpListener::bind("127.0.0.1:8080")?;
    // let listener = match TcpListener::bind("127.0.0.1:8080"){
    //     Ok(a)   => a,
    //     Err(e)  => 
    // };
    
    let (tx, rx) :(std::sync::mpsc::Sender<Logstruct>, std::sync::mpsc::Receiver<Logstruct>) = channel::<Logstruct>();
    thread::spawn(move || {
        logger(rx);
    });
    
    for stream in listener.incoming(){
        // let stream = stream.expect("stream error");
        let stream = match stream{
            Ok(s)   => s,
            Err(_)  => {continue},
        };
        let tx = tx.clone();

        thread::spawn(move || {
            handle_connection(stream, tx).unwrap_or_default();
        });   
    }
    Ok(())
}

//handle connection
fn handle_connection(mut stream: impl Stream, tx: std::sync::mpsc::Sender<Logstruct>) -> io::Result<()> {
    let mut buffer = [0; 512];
    // let get = b"GET";
    let mut filename: String;
    let mut get_path;
    let mut http;

    // println!("handle connection");
    
    //read data from stream
    //time out 30 seconds
    let start = Instant::now();
    loop{
        match stream.read(&mut buffer){
            Ok(_) => {break;},
            Err(e) => {
                if start.elapsed() > Duration::from_secs(30){
                    response_status(stream, Response::InternalServerError500, tx, String::new())?;
                    return Err(e);
                }
            },
        };
    }

    //input data
    let input = String::from_utf8_lossy(&buffer[..]);
    let input = input.split("\r\n").next();
    // let mut input0 : String;
    let input0: String = match input{
        Some(a) => {a.to_string()},
        None  => {return Result::Err(std::io::Error::new(std::io::ErrorKind::Other, "No path"));},
    };


    //if get request
    if buffer.starts_with(b"GET") {
        
        // let file_loc = input0.clone();
        //GET /path HTTP
        let mut file_loc = input0.split(' ').skip(1);

        match file_loc.next(){
            Some(a) => get_path = a.to_string(),
            None => {return Result::Err(std::io::Error::new(std::io::ErrorKind::Other, "No path"));}
        }

        match file_loc.next(){
            Some(a) => http = a.to_string(),
            None => {return Result::Err(std::io::Error::new(std::io::ErrorKind::Other, "No http"));}
        }

        if get_path.contains(".."){
            // println!("going back, not allowed");
            return Result::Err(std::io::Error::new(std::io::ErrorKind::Other, "going back, not allowed"));
        }

        //error 400, not a properly formatted GET command
        if !http.starts_with("HTTP") || !get_path.starts_with('/'){
            // println!("not start with /");
            response_status(stream, Response::BadRequest400, tx, input0)?;
            return Result::Err(std::io::Error::new(std::io::ErrorKind::Other, "No path"))
        }

        //get filelocation
        get_path.remove(0);
        // let tmp = get_path.clone();
        let file_loc = get_path.clone(); 
        let md = metadata(&file_loc);
        filename = file_loc;

        //check if file exist
        let path_filename = format!("./{}", filename);
        let path = Path::new(&path_filename);
        if !path.exists(){
            // println!("path doesn't find");
            response_status(stream, Response::NotFound404, tx, input0)?;
            return Result::Err(std::io::Error::new(std::io::ErrorKind::Other, "path doesn't find"))
        }

        //if have permission to read file and check if it is directory
        let mut with_permission = true;
        let mut is_dir = false;
        match md{
            Ok(a) => {
                is_dir = a.is_dir();
                },
            Err(_) => {
                with_permission = false;
            },
        };
        
        //permission?
        //403 forbidden
        if !with_permission{
            response_status(stream, Response::ForBidden403, tx, input0)?;
            return Result::Err(std::io::Error::new(std::io::ErrorKind::Other, "403 forbidden"));
        }

        //is directory?
        let mut find_file = false;
        if is_dir{
            
            let try_file = vec!["index.html", "index.shtml", "index.txt"];

            for i in try_file{
                filename = format!("{}/{}", get_path, i);
                let md = metadata(&filename);
                // println!("filename: {}", filename);

                if let Ok(a) = md {
                    if a.is_file(){
                        find_file = true;
                        break;
                    }
                }
            }

            //doesn't find a proper file in the directory
            if !find_file{
                // println!("doesn't find a proper file");
                response_status(stream, Response::NotFound404, tx, input0)?;
                return Result::Err(std::io::Error::new(std::io::ErrorKind::Other, "doesn't find a proper file"));
            }
        }
        
        let mut contents: Vec<u8>;
        
        //read data from file
        // match fs::read(&filename){
        //     Ok(a) => contents = a,
        //     Err(_) => {
        //                 println!("read data from file error");
        //                 response_status(stream, Response::NotFound404, tx, input0);
        //                 return;
        //                 },
        // }
        match fs::read(&filename){
            Ok(a) => contents = a,
            Err(_) => {
                        // println!("read data from file error");
                        response_status(stream, Response::NotFound404, tx, input0)?;
                        return Result::Err(std::io::Error::new(std::io::ErrorKind::Other, "read data from file error"));
                        },
        }

        //200 OK

        let content_type = if filename.ends_with("html"){
            "html".to_string()
        }   else{
            "plain".to_string()
        };

        let response = format!("HTTP/1.0 200 OK\r\nServer: yishanlin\r\nContent-type: text/{}\r\nContent-length: {}\r\n\r\n", content_type, contents.len());

        io::copy(&mut response.as_bytes(), &mut stream)?;//.expect("write error 110");
        io::copy(&mut contents.as_slice(), &mut stream)?;//.expect("write error 110");
        
        stream.flush()?;//.expect("stream flush error");
        
        tx.send(Logstruct{
            timestamp: Utc::now(),
            uri: input0.replace("\n", ""),
            client_ip: String::new(),
            status_response: "HTTP/1.0 200 OK".to_string(),
        }).expect("tx send log error");

    } else{ // if not get request
        //400 bad request
        response_status(stream, Response::BadRequest400, tx, input0)?;
    }
    Ok(())
}

//logger
fn logger(rx: std::sync::mpsc::Receiver<Logstruct>) {
    let mut file = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(true)
        .open("log.txt")
        .expect("couldn't open log.txt");
    
    if let Err(e) = writeln!(file, "start a new log, UTC time:{}", Utc::now().format("%a %b %e %T %Y")){
            eprintln!("Couldn't write to file: {}", e);
        }

    
    loop{
        let log: Logstruct = rx.recv().expect("log receive error");
        let log = format!("UTC timestamp:{}\nURI:{}\nClient IP:{}\nstatus of response:{}\n",
                         log.timestamp.format("%a %b %e %T %Y"), log.uri, log.client_ip, log.status_response);
        if let Err(e) = writeln!(file, "{}", log){
            eprintln!("Couldn't write to file: {}", e);
        }
    }
}

//handle response status and tell logger
fn response_status(mut stream: impl Stream, rs_status: Response, tx: std::sync::mpsc::Sender<Logstruct>, input0: String) -> io::Result<()>{
    match rs_status{
        // Response::OK200 =>                  {},
        Response::BadRequest400 =>          {
                                                let (status_line, filename) = ("HTTP/1.0 400 not a properly formatted GET command\r\n\r\n", "400.html");
                                                let contents = fs::read_to_string(&filename)?;//.expect("read content error");
                                                let response = format!("{}{}", status_line, contents);

                                                io::copy(&mut response.as_bytes(), &mut stream)?;//.expect("write error 110");

                                                stream.flush()?;//.expect("stream flush error");

                                                tx.send(Logstruct{
                                                    timestamp: Utc::now(),
                                                    uri: input0.replace("\n", ""),
                                                    client_ip: String::new(),
                                                    status_response: "HTTP/1.0 400 not a properly formatted GET command".to_string(),
                                                }).expect("tx send log error");
        },
        Response::ForBidden403 =>           {
                                                let status_line = "HTTP/1.0 403 Forbidden\r\n\r\n".to_string();
                                                let filename = "403.html".to_string();
                                                
                                                let contents = fs::read_to_string(filename)?;//.expect("read content error");
                                                let response = format!("{}{}", status_line, contents);

                                                io::copy(&mut response.as_bytes(), &mut stream)?;//.expect("write error 110");

                                                stream.flush()?;//.expect("stream flush error");
                                                tx.send(Logstruct{
                                                    timestamp: Utc::now(),
                                                    uri: input0.replace("\n", ""),
                                                    client_ip: String::new(),
                                                    status_response: 
                                                    "HTTP/1.0 403 Forbidden, which rejects a command because it specifies a file that is off-limits".to_string(),
                                                }).expect("tx send log error");
        },
        Response::NotFound404 =>            {
                                                let (status_line, filename) = ("HTTP/1.0 404 Not Found\r\n\r\n", "404.html");
                                                let contents = fs::read_to_string(&filename)?;//.expect("read content error");
                                                let response = format!("{}{}", status_line, contents);
                                                
                                                io::copy(&mut response.as_bytes(), &mut stream)?;//.expect("write error 110");

                                                stream.flush()?;//.expect("stream flush error");
            
                                                tx.send(Logstruct{
                                                    timestamp: Utc::now(),
                                                    uri: input0.replace("\n", ""),
                                                    client_ip: String::new(),
                                                    status_response: "HTTP/1.0 404 Not Found".to_string(),
                                                }).expect("tx send log error");
        },
        Response::InternalServerError500 => {
                                                let (status_line, filename) = ("HTTP/1.0 500 Internal Server Error\r\n\r\n", "500.html");
                                                let contents = fs::read_to_string(&filename)?;//.expect("read content error");
                                                let response = format!("{}{}", status_line, contents);

                                                io::copy(&mut response.as_bytes(), &mut stream)?;//.expect("write error 110");

                                                stream.flush()?;//.expect("flush error");
            
                                                tx.send(Logstruct{
                                                    timestamp: Utc::now(),
                                                    uri: String::new(),
                                                    client_ip: String::new(),
                                                    status_response: "HTTP/1.0 500 Internal Server Error, read stream".to_string(),
                                                }).expect("tx send log error");
        },
    }
    Ok(())
}

#[cfg(test)]
mod tests{
    use std::io;
    use std::net::TcpStream;
    use mockstream::*;
    use std::sync::mpsc::channel;
    use super::*;
    use std::str;

    enum NetStream {
        Mocked(SharedMockStream),
        Tcp(TcpStream)
    }

    impl io::Read for NetStream {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            match *self {
                NetStream::Mocked(ref mut s) => s.read(buf),
                NetStream::Tcp(ref mut s) => s.read(buf),
            }
        }
    }

    impl io::Write for NetStream {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            match *self {
                NetStream::Mocked(ref mut s) => s.write(buf),
                NetStream::Tcp(ref mut s) => s.write(buf),
            }
        }

        fn flush(&mut self) -> io::Result<()> {
            match *self {
                NetStream::Mocked(ref mut s) => s.flush(),
                NetStream::Tcp(ref mut s) => s.flush(),
            }
        }
    }

    impl Stream for SharedMockStream{

    }

    // test 1
    // GET /data/testTxt HTTP/1.0
    // HTTP/1.0 200 OK
    // Server: yishanlin
    // Content-type: text/plain
    // Content-length: 18

    #[test]
    // test txt index.txt
    fn test1(){
        let mut s = SharedMockStream::new();

        let (tx, _rx) :(std::sync::mpsc::Sender<Logstruct>, std::sync::mpsc::Receiver<Logstruct>) = channel::<Logstruct>();
        // thread::spawn(move || {
        //             logger(rx);
        //         });


        s.push_bytes_to_read(b"GET /data/testTxt HTTP/1.0");     

        handle_connection(s.clone(), tx).unwrap();
        // rx.recv().expect("log receive error");

        let re = s.pop_bytes_written();

        let ok200 = "HTTP/1.0 200 OK\r\nServer: yishanlin\r\nContent-type: text/plain\r\nContent-length: 18\r\n\r\ntest txt index.txt".to_string();

        assert_eq!(str::from_utf8(&re).unwrap(), ok200);
        // assert_eq!(re, ok200.as_bytes());

        assert!(re.starts_with(b"HTTP/1.0 200 OK"));
    }


    // test 2
    // GET /data/testShtml HTTP/1.0
    // HTTP/1.0 200 OK
    // Server: yishanlin
    // Content-type: text/html
    // Content-length: 191

    // <!DOCTYPE html>
    // <html lang="en">
    // <head>
    //     <meta charset="utf-8">
    //     <title>Hello!</title>
    // </head>
    // <body>
    //     <h1>Hello!</h1>
    //     <p>Test shtml index</p>
    // </body>
    // </html>
    #[test]
    fn test2(){
        let mut s = SharedMockStream::new();

        let (tx, _rx) :(std::sync::mpsc::Sender<Logstruct>, std::sync::mpsc::Receiver<Logstruct>) = channel::<Logstruct>();
                // thread::spawn(move || {
                //     logger(rx);
                // });


        s.push_bytes_to_read(b"GET /data/testShtml HTTP/1.0");     

        handle_connection(s.clone(), tx).unwrap();

        let re = s.pop_bytes_written();

        let content = fs::read("/data/testShtml/index.shtml");

        let content = match content{
            Ok(a) => a,
            Err(_) => {return;},
        };

        // let ok200 = "HTTP/1.0 200 OK\r\nServer: yishanlin\r\nContent-type: text/html\r\nContent-length: {}\r\n\r\n<!DOCTYPE html>\r\n<html lang=\"en\">\r\n  <head>\r\n    <meta charset=\"utf-8\">\r\n    <title>Hello!</title>\r\n  </head>\r\n  <body>\r\n    <h1>Hello!</h1>\r\n    <p>Test shtml index</p>\r\n  </body>\r\n</html>\r\n".to_string();
        let ok200 = format!("HTTP/1.0 200 OK\r\nServer: yishanlin\r\nContent-type: text/html\r\nContent-length: {}\r\n\r\n{}", content.len(), str::from_utf8(&content).unwrap());

        assert_eq!(str::from_utf8(&re).unwrap(), ok200);
        assert_eq!(re, ok200.as_bytes());

        assert!(re.starts_with(b"HTTP/1.0 200 OK"));
    }

    //test 3
    // GET /data/testHtml HTTP/1.0
    // HTTP/1.0 200 OK
    // Server: yishanlin
    // Content-type: text/html
    // Content-length: 185

    // <!DOCTYPE html>
    // <html lang="en">
    // <head>
    //     <meta charset="utf-8">
    //     <title>Hello!</title>
    // </head>
    // <body>
    //     <h1>Hello!</h1>
    //     <p>Test index</p>
    // </body>
    // </html>
    #[test]
    fn test3(){
        let mut s = SharedMockStream::new();

        let (tx, _rx) :(std::sync::mpsc::Sender<Logstruct>, std::sync::mpsc::Receiver<Logstruct>) = channel::<Logstruct>();
                // thread::spawn(move || {
                //     logger(rx);
                // });


        s.push_bytes_to_read(b"GET /data/testHtml HTTP/1.0");     

        handle_connection(s.clone(), tx).unwrap();

        let re = s.pop_bytes_written();

        let content = fs::read("/data/testHtml/index.html");

        let content = match content{
            Ok(a) => a,
            Err(_) => {return;},
        };

        // let ok200 = "HTTP/1.0 200 OK\r\nServer: yishanlin\r\nContent-type: text/html\r\nContent-length: 185\r\n\r\n<!DOCTYPE html>\r\n<html lang=\"en\">\r\n  <head>\r\n    <meta charset=\"utf-8\">\r\n    <title>Hello!</title>\r\n  </head>\r\n  <body>\r\n    <h1>Hello!</h1>\r\n    <p>Test index</p>\r\n  </body>\r\n</html>\r\n".to_string();
        let ok200 = format!("HTTP/1.0 200 OK\r\nServer: yishanlin\r\nContent-type: text/html\r\nContent-length: {}\r\n\r\n{}", content.len(), str::from_utf8(&content).unwrap());
        assert_eq!(str::from_utf8(&re).unwrap(), ok200);
        assert_eq!(re, ok200.as_bytes());

        assert!(re.starts_with(b"HTTP/1.0 200 OK"));
    }

    //test 4
    // GET /hello.html HTTP
    // HTTP/1.0 200 OK
    // Server: yishanlin
    // Content-type: text/html
    // Content-length: 187

    // <!DOCTYPE html>
    // <html lang="en">
    // <head>
    //     <meta charset="utf-8">
    //     <title>Hello!</title>
    // </head>
    // <body>
    //     <h1>Hello!</h1>
    //     <p>Hi from Rust</p>
    // </body>
    // </html>
    #[test]
    fn test4(){
        let mut s = SharedMockStream::new();

        let (tx, _rx) :(std::sync::mpsc::Sender<Logstruct>, std::sync::mpsc::Receiver<Logstruct>) = channel::<Logstruct>();
                // thread::spawn(move || {
                //     logger(rx);
                // });


        s.push_bytes_to_read(b"GET /hello.html HTTP");     

        handle_connection(s.clone(), tx).unwrap();

        let re = s.pop_bytes_written();

        let content = fs::read("hello.html");

        let content = match content{
            Ok(a) => a,
            Err(_) => {return;},
        };

        // let ok200 = "HTTP/1.0 200 OK\r\nServer: yishanlin\r\nContent-type: text/html\r\nContent-length: 187\r\n\r\n<!DOCTYPE html>\r\n<html lang=\"en\">\r\n  <head>\r\n    <meta charset=\"utf-8\">\r\n    <title>Hello!</title>\r\n  </head>\r\n  <body>\r\n    <h1>Hello!</h1>\r\n    <p>Hi from Rust</p>\r\n  </body>\r\n</html>\r\n".to_string();
        let ok200 = format!("HTTP/1.0 200 OK\r\nServer: yishanlin\r\nContent-type: text/html\r\nContent-length: {}\r\n\r\n{}", content.len(), str::from_utf8(&content).unwrap());
        assert_eq!(str::from_utf8(&re).unwrap(), ok200);
        // assert_eq!(re, ok200.as_bytes());
        assert!(re.starts_with(b"HTTP/1.0 200 OK"));
    }

    //test 5
    // GET /hello. HT TP
    // HTTP/1.0 400 not a properly formatted GET command

    // <!DOCTYPE html>
    // <html lang="en">
    // <head>
    //     <meta charset="utf-8">
    //     <title>Hello!</title>
    // </head>
    // <body>
    //     <h1>Oops!</h1>
    //     <p>Sorry, 400 not a properly formatted GET command</p>
    // </body>
    // </html>
    #[test]
    #[should_panic]
    fn test5(){
        let mut s = SharedMockStream::new();

        let (tx, _rx) :(std::sync::mpsc::Sender<Logstruct>, std::sync::mpsc::Receiver<Logstruct>) = channel::<Logstruct>();
                // thread::spawn(move || {
                //     logger(rx);
                // });


        s.push_bytes_to_read(b"GET /hello. HT TP");     

        handle_connection(s.clone(), tx).unwrap();

        let re = s.pop_bytes_written();

        let content = fs::read("400.html");

        let content = match content{
            Ok(a) => a,
            Err(_) => {return;},
        };

        // let not400 = "HTTP/1.0 400 not a properly formatted GET command\r\n\r\n<!DOCTYPE html>\r\n<html lang=\"en\">\r\n  <head>\r\n    <meta charset=\"utf-8\">\r\n    <title>Hello!</title>\r\n  </head>\r\n  <body>\r\n    <h1>Oops!</h1>\r\n    <p>Sorry, 400 not a properly formatted GET command</p>\r\n  </body>\r\n</html>\r\n".to_string();
        let not400 = format!("HTTP/1.0 400 not a properly formatted GET command\r\n\r\n{}", str::from_utf8(&content).unwrap());

        assert_eq!(str::from_utf8(&re).unwrap(), not400);
        // assert_eq!(re, not400.as_bytes());
        assert!(re.starts_with(b"HTTP/1.0 400 not"));
    }

    //test 6
    //GET \hello.html HTTP
    // HTTP/1.0 400 not a properly formatted GET command

    // <!DOCTYPE html>
    // <html lang="en">
    // <head>
    //     <meta charset="utf-8">
    //     <title>Hello!</title>
    // </head>
    // <body>
    //     <h1>Oops!</h1>
    //     <p>Sorry, 400 not a properly formatted GET command</p>
    // </body>
    // </html>
    #[test]
    #[should_panic]
    fn test6(){
        let mut s = SharedMockStream::new();

        let (tx, _rx) :(std::sync::mpsc::Sender<Logstruct>, std::sync::mpsc::Receiver<Logstruct>) = channel::<Logstruct>();
                // thread::spawn(move || {
                //     logger(rx);
                // });


        s.push_bytes_to_read(b"GET \\hello.html HTTP");     

        handle_connection(s.clone(), tx).unwrap();

        let re = s.pop_bytes_written();

        let content = fs::read("400.html");

        let content = match content{
            Ok(a) => a,
            Err(_) => {return;},
        };

        // let not400 = "HTTP/1.0 400 not a properly formatted GET command\r\n\r\n<!DOCTYPE html>\r\n<html lang=\"en\">\r\n  <head>\r\n    <meta charset=\"utf-8\">\r\n    <title>Hello!</title>\r\n  </head>\r\n  <body>\r\n    <h1>Oops!</h1>\r\n    <p>Sorry, 400 not a properly formatted GET command</p>\r\n  </body>\r\n</html>\r\n".to_string();
        let not400 = format!("HTTP/1.0 400 not a properly formatted GET command\r\n\r\n{}", str::from_utf8(&content).unwrap());

        assert_eq!(str::from_utf8(&re).unwrap(), not400);
        assert_eq!(re, not400.as_bytes());
        assert!(re.starts_with(b"HTTP/1.0 400 not"));
    }

    //test 7
    // GET /data HTTP
    // HTTP/1.0 404 Not Found

    // <!DOCTYPE html>
    // <html lang="en">
    // <head>
    //     <meta charset="utf-8">
    //     <title>Hello!</title>
    // </head>
    // <body>
    //     <h1>Oops!</h1>
    //     <p>Sorry, I don't know what you're asking for.</p>
    // </body>
    // </html>
    #[test]
    #[should_panic]
    fn test7(){
        let mut s = SharedMockStream::new();

        let (tx, _rx) :(std::sync::mpsc::Sender<Logstruct>, std::sync::mpsc::Receiver<Logstruct>) = channel::<Logstruct>();
                // thread::spawn(move || {
                //     logger(rx);
                // });


        s.push_bytes_to_read(b"GET /data HTTP");     

        handle_connection(s.clone(), tx).unwrap();

        let re = s.pop_bytes_written();

        let content = fs::read("404.html");

        let content = match content{
            Ok(a) => a,
            Err(_) => {return;},
        };

        // let not404 = "HTTP/1.0 404 Not Found\r\n\r\n<!DOCTYPE html>\r\n<html lang=\"en\">\r\n  <head>\r\n    <meta charset=\"utf-8\">\r\n    <title>Hello!</title>\r\n  </head>\r\n  <body>\r\n    <h1>Oops!</h1>\r\n    <p>Sorry, I don\'t know what you\'re asking for.</p>\r\n  </body>\r\n</html>\r\n".to_string();
        let not404 = format!("HTTP/1.0 404 Not Found\r\n\r\n{}", str::from_utf8(&content).unwrap());
        
        assert_eq!(str::from_utf8(&re).unwrap(), not404);
        assert_eq!(re, not404.as_bytes());
        assert!(re.starts_with(b"HTTP/1.0 404 Not"));
    }

    //GET /Untitled.png HTTP
    #[test]
    fn test8(){
        let mut s = SharedMockStream::new();

        let (tx, _rx) :(std::sync::mpsc::Sender<Logstruct>, std::sync::mpsc::Receiver<Logstruct>) = channel::<Logstruct>();
                // thread::spawn(move || {
                //     logger(rx);
                // });


        s.push_bytes_to_read(b"GET /Untitled.png HTTP");     

        handle_connection(s.clone(), tx).unwrap();

        let re = s.pop_bytes_written();

        let content = fs::read("Untitled.png");

        let content = match content{
            Ok(a) => a,
            Err(_) => {return;},
        };

        // let ok200 = "HTTP/1.0 404 Not Found\r\n\r\n<!DOCTYPE html>\r\n<html lang=\"en\">\r\n  <head>\r\n    <meta charset=\"utf-8\">\r\n    <title>Hello!</title>\r\n  </head>\r\n  <body>\r\n    <h1>Oops!</h1>\r\n    <p>Sorry, I don\'t know what you\'re asking for.</p>\r\n  </body>\r\n</html>\r\n".to_string();
        let ok200 = format!("HTTP/1.0 200 OK\r\nServer: yishanlin\r\nContent-type: text/plain\r\nContent-length: {}\r\n\r\n", content.len());
        let ok200 = ok200.as_bytes();
        let ok200 = [ok200, &content].concat();

        // let te = vec![112, 108, 97, 105, 110];
        // assert_eq!("0".to_string(), str::from_utf8(&te).unwrap());

        assert_eq!(re, ok200);
        assert!(re.starts_with(b"HTTP/1.0 200 OK"));
    }
}