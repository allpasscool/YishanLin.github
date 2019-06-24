///The purpose of graph is to find paths in graphs. 
/// It reads a graph specification from a file, 
/// and then answers routing queries read from the standard input.
/// 
/// Assumption:
///1. Every word is unique in the graph. Otherwise, the graph might be confusing.
///For example: a b c 
///             e a              
///2. no special charatcer, including {' \ ) ( [ ] { } . # @ $ % ^ ! & * - = + _ : ,)}
///
///
///
use std::fs::File;
use std::io::{self, Read, stdin,stdout,Write};
use std::env;

extern crate hw2;

///read file and output the content into &str
fn filename_to_string(s: &str) -> io::Result<String> {
    let mut file = File::open(s)?;
    let mut s = String::new();
    file.read_to_string(&mut s)?;
    Ok(s)
}

///input content, output distince word list in Vec<String>
fn get_words_list(lines: &Vec<Vec<&str>>) -> Vec<String>{
    let mut words: Vec<String> = Vec::new();

    for i in lines{
        for j in i{
            let mut new_word = true;
            //check if it is in words
            for k in words.iter(){
                //not new word
                if k == j{
                    new_word = false;
                    break;
                }
            }
            if new_word{
                words.push(j.to_string());
            }
        }
    }

    words
}

fn example_use() {
    let args: Vec<String> = env::args().collect();

    //read file
    let whole_file = filename_to_string(&args[1]).unwrap();
    //println!("while_file: {:?}", whole_file);

    //get all distinct words
    let lines: Vec<Vec<&str>> = whole_file.lines().map(|line| {
        line.split_whitespace().collect()
    }).collect();
    let words = get_words_list(&lines);

    //println!("words: {:?}", words);

    //Graph
    let mut graph = hw2::Graph::default();

    //pair words with integet
    graph.build_map(&words);

    //build adjacency matrix
    //initialize adj matrix
    graph.init_adj_matrix(&words);
    
    // input adj matrix
    graph.build_adj_matrix(&lines);
    //println!("graph after input adj_matrix: {:?}", graph.adjacency_list);

    //read input query and output answer, until eof
    loop{
        let mut input = String::new();
        print!("Please enter query:");
        let _= stdout().flush();
        stdin().read_line(&mut input).expect("Did not enter anything");

        let bytes = input.bytes();
        //reach eof
        if bytes.len() == 0{
            println!("\nreach eof");
            break;
        }

        //println!("You typed: {}", input);

        //input query
        let input: Vec<&str> = input.split_whitespace().collect();
        //println!("input:{:?}", input);

        //DFS
        graph.dfs(&input[0].to_string(), &input[1].to_string(), &words);
    }
}


fn main() {
    example_use();
}
