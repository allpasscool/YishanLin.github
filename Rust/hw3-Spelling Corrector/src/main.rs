use std::fs::File;
use std::io::{self, Read, stdin,stdout,Write,};
use std::env;
use correct::trie;


fn main() {
    //read arguments
    let args: Vec<String> = env::args().collect();

    //read while file
    let whole_file = filename_to_string(&args[1]).unwrap();

    //get all distinct words
    let lines: Vec<Vec<&str>> = whole_file.lines().map(|line| {
        line.split_whitespace().collect()
    }).collect();
    let words = get_words_list(&lines);

    //prefix tree
    let mut prefix_tree =  trie::CorrectorModel::new();

    //build prefix tree
    for mut i in &words{
        //only lowercase
        // i = i.to_lowercase();
        prefix_tree.learn(&i.to_lowercase());
    }

    //read input query until eof
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
        
        //only enter newline
        if bytes.len() == 2 && (input == "\n" || input == "\r\n"){
            println!("only enter new line");
            continue;
        }

        //only lowercase
        let input: Vec<&str> = input.split_whitespace().collect();
        let input = input[0].to_lowercase().to_string();

        let result = prefix_tree.suggest(&input);
        match result {
            trie::Correction::Correct => println!("{}",input),
            trie::Correction::Incorrect => println!("{}, -",input),
            trie::Correction::Suggestion(s) => println!("{}, {}",input, s),
        }
    }
}

///read file and output the content into &str
fn filename_to_string(s: &str) -> io::Result<String> {
    let mut file = File::open(s)?;
    let mut s = String::new();
    file.read_to_string(&mut s)?;
    Ok(s)
}

///input content, output distince word list in Vec<String>
fn get_words_list(lines: &[Vec<&str>]) -> Vec<String>{
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

