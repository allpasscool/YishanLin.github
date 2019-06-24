/// # Word frequency calculator
/// ### Some notes on this implementation:
/// * Any words separated by non-alphanumerics other than hyphen(-) or apostrophe(') are split
/// * This works more like a tokenizer– no dictionary checking or the like is done
/// *  


use std::collections::HashMap;
extern crate regex;
use regex::Regex;
use std::io::{self, Read};

/// Splits text into component words
fn split_words(text: &str) -> Vec<String> {
    // Makes sure everything is lowercase for comparison 
    //let text = text.to_lowercase();
    
    // Builds regex pattern that matches any words
    // Original regex was [\w]+ but needed to be reworked after implementing compound word tests
    let re = Regex::new(r"(?:\w|['-]\w)+").unwrap();

    // Finds words using pattern  
    re.find_iter(&text.to_lowercase())
      .map(|s| s.as_str().to_owned())
      .collect()
}

/// Counts instances of each word
fn calculate_frequency(words: Vec<String>) -> HashMap<String, usize> {
    let mut freq = HashMap::new();
    
    // Increments or starts word counters
    for w in words {
        freq.entry(w)
            .and_modify(|c| *c += 1)
            .or_insert(1);
    }

    return freq;
}

/// Sorts word frequencies in descending order
fn sort_by_freq(freq: HashMap<String, usize>) -> Vec<(String, usize)> {
    let mut sorted: Vec<_> = freq.into_iter().collect();
    
    // Sorts on frequency
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    
    // Working with ownership. Not sure if there's a better way to do this. .to_owner() had to be
    // called twice if I used it on the usize which felt silly. 
    sorted.into_iter().map(|(s, n)| (s.to_string(), n)).collect()
}

fn main() {
    let mut text = String::new();
    
    // Reads from stdin to a string
    io::stdin().read_to_string(&mut text).expect("Read std input fail!");
    
    let words = split_words(&text);
    
    let freq = calculate_frequency(words);
    
    let sorted = sort_by_freq(freq);
    
    // Prints count and frequencies in descending order
    for (word, count) in &sorted {
        println!("{}: {}", word, count);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // This macro helps making writing the testing more concise
    // https://stackoverflow.com/questions/38183551/concisely-initializing-a-vector-of-strings
    macro_rules! vec_of_strings {
        ($($x:expr),*) => (vec![$($x.to_string()),*]);
    }
    
    //#[test]
    //fn test_read_text(){
    //    let t = &String::from("test.txt");
    //    assert_eq!(read_text(t), "hello world\ngoodbye world");
    //}
    
    #[test]
    fn lower_case(){
        assert_eq!(split_words("AbrACAdAbrA"), ["abracadabra"]);
    }

    #[test]
    fn split_standard_words(){
        assert_eq!(split_words("one sentence. a second Sentence"), 
        ["one", "sentence", "a", "second", "sentence"]);
    }
    
    #[test]
    fn split_punctuated_words(){
        assert_eq!(
            split_words("merry-go-round, forget-me-not, hello.goodbye, they're"), 
        ["merry-go-round", "forget-me-not", "hello", "goodbye", "they're"]);
    }

    #[test]
    fn frequency_test(){
        let f = calculate_frequency(
            vec_of_strings!("one", "two", "two", "three", "three", "three"));
        assert_eq!(f["one"], 1);
        assert_eq!(f["two"], 2);
        assert_eq!(f["three"], 3);
    }

    #[test]
    fn sort_test(){
        let mut d :HashMap<String, usize> = HashMap::new();
        d.insert("three".to_string(), 3);
        d.insert("one".to_string(), 1);
        d.insert("two".to_string(), 2);
        d.insert("four".to_string(), 4);
        
        let sorted = sort_by_freq(d);
        let res = vec![("four".to_string(), 4),
         ("three".to_string(), 3), ("two".to_string(), 2), ("one".to_string(), 1)];
        for (i, v)  in res.iter().enumerate(){
            assert_eq!(v.0, sorted[i].0);
            assert_eq!(v.1, sorted[i].1);
        }
    }
}
