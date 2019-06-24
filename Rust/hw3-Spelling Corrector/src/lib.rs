//use spell_bench;
//use BufRead;

pub mod trie {


    fn c_to_i(c: char) -> usize {
        if c <= 'z' && c >= 'a'{
            c as usize - 'a' as usize
        } else{
            panic!("received unexpected character {}", c)
        }
    }

    // pub enum Correction<'a> {
    //     Correct,
    //     Incorrect,
    //     Suggestion(&'a str)
    // }
    pub enum Correction {
        Correct,
        Incorrect,
        Suggestion(String)
    }

    // pub struct CorrectorModel {
    //     root: Option<Box<Node>>,
    // }
    pub struct CorrectorModel<'a> {
        root: Option<Box<Node<'a>>>,
    }

    // struct Node {
    //     key: String,
    //     value: usize,
    //     children: [Option<Box<Node>>; 26],
    // }
    struct Node<'a> {
        key: &'a str,
        value: usize,
        children: [Option<Box<Node<'a>>>; 26],
    }

/*
    impl<'a> Correction<'a> {
        pub fn print(self) {
            match self {
                Correction::Correct => println!("correct"),
                Correction::Incorrect => println!("incorrect"),
                Correction::Suggestion(s) => println!("suggestion: {}", s),
            }
        }
    }
*/
    // impl CorrectorModel {
    //     pub fn new() -> Self {
    //         CorrectorModel {
    //             root: Node::new(String::new())
    //         }
    //     }

    //     pub fn learn(&mut self, word: String) {
    //         if self.root.is_some() {
    //             let mut node = self.root.as_mut().unwrap();
    //             for (i, c) in word.chars().enumerate() {
    //                 if node.children[c_to_i(c)].is_some() {
    //                     node = node.children[c_to_i(c)].as_mut().unwrap();
    //                 }
    //                 else {
    //                     node.children[c_to_i(c)] = Node::new(word[0..=i].to_string());
    //                     node = node.children[c_to_i(c)].as_mut().unwrap();
    //                 }
    //             }
    //             node.increment_value();
    //         }
    //     }

    //     pub fn suggest(&self, candidate: &str) -> Correction {
    //                        // should helper exist in tree or node
    //         let results = self.root.as_ref().unwrap().suggest_helper(candidate, 0usize); 
    //         let mut best_s = None;
    //         let mut best_count = 0;
    //         for (i,j) in results {
    //             if let Correction::Correct = i {
    //                 return Correction::Correct
    //             }
    //             else if let Correction::Suggestion(s) = i {
    //                 if j > best_count {
    //                     best_s = Some(s);
    //                     best_count = j;
    //                 }
    //             }
    //         }
    //         if best_s.is_none() {
    //             Correction::Incorrect
    //         }
    //         else {
    //             Correction::Suggestion(best_s.unwrap())
    //         }
    //     }


    // }

    impl<'a> CorrectorModel<'a> {
        pub fn new() -> Self {
            CorrectorModel {
                root: Node::new("")
            }
        }

        pub fn learn(&mut self, word: &'a str) {
            if self.root.is_some() {
                let mut node = self.root.as_mut().unwrap();
                for (i, c) in word.chars().enumerate() {
                    if node.children[c_to_i(c)].is_some() {
                        node = node.children[c_to_i(c)].as_mut().unwrap();
                    }
                    else {
                        node.children[c_to_i(c)] = Node::new(&word[0..(i+1)]);
                        node = node.children[c_to_i(c)].as_mut().unwrap();
                    }
                }
                node.increment_value();
            }
        }

        pub fn suggest(&self, candidate: &str) -> Correction {
                            // should helper exist in tree or node
            let results = self.root.as_ref().unwrap().suggest_helper(candidate, 0usize); 
            let mut best_s = None;
            let mut best_count = 0;
            for (i,j) in results {
                if let Correction::Correct = i {
                    return Correction::Correct
                }
                else if let Correction::Suggestion(s) = i {
                    if j > best_count {
                        best_s = Some(s);
                        best_count = j;
                    }
                }
            }
            if best_s.is_none() {
                Correction::Incorrect
            }
            else {
                Correction::Suggestion(best_s.unwrap())
            }
        }


    }

/*
    use std::io::BufRead;
//    use crate::{DefaultTokenizer, Corrector, Correction};


    impl<'a> spell_bench::Corrector for CorrectorModel<'a> {
        fn from_corpus<R: BufRead>(corpus: R) -> Self {
            let mut model = CorrectorModel::new();
            for line in corpus.lines() {
                for token in line.unwrap().split_whitespace() {
                    model.learn(token.to_lowercase().as_ref());
                }
            }
            model
//            panic!("OneWordCorrector: no tokens");
        }

        fn suggest(&self, word: &str) -> spell_bench::Correction {
//            self.suggest(word)
//            use Correction::*;
            match self.suggest(word) {
                Correction::Correct => spell_bench::Correction::Correct,
                Correction::Suggestion(s) => spell_bench::Correction::Suggestion(s.into()),
                Correction::Incorrect => spell_bench::Correction::Uncorrectable
            }
        }

        type Tokens = spell_bench::DefaultTokenizer;
    }
*/

    /*
    impl Node{

        fn new(key: String) -> Option<Box<Self>> {
            Some(Box::new(Node {
                key,
                value: 0usize,
                children: [
                    None, None, None, None, None, None, None, None, None, None, None, None, None,
                    None, None, None, None, None, None, None, None, None, None, None, None, None,
                ],
            }))
        }

        fn increment_value(&mut self) {
            self.value += 1;
        }


        // Recursive function so risks blowing the stack
        // BUT stack depth is only as deep as the length of the word so not a huge risk
        fn suggest_helper(&self, word: &str, edits: usize) -> Vec<(Correction, usize)> {
            if word.is_empty() { //if reached then all chars in word have been processed
                // return (Correction::Correct, 0) if node.value > 0 and edits == 0
                // return (Correction::Suggestion(node.key), node.value) if node.value > 0 and edits > 0
                // recur (using letter insertion) if node.value == 0 and edits < 2
                // return (Correction::Incorrect, 0) if node.value == 0 and edits >= 2
                if self.value > 0 {
                    if edits == 0 {
                        vec![(Correction::Correct, 0)]
                    }
                    else {
                        vec![(Correction::Suggestion(&self.key), self.value)]
                    }
                }
                else if edits < 2{
                    let mut ls = Vec::new();
                    for child in &self.children {
                        // Insert letter
                        if child.is_some() {
                            ls.append(&mut child.as_ref().unwrap().suggest_helper(word, edits + 1))
                        }
                    }
                    ls
                }
                else{
                    vec![(Correction::Incorrect, 0)]
                }
            }
            else { // still more chars to process
                let child = self.children[c_to_i(word.chars().next().unwrap())].as_ref();
                if child.is_some() {
                    (*(child.unwrap())).suggest_helper(&word[1..], edits)
                }
                else if edits < 2 { // try edits
                    // Delete letter
                    let mut ls = self.suggest_helper(&word[1..], edits + 1);
                    // Replace letter
                    for child in &self.children {
                        if child.is_some() {
                            ls.append(&mut child.as_ref().unwrap().suggest_helper(&word[1..], edits + 1))
                        }
                    }
                    // Insert letter
                    for child in &self.children {
                        if child.is_some() {
                            ls.append(&mut child.as_ref().unwrap().suggest_helper(word, edits + 1))
                        }
                    }
                    // Transpose letters
                    if word.len() >= 2 {
                        let mut new_str = (&word[1..2]).to_owned();
                        new_str.push_str(&word[0..1]);
                        new_str.push_str(&word[2..]);
                        ls.append(&mut self.suggest_helper(&new_str, edits + 1));
                    }
                    ls
                }
                else {
                    vec![(Correction::Incorrect, 0)]
                }
            }
        }

    }
    */

    impl <'a> Node<'a>{

        fn new(key: &'a str) -> Option<Box<Self>> {
            Some(Box::new(Node {
                key,
                value: 0usize,
                children: [
                    None, None, None, None, None, None, None, None, None, None, None, None, None,
                    None, None, None, None, None, None, None, None, None, None, None, None, None,
                ],
            }))
        }

        fn increment_value(&mut self) {
            self.value += 1;
        }


        // Recursive function so risks blowing the stack
        // BUT stack depth is only as deep as the length of the word so not a huge risk
        fn suggest_helper(&self, word: &str, edits: usize) -> Vec<(Correction, usize)> {
            if word.len() == 0 { //if reached then all chars in word have been processed
                // return (Correction::Correct, 0) if node.value > 0 and edits == 0
                // return (Correction::Suggestion(node.key), node.value) if node.value > 0 and edits > 0
                // recur (using letter insertion) if node.value == 0 and edits < 2
                // return (Correction::Incorrect, 0) if node.value == 0 and edits >= 2
                if self.value > 0 {
                    if edits == 0 {
                        vec![(Correction::Correct, 0)]
                    }
                    else {
                        vec![(Correction::Suggestion(String::from(self.key)), self.value)]
                    }
                }
                else {
                    if edits < 2 {
                        let mut ls = Vec::new();
                        for child in &self.children {
                            // Insert letter
                            if child.is_some() {
                                ls.append(&mut child.as_ref().unwrap().suggest_helper(word, edits + 1))
                            }
                        }
                        ls
                    }
                    else {
                        vec![(Correction::Incorrect, 0)]
                    }
                }
            }
            else { // still more chars to process
                let child = self.children[c_to_i(word.chars().next().unwrap())].as_ref();
                if child.is_some() {
                    (*(child.unwrap())).suggest_helper(&word[1..], edits)
                }
                else if edits < 2 { // try edits
                    // Delete letter
                    let mut ls = self.suggest_helper(&word[1..], edits + 1);
                    // Replace letter
                    for child in &self.children {
                        if child.is_some() {
                            ls.append(&mut child.as_ref().unwrap().suggest_helper(&word[1..], edits + 1))
                        }
                    }
                    // Insert letter
                    for child in &self.children {
                        if child.is_some() {
                            ls.append(&mut child.as_ref().unwrap().suggest_helper(word, edits + 1))
                        }
                    }
                    // Transpose letters
                    if word.len() >= 2 {
                        let mut new_str = (&word[1..2]).to_owned();
                        new_str.push_str(&word[0..1]);
                        new_str.push_str(&word[2..]);
                        ls.append(&mut self.suggest_helper(&new_str, edits + 1));
                    }
                    ls
                }
                else {
                    vec![(Correction::Incorrect, 0)]
                }
            }
        }

    }

    #[cfg(test)]
    mod tests {

        #[test]
        fn c_to_i(){
            assert_eq!(0, super::c_to_i('a'));
            assert_eq!(25, super::c_to_i('z'));
            assert_eq!(5, super::c_to_i('f'));
            assert_ne!(0, super::c_to_i('b'));
            assert_ne!(13, super::c_to_i('z'));
        }

        #[test]
        #[should_panic]
        fn c_to_i_should_panic_1(){
            super::c_to_i('!');
        }

        #[test]
        #[should_panic]
        fn c_to_i_should_panic_2(){
            super::c_to_i('\n');
        }

        #[test]
        #[should_panic]
        fn c_to_i_should_panic_3(){
            super::c_to_i('A');
        }

        #[test]
        #[should_panic]
        fn c_to_i_should_panic_4(){
            super::c_to_i('Z');
        }

        #[test]
        fn suggest(){
            //prefix tree
            let mut prefix_tree =  super::CorrectorModel::new();

            //build prefix tree
            let corpus = "foo bar bug bing sing string shoe flight from frog 
                alphabet soccer bingo flick with without dog car cat bird bug cat
                 dog string shoe from foo foo foo bar bar sing string bucket foo 
                 from bar bar bing bank back cat";
            for i in corpus.split_whitespace() {
               prefix_tree.learn(i);
            }

            let input = "sing foo bar alphabet alphbet alphebet alphaabet alhpabet
                         buz orange flightzqr flightzq";
            let expect_suggest = "correct correct correct correct alphabet alphabet 
                                    alphabet alphabet bug - - flight";
            let mut expect_suggest = expect_suggest.split_whitespace();
            for word in input.split_whitespace(){
                let result = prefix_tree.suggest(word);
                match result {
                    super::Correction::Correct => {println!("{}",word);
                        assert_eq!(expect_suggest.next(), Some("correct"))},
                    super::Correction::Incorrect => {println!("{}, -",word);
                        assert_eq!(expect_suggest.next(), Some("-"))},
                    super::Correction::Suggestion(s) => {println!("{}, {}",word, s);
                         assert_eq!(expect_suggest.next(), Some(s.as_ref()))},
                }
            }


        }

    }
}