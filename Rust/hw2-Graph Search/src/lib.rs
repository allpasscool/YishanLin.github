//! Provides functions and struct for building a graph

///Build a graph 
///
/// #Examples
/// 
/// ```
/// 
/// 
/// ```
use std::collections::HashMap;

///Graphy with adjacency matrix and hashmap for transfer name to number
pub struct Graph{
    adjacency_list: Vec<Vec<usize>>,
    map: HashMap<String, usize>,
}

///default acjacency_list and map are empty
impl Default for Graph{
    fn default() -> Graph{
        Graph{
            adjacency_list: Vec::new(),
            map: HashMap::new(),
        }
    }
}

impl Graph{
    ///transfer name to number
    pub fn name_to_int(&self, name: &String) -> usize{
        self.map[name]
    }

    ///run dfs to find a path
    pub fn dfs(&self, source: &String, target: &String, words: &Vec<String>){
        let mut trace: Vec<usize> = Vec::new();
        let mut visited: Vec<usize> = vec![];

        //check if input are known
        if !self.map.contains_key(source){
            println!("Unknown node: {}", source);
            return;
        }
        let source_int = self.name_to_int(&source);
        
        //check if input are known
        if !self.map.contains_key(target){
            println!("Unknown node: {}", target);
            return;
        }
        let target_int = self.name_to_int(&target);
        let mut now = source_int;

        visited.push(source_int);
        trace.push(source_int);

        loop{
            //println!("trace: {:?}", trace);
            //got it
            if self.adjacency_list[now][target_int] == 1{
                println!("got it");
                visited.push(target_int);
                trace.push(target_int);

                for i in 0..trace.len(){
                    print!("{} ", words[trace[i]]);
                }
                println!();
                trace.clear();
                visited.clear();
                break;
            }
            //look for another
            let mut end = true;
            for i in 0..words.len(){
                let mut vis = false;
                //visited before
                if self.adjacency_list[now][i] == 1 {
                    for j in &visited{
                        if i == *j{
                            vis = true;
                            break;
                        }
                    }
                }
                //not visited
                if self.adjacency_list[now][i]== 1 && !vis{
                    end = false;
                    now = i;
                    visited.push(i);
                    trace.push(i);
                    break;
                }
            }
            //reach end
            if end{
                //println!("reach end, pop!");
                trace.pop();
                if trace.len() == 0{
                    println!("doesn't find a path");
                    break;
                }
                now = trace[trace.len()-1];
            }
        }
    }

    ///init adjacency list
    pub fn init_adj_matrix(&mut self, words: &Vec<String>){
        for _ in 0..words.len(){
            let mut tmp_adj_list = vec![];
            for _ in 0..words.len(){
                tmp_adj_list.push(0);
            }
            self.adjacency_list.push(tmp_adj_list);
        }
    }

    ///build adjacency list
    pub fn build_adj_matrix(&mut self, lines: &Vec<Vec<&str>>){
        for i in 0..lines.len(){
            let source = self.map[lines[i][0]];
            for j in 1..lines[i].len(){
                let neighbor = self.map[lines[i][j]];
                self.adjacency_list[source][neighbor] = 1;
                self.adjacency_list[neighbor][source] = 1;
            }
        }
    }

    ///pair words with integet
    pub fn build_map(&mut self, words: &Vec<String>){
        let mut count = 0;
        for i in words{
            self.map.insert(String::from(i.clone()), count);
            count += 1;
        }
        println!("graph maps: {:?}", self.map);
    }
}


#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn name_to_int(){
        let mut graph = Graph::default();
        graph.map.insert("abc".to_string(), 0);
        graph.map.insert("def".to_string(), 1);
        graph.map.insert("ghi".to_string(), 2);
        assert_eq!(graph.name_to_int(&"abc".to_string()), 0);
        assert_eq!(graph.name_to_int(&"def".to_string()), 1); 
        assert_eq!(graph.name_to_int(&"ghi".to_string()), 2);  
    }

    #[test]
    fn dfs(){
        let mut graph = Graph::default();
        let mut words: Vec<String> = Vec::new();
        words.push("happy".to_string());
        words.push("bird".to_string());
        words.push("QQ".to_string());
        graph.init_adj_matrix(&words);
        let lines: Vec<Vec<&str>> = vec![vec!["happy", "bird", "QQ"], vec!["bird", "happy", "QQ"], vec!["QQ", "happy", "bird"]];
        graph.build_map(&words);
        graph.build_adj_matrix(&lines);
         assert_eq!(graph.adjacency_list[graph.name_to_int(&"happy".to_string())][graph.name_to_int(&"QQ".to_string())], 1);
    }

    #[test]
    fn build_adj_matrix(){
        let mut graph = Graph::default();
        let mut words: Vec<String> = Vec::new();
        words.push("happy".to_string());
        words.push("bird".to_string());
        words.push("QQ".to_string());
        graph.init_adj_matrix(&words);
        let lines: Vec<Vec<&str>> = vec![vec!["happy", "bird", "QQ"], vec!["bird", "happy", "QQ"], vec!["QQ", "happy", "bird"]];
        graph.build_map(&words);
        graph.build_adj_matrix(&lines);
        assert_eq!(graph.adjacency_list[0][1], 1);
        assert_eq!(graph.adjacency_list[0][2], 1);
        assert_eq!(graph.adjacency_list[1][0], 1);
        assert_eq!(graph.adjacency_list[1][2], 1);
        assert_eq!(graph.adjacency_list[2][0], 1);
        assert_eq!(graph.adjacency_list[2][1], 1);
        assert_eq!(graph.adjacency_list[2][2], 0); 
    }

    #[test]
    fn build_map(){
        let mut graph = Graph::default();
        let mut words: Vec<String> = Vec::new();
        words.push("happy".to_string());
        words.push("QQ".to_string());
        words.push("ABC".to_string());
        graph.build_map(&words);
        assert_eq!(graph.map["happy"], 0); 
        assert_eq!(graph.map["QQ"], 1); 
        assert_eq!(graph.map["ABC"], 2); 
    }
}