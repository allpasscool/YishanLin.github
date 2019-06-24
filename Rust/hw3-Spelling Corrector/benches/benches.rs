


#![feature(test)]

extern crate test;


spell_bench::spell_bench! {
    mod benches {
        use correct::trie::CorrectorModel as Corrector;
        bench_parser!();
//        bench_corrector!();
    }
}

