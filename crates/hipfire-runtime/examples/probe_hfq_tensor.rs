use hipfire_runtime::hfq::HfqFile;
use std::{env, fs, path::Path};

fn main() {
    let args: Vec<String> = env::args().collect();
    assert!(args.len() >= 3, "usage: probe_hfq_tensor <model> <tensor> [output]");
    let hfq = HfqFile::open(Path::new(&args[1])).expect("open HFQ");
    let (info, data) = hfq.tensor_data_vec(&args[2]).expect("tensor not found");
    println!("name={} qt={} group={} shape={:?} bytes={}",
        info.name, info.quant_type, info.group_size, info.shape, info.data_size);
    if let Some(output) = args.get(3) {
        fs::write(output, data).expect("write tensor");
        println!("wrote={output}");
    }
}
