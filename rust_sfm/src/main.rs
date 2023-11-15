extern crate dng;

use std::fs;
use dng::DngReader;


fn read_dng_file(file_path: &str) -> Result<DngReader<fs::File>, std::io::Error> {
    let file = fs::File::open(file_path)?;
    let reader = DngReader::read(file).expect("Failed to read DNG file");
    Ok(reader)
}

fn list_dng_files_in_dir(directory_path: &str) -> Result<Vec<std::path::PathBuf>, std::io::Error> {
    // Get list of all files in directory
    let mut dng_files = fs::read_dir(directory_path)?
        .map(|res| res.map(|e| e.path()))
        .collect::<Result<Vec<_>, std::io::Error>>()?;
    // Remove any non-DNG files
    dng_files.retain(|file_path| file_path.extension().unwrap() == "dng");
    // Sort the list of files
    dng_files.sort();
    // Return the list of files
    Ok(dng_files)
}

fn main() {
    // Get current working directory
    let cwd = std::env::current_dir().unwrap();
    // Get the root directory of the repo
    let root_dir = cwd.parent().unwrap();
    // Get the image directory
    let image_dir = root_dir.join("test_imgs");
    // Get list of files in image directory (only keep elements with 'dng' extension)
    let image_files = list_dng_files_in_dir(image_dir.to_str().unwrap()).unwrap();
    // Iterate through each 
}
