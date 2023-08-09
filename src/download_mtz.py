import urllib.request
import argparse
import os 
import multiprocessing.pool
from tqdm import tqdm

def worker(data):
    pdb_code, output_dir = data
    url = f'https://edmaps.rcsb.org/coefficients/{pdb_code}.mtz'
    try: 
        urllib.request.urlretrieve(url, os.path.join(output_dir, f"{pdb_code}.mtz"))
    except: 
        return

def main(): 
    parser = argparse.ArgumentParser(
                    prog='MTZ Downloader',
                    description='Download MTZ files from RCSB from a filelist',
                    epilog='Jordan Dialpuri')
    parser.add_argument('-f', '--file_list', required=True)      # option that takes a value
    parser.add_argument('-o', '--output_dir', required=True) 
    parser.add_argument('-m', '--multiprocessing', action='store_true') 
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.file_list):
        print("Could not find that file_list")
        return
    
    data = None
    with open(args.file_list, "r", encoding="UTF-8") as file_list:
        data = [(x.rstrip("\n").lower(), args.output_dir) for x in file_list]
        
    
    if args.multiprocessing: 
       with multiprocessing.Pool() as pool_: 
            x = list(tqdm(pool_.imap_unordered(worker, data), total=len(data)))
    else: 
        for x in tqdm(data, total=len(data)): 
            worker(x, data)


if __name__ == "__main__":
    main()