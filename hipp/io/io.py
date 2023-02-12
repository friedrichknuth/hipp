import glob
import gzip
import os
import pathlib
import shutil
from subprocess import Popen, PIPE, STDOUT
from tqdm import tqdm
import concurrent

import hipp.io


"""
Library interface with command line and file system.
"""

def gunzip_dir(input_directory,
               keep    = False,
               verbose = False):
    print('gunzipping files in', input_directory)
    files = sorted(glob.glob(os.path.join(input_directory,'*.gz')))
    calls = []
    for f in files:
        call = ['gunzip', f]
        if keep:
            call.extend('-k')
        calls.append(call)
#         hipp.io.run_command(call, verbose = verbose)
            
    with tqdm(total=len(calls)) as pbar:
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        futures = {pool.submit(hipp.io.run_command, x): x for x in calls}
        for future in concurrent.futures.as_completed(futures):
            r = future.result()
            pbar.update(1)

    if not keep:
        files = sorted(glob.glob(os.path.join(input_directory,'*.gz')))
        for f in files:
            print(os.path.splitext(f)[0], 'already exists. -- skipped')

def gzip_file(fn):
    fn_out=pathlib.Path(fn).with_suffix('')
    with gzip.open(fn, 'r') as f_in, open(fn_out, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    
def gzip_dir(input_directory,
               keep    = False):
    print('gzipping files in', input_directory)
    files = sorted(glob.glob(os.path.join(input_directory,'*.gz')))
    with tqdm(total=len(files)) as pbar:
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        futures = {pool.submit(gzip_file, x): x for x in files}
        for future in concurrent.futures.as_completed(futures):
            r = future.result()
            pbar.update(1)
    
#     for fn in files:
#         bn=os.path.basename(fn).split('.gz')[0]   
#         newpath=os.path.join(input_directory,bn)

#         with gzip.open(fn, 'r') as f_in, open(newpath, 'wb') as f_out:
#             shutil.copyfileobj(f_in, f_out)
    if not keep:
        for fn in files:
            os.remove(fn)

def move_files(input_directory, 
               output_directory, 
               extension):
    pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(os.path.join(input_directory,'*'+extension)))
    for f in files:
        path, name, ext = hipp.io.split_file(f)
        out = os.path.join(output_directory, name+ext)
        pathlib.Path(f).rename(out)
        
def split_file(file_path_and_name):
    file_path = os.path.split(file_path_and_name)[0]
    file_name = os.path.splitext(os.path.split(file_path_and_name)[-1])[0]
    file_extension = os.path.splitext(os.path.split(file_path_and_name)[-1])[-1]
    return file_path, file_name, file_extension
    
def run_command(command, verbose=False, log_directory=None, shell=False):
    p = Popen(command,
              stdout=PIPE,
              stderr=STDOUT,
              shell=shell)
    if log_directory != None:
        log_file_name = os.path.join(log_directory,command[0]+'_log.txt')
        hsfm.io.create_dir(log_directory)
    
        with open(log_file_name, "w") as log_file:
            
            while p.poll() is None:
                line = (p.stdout.readline()).decode('ASCII').rstrip('\n')
                if verbose == True:
                    print(line)
                log_file.write(line)
        return log_file_name
    else:
        while p.poll() is None:
            line = (p.stdout.readline()).decode('ASCII').rstrip('\n')
            if verbose == True:
                print(line)