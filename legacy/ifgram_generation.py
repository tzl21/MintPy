##### for ISCE2 Only #####
import argparse
import datetime
import glob
import inspect
import os
import sys
import subprocess
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from mintpy.objects import  sensor

SENSOR_NAMES = [i.capitalize() for i in sensor.SENSOR_NAMES]
#########################################################################
REFERENCE = """references:

"""

EXAMPLE = """examples:
python script.py    --work-dir /path/to/SLCs --slc ./merged/SLC -w ./ --outdir ifgramStack \
                    --ifgram-dir ifgramStack/ifgrams --num-connections 5 --oneyear-intferograms 30 \
                    --num-workers 4 --continue-on-error --timeout 7200 \
                    --isce-home /path/to/isce/home --isce-root /path/to/isce/root \
                    --stack-path /path/to/stack --tops-stack-path /path/to/topsStack
"""

# TEMPLATE = get_template_content('generate_ifgram')
TEMPLATE = """########## Generate Interferometric Pairs
# generate interferometric pairs based on nearest neighbor connections
# and/or one year interferograms
# parameters:
  num_connections = 10          # number of new pairs for each new acquisition
  oneyear_intferograms = None   # number of days range to search for one year interferograms
  sensor = None                 # Name of sensor, choose from the list below:
                                # ['Alos', 'Alos2', 'CosmoSkymed', 'Envisat', 'Gaofen3', 'Gaofen7',
                                #  'Landsat8', 'NISAR', 'Radarsat2', 'Sentinel1', 'TerraSarX']
"""
#########################################################################
def create_parser():
    parser = argparse.ArgumentParser(description='Generate Interferometric Pairs.',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=REFERENCE+'\n'+TEMPLATE+'\n'+EXAMPLE)

    # Input directories
    
    parser.add_argument('--slc', dest='slc_dirname', default='merged/SLC',
                        help='dir of all merged SLCs')
    
    parser.add_argument('-w','--work-dir', dest='work_dir', default='./',
                       help='Working directory containing date folders')
    
    parser.add_argument('-r','--reference', dest='reference_dir', default='reference',
                       help='Working directory containing date folders')
    
    parser.add_argument('-s','--secondary', dest='secondary_dir', default='coreg_secondarys',
                       help='Working directory containing date folders')
    # Output directories
    parser.add_argument('-o', '--outdir', dest='out_dir', default='ifgramStack',
                        help='Output directory for ifgram_list.txt and interferograms')
    
    parser.add_argument('--ifgram-dir', dest='ifgram_dir', default='ifgramStack/ifgrams',
                       help='Output directory for interferograms')

    # Method for generating network
    method = parser.add_argument_group('Methods to generate the initial network')
    method.add_argument('-n','--num_connections', dest='num_connections', type=int, default=10,
                        help='number of new pairs for each new acquisition')
    
    method.add_argument('--oneyear_intferograms', dest='oneyear_intferograms', type=int, default=None,
                        help='number of days range to search for one year interferograms')
    
    method.add_argument('--sensor', help='Name of sensor, choose from the list below:\n'+str(SENSOR_NAMES))

    # Parameters for generateIgram.py
    igram = parser.add_argument_group('Parameters for generateIgram.py')
    igram.add_argument('--reference-suffix', dest='reference_suffix', default='',
                       help='Suffix for reference burst files')
    
    igram.add_argument('--secondary-suffix', dest='secondary_suffix', default='',
                       help='Suffix for secondary burst files')
    
    igram.add_argument('--overlap', dest='overlap', action='store_true',default=False,
                       help='Process overlap regions')
    
    igram.add_argument('--flatten', dest='flatten', action='store_true', default=False,
                       help='Flatten the interferograms (default: True)')

    # Parallel processing parameters
    parallel = parser.add_argument_group('Parallel processing parameters')
    parallel.add_argument('--num-workers', dest='num_workers', type=int, default=1,
                       help='Number of parallel workers (default: 1 - sequential)')
    
    parallel.add_argument('--continue-on-error', dest='continue_on_error', action='store_true',
                       help='Continue processing other pairs if one fails')
    
    parallel.add_argument('--timeout', dest='timeout', type=int, default=3600,
                       help='Timeout in seconds for each pair processing (default: 3600)')

    # ISCE environment parameters
    isce_env = parser.add_argument_group('ISCE environment parameters')
    isce_env.add_argument('--isce-home', dest='isce_home', required=True,
                       help='Path to ISCE home directory. e.g., /home/xxx/tools/bash/envs/isce2/lib/python3.11/site-packages/isce')
    isce_env.add_argument('--isce-root', dest='isce_root', required=True,
                       help='Path to ISCE root directory. e.g., /home/xxx/tools/bash/envs/isce2/lib/python3.11/site-packages/isce')
    isce_env.add_argument('--stack-path', dest='stack_path', required=True,
                       help='Path to ISCE stack directory. e.g., /home/xxx/tools/isce2/contrib/stack')
    isce_env.add_argument('--tops-stack-path', dest='tops_stack_path', required=True,
                       help='Path to ISCE topsStack directory. e.g., /home/xxx/tools/isce2/contrib/stack/topsStack')

    return parser


def cmd_line_parse(iargs=None):
    parser = create_parser()
    inps = parser.parse_args(args=iargs)

    return inps

#########################################################################
def log(msg):
    """Log function written by Falk"""
    f = open('log', 'a')
    callingFunction = os.path.basename(inspect.stack()[1][1])
    dateStr = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%dT%H:%M:%S')
    string = dateStr+" * "+msg
    print(string)
    f.write(string+"\n")
    f.close()


def read_data_info(slc_dirname):
    """Read date, bperp and/or DOP info
    Parameters: baseline_file : str, path of bl_list.txt file
                reference_file : str, path of ifgramStack.h5 file
                
    Returns:    date_list : list of str in YYMMDD format
    """


    date_list =  sorted([os.path.basename(path) for path in glob.glob(slc_dirname + "/*")])

    return date_list


def selectNeighborPairs(inps):
    """Select nearest neighbor acquisitions to form seqential pairs."""

    date_list = read_data_info(slc_dirname=inps.slc_dirname)
    dateList = date_list
    num_connections = inps.num_connections
    oneyear_intferograms = inps.oneyear_intferograms
    pairs = []
    num_date = len(dateList)

    # translate num_connections input
    if num_connections == 'all':
        num_connections = len(dateList) - 1
    else:
        num_connections = int(num_connections)

    # selecting nearest pairs based on dateList and num_connections
    num_connections = num_connections + 1
    for i in range(num_date-1):
        for j in range(i+1, i+num_connections):
            if j < num_date:
                pairs.append((dateList[i], dateList[j]))
    print('selecting pairs with {} nearest neighbor connections: {}'.format(num_connections-1, len(pairs)))
    
    from datetime import datetime, timedelta
    # selecting one year nearest pairs based on dateList and num_connections


    if oneyear_intferograms != None:
        oneyear_intferograms = int(oneyear_intferograms)
        print("test",dateList)
        dates = np.array([datetime.strptime(date, '%Y%m%d') for date in dateList])
        days_range = np.ptp([datetime.strptime(date, '%Y%m%d').toordinal() for date in dateList])
        if days_range < 365 + 2*oneyear_intferograms:
            print('small days range,can not generate one year int!')
        else :    
            for i, date in enumerate(dates):
                range_1 = date + timedelta(days=365) - timedelta(days=oneyear_intferograms)
                range_2 = date + timedelta(days=365) + timedelta(days=oneyear_intferograms)
                index = np.where((dates >= range_1) * (dates <= range_2))[0]
                if len(index) >= 1:
                    for ind_date in index:
                        date2 = dateList[ind_date]
                        pairs.append((dateList[i], date2))
        print('selecting pairs with {} nearest neighbor connections and {} year range connections: {}'.format(num_connections-1, oneyear_intferograms ,len(pairs)))


    inps.Ifgram_pairs = pairs
    inps.date_list = dateList
    inps.date12_list = ['{}-{}'.format(i[0], i[1]) for i in pairs]
    return inps

def write_ifgram_list(inps):
    # Output directory/filename
    inps.out_dir = os.path.abspath(inps.out_dir)

    if not os.path.isdir(inps.out_dir):
        os.makedirs(inps.out_dir)

    ifgram_num = len(inps.date12_list)

    # Write txt file
    ifgram_file = inps.out_dir + '/ifgram_list.txt'
    f = open(ifgram_file, 'w')
    f.write('#Interferograms configuration generated by ifgram_generation.py\n')
    f.write('#   {:<18}\n'.format('Date12'))

    for i in range(ifgram_num):
        line = '    {:<18}'.format(inps.date12_list[i])
        f.write(line+'\n')
    f.close()
    log(f'write network/pairs info into file: {ifgram_file}')
    return ifgram_file

def setup_unified_paths(inps):
    """
    Set up path mapping for unified structure without actually copying data
    
    Parameters:
    inps: Input parameters object
    
    Returns:
    inps: Updated input parameters object with path mapping
    """
    # Create a mapping from date to actual directory path
    date_to_path = {}
    
    # Reference date
    if hasattr(inps, 'date_list') and inps.date_list:
        ref_date = inps.date_list[0]
        date_to_path[ref_date] = os.path.join(inps.work_dir, inps.reference_dir)
        
        # Secondary dates
        for date in inps.date_list[1:]:
            date_path = os.path.join(inps.work_dir, inps.secondary_dir, date)
            if os.path.exists(date_path):
                date_to_path[date] = date_path
            else:
                print(f"Warning: Directory for date {date} does not exist: {date_path}")
    
    # Store the mapping in inps
    inps.date_to_path = date_to_path
    return inps

def generate_ifgramStack_from_ifgramList(inps, ifgram_file):
    """
    Generate interferogram stack from pair list by calling generateIgram.py for each pair
    
    Parameters:
    inps: Input parameters object
    ifgram_file: Path to interferogram pair list file
    
    Returns:
    inps: Updated input parameters object
    """
    
    # Read interferogram pairs list
    pairs = read_ifgram_list(ifgram_file)
    
    # Check if parallel processing is enabled
    if hasattr(inps, 'num_workers') and inps.num_workers > 1:
        print(f"Starting parallel processing with {inps.num_workers} workers")
        inps = process_pairs_parallel(inps, pairs)
    else:
        print("Starting sequential processing")
        inps = process_pairs_sequential(inps, pairs)
    
    # Update inps object
    inps.ifgram_stack_generated = True
    inps.processed_pairs = pairs
    
    return inps

def process_pairs_sequential(inps, pairs):
    """
    Process interferogram pairs sequentially
    
    Parameters:
    inps: Input parameters object
    pairs: List of interferogram pairs
    
    Returns:
    inps: Updated input parameters object
    """
    for pair in pairs:
        date1, date2 = pair
        process_single_pair(inps, date1, date2)
    
    return inps

def process_pairs_parallel(inps, pairs):
    """
    Process interferogram pairs in parallel using multiple workers
    
    Parameters:
    inps: Input parameters object
    pairs: List of interferogram pairs
    
    Returns:
    inps: Updated input parameters object
    """
    num_workers = min(inps.num_workers, len(pairs), multiprocessing.cpu_count())
    
    print(f"Using {num_workers} parallel workers for {len(pairs)} pairs")
    
    # Use ProcessPoolExecutor for parallel execution
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks using list comprehension with proper variable scope
        future_to_pair = {
            executor.submit(process_single_pair_wrapper, (inps, date1, date2)): (date1, date2)
            for date1, date2 in pairs
        }
        
        # Collect results as they complete
        successful_pairs = []
        failed_pairs = []
        
        for future in as_completed(future_to_pair):
            pair = future_to_pair[future]
            try:
                result = future.result()
                successful_pairs.append(pair)
                print(f"✓ Successfully completed: {pair[0]}-{pair[1]}")
            except Exception as e:
                failed_pairs.append(pair)
                print(f"✗ Failed: {pair[0]}-{pair[1]} - Error: {str(e)}")
                
                # Stop processing if continue_on_error is False
                if not getattr(inps, 'continue_on_error', False):
                    print("Stopping due to error (continue_on_error=False)")
                    # Shutdown remaining tasks
                    for f in future_to_pair:
                        if not f.done():
                            f.cancel()
                    break
        
        print(f"\nProcessing Summary:")
        print(f"Successful: {len(successful_pairs)} pairs")
        print(f"Failed: {len(failed_pairs)} pairs")
        
        if failed_pairs:
            print("Failed pairs:", failed_pairs)
    
    return inps

def process_single_pair_wrapper(args):
    """
    Wrapper function for parallel processing
    
    Parameters:
    args: Tuple of (inps, date1, date2)
    
    Returns:
    bool: True if successful
    """
    inps, date1, date2 = args
    return process_single_pair(inps, date1, date2)

def process_single_pair(inps, date1, date2):
    """
    Process a single interferogram pair by calling generateIgram.py
    
    Parameters:
    inps: Input parameters object
    date1: Reference date
    date2: Secondary date
    
    Returns:
    bool: True if successful, raises exception otherwise
    """

    reference_dir = inps.date_to_path[date1]
    secondary_dir = inps.date_to_path[date2]
    
    # Build output interferogram directory
    ifgram_dir = os.path.join(inps.ifgram_dir, f"{date1}_{date2}")
    os.makedirs(ifgram_dir, exist_ok=True)
    
    # Build command line arguments for generateIgram.py
    cmd_args = [
        'generateIgram.py',
        '-m', reference_dir,      # Reference image directory
        '-s', secondary_dir,      # Secondary image directory
        '-i', ifgram_dir,         # Output interferogram directory
        '-p', 'fine',              # Interferogram prefix
    ]
    
    # Add flatten parameter if enabled
    if getattr(inps, 'flatten', True):
        cmd_args.append('-f')
    
    # Add optional parameters
    if hasattr(inps, 'reference_suffix') and inps.reference_suffix:
        cmd_args.extend(['-x', inps.reference_suffix])
    
    if hasattr(inps, 'secondary_suffix') and inps.secondary_suffix:
        cmd_args.extend(['-y', inps.secondary_suffix])
        
    if hasattr(inps, 'overlap') and inps.overlap:
        cmd_args.append('-v')
    
    print(f"Processing pair: {date1}-{date2}")
    print(f"Reference: {reference_dir}")
    print(f"Secondary: {secondary_dir}")
    print(f"Output: {ifgram_dir}")
    
    # Set up ISCE environment
    env = setup_isce_environment(inps)
    
    # Execute generateIgram.py with custom environment
    try:
        result = subprocess.run(
            cmd_args, 
            check=True, 
            capture_output=True, 
            text=True,
            timeout=getattr(inps, 'timeout', 3600),
            env=env  # Pass the custom environment
        )
        
        if result.stdout:
            print(f"Output for {date1}_{date2}: {result.stdout.strip()}")
            
        return True
        
    except subprocess.CalledProcessError as e:
        error_msg = f"Command failed with return code {e.returncode}: {e.stderr}"
        print(f"Error processing {date1}-{date2}: {error_msg}")
        raise Exception(error_msg)
        
    except subprocess.TimeoutExpired:
        error_msg = f"Processing timed out after {getattr(inps, 'timeout', 3600)} seconds"
        print(f"Error processing {date1}-{date2}: {error_msg}")
        raise Exception(error_msg)
    
def read_ifgram_list(ifgram_file):
    """
    Read interferogram pair list file and extract date pairs
    
    Parameters:
    ifgram_file: Path to interferogram pair list file
    
    Returns:
    list: List of (date1, date2) tuples
    """
    pairs = []
    
    with open(ifgram_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip comment lines and empty lines
            if line.startswith('#') or not line:
                continue
            
            # Parse date pair
            parts = line.split()
            if len(parts) >= 1:
                date_pair = parts[0]
                if '-' in date_pair:
                    date1, date2 = date_pair.split('-')
                    pairs.append((date1, date2))
    
    print(f"Read {len(pairs)} interferogram pairs from {ifgram_file}")
    return pairs

def setup_isce_environment(inps):
    """
    Set up ISCE environment variables for subprocess calls
    
    Returns:
    dict: Environment variables dictionary
    """
    env = os.environ.copy()
    
    # Set ISCE paths
    isce_home = inps.isce_home
    isce_root = inps.isce_root
    stack_path = inps.stack_path
    tops_stack_path = inps.tops_stack_path
    
    # Set environment variables
    env['ISCE_HOME'] = isce_home
    env['ISCE_ROOT'] = isce_root
    
    # Update PATH
    env['PATH'] = f"{isce_home}/bin:{isce_home}/applications:{env['PATH']}"
    env['PATH'] = f"{env['PATH']}:{tops_stack_path}"
    
    # Update PYTHONPATH
    pythonpath_parts = [
        isce_root,
        f"{isce_home}/applications",
        f"{isce_home}/components",
        stack_path
    ]
    
    # Add existing PYTHONPATH if it exists
    existing_pythonpath = env.get('PYTHONPATH', '')
    if existing_pythonpath:
        pythonpath_parts.insert(0, existing_pythonpath)
    
    env['PYTHONPATH'] = ':'.join(pythonpath_parts)
    
    return env


#########################################################################
def main(iargs=None):
    inps = cmd_line_parse(iargs)

    inps.work_dir = os.path.abspath(inps.work_dir)
    inps.slc_dirname = os.path.join(inps.work_dir, inps.slc_dirname)
    # Make sure out_dir and ifgram_dir are relative to work_dir
    if not os.path.isabs(inps.out_dir):
        inps.out_dir = os.path.join(inps.work_dir, inps.out_dir)
    
    if not os.path.isabs(inps.ifgram_dir):
        inps.ifgram_dir = os.path.join(inps.work_dir, inps.ifgram_dir)

    inps = selectNeighborPairs(inps)

    # Set up unified paths without copying data
    inps = setup_unified_paths(inps)



    # inps = reorganize_slc_structure(inps)

    ifgram_file = write_ifgram_list(inps)

    inps = generate_ifgramStack_from_ifgramList(inps, ifgram_file)

    print("Interferogram stack generation completed!")


###########################################################################
if __name__ == '__main__':
    main(sys.argv[1:])
