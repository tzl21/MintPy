#!/usr/bin/env python3
############################################################
# Program is part of MintPy / slc2ifg (moved from insarflow)  #
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""
Script for cropping SLC files in radar coordinates with parallel processing.

This script provides command-line interface to crop multiple SLC files
based on geographic coordinates using lon/lat lookup tables.
"""

import os
import sys
import argparse
import rasterio
import numpy as np
import glob
import concurrent.futures
import logging
from pathlib import Path
from .utils.slc2ifg_utils import (
    create_xml_file,
    tqdm_progress,
)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def parse_arguments(args_list=None):
    """Parse command line arguments.

    Args:
        args_list: Optional list of arguments to parse. If None, uses sys.argv.

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Crop SLC files in radar coordinates with parallel processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic parallel processing with 4 workers
  %(prog)s --input-dir ./input --output-dir ./output --wsen 102.8 27.3 103.3 27.6 --max-workers 4 --geom-dir /path/to/geom

  # With buffer and custom file pattern
  %(prog)s --input-dir /data/slc --pattern "*.slc" --output-dir /output/cropped --wsen 102.5 27.0 103.5 27.8 --buffer 0.01 --max-workers 8 --geom-dir /path/to/geom

  # Using wildcard patterns in input directory
  %(prog)s --input-dir "./*/slc" --pattern "*.slc" --output-dir ./cropped --wsen 102.8 27.3 103.3 27.6 --max-workers 4 --geom-dir /path/to/geom
        """
    )

    # Required arguments
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing input SLC files. Can contain wildcards (e.g., '/path/to/SLC/*/')"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where cropped SLC files will be saved"
    )
    parser.add_argument(
        "--wsen",
        type=float,
        nargs=4,
        required=True,
        metavar=('WEST', 'SOUTH', 'EAST', 'NORTH'),
        help="Crop bounds in WSEN format (West, South, East, North)"
    )
    parser.add_argument(
        "--geom-dir",
        type=str,
        required=True,
        help="Directory containing full coordinate files (must contain lon.rdr.full and lat.rdr.full)"
    )

    # Optional arguments
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.slc",
        help="File pattern for input SLC files (e.g., '*.slc', '*.tif')"
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=0.0,
        help="Buffer to add around the crop area in degrees"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers. If not specified, uses all available CPUs"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix to add to output SLC filenames"
    )
    parser.add_argument(
        "--file-list",
        type=str,
        help="File containing list of input files to process"
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        dest="no_skip_existing",
        default=False,
        help="Do not skip existing files (reprocess everything)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    if args_list is None:
        return parser.parse_args()
    else:
        return parser.parse_args(args_list)


def get_radar_files(args):
    """
    Retrieve a list of radar SLC files to process.

    Args:
        args: Parsed command line arguments

    Returns:
        List of input file paths
    """
    logger = logging.getLogger(__name__)

    if args.file_list:
        # Read files from file list
        try:
            with open(args.file_list, 'r') as f:
                file_list = [line.strip() for line in f if line.strip()]

            # Filter out non-existent files
            file_list = [f for f in file_list if os.path.exists(f)]

            logger.info(f"Read {len(file_list)} files from file list: {args.file_list}")
            return file_list
        except Exception as e:
            logger.error(f"Error reading file list: {e}")
            return []
    else:
        # Get files from directory pattern
        file_list = []
        matched_dirs = glob.glob(args.input_dir)

        if not matched_dirs:
            logger.warning(f"No directories found matching pattern: {args.input_dir}")
            return []

        for dir_path in matched_dirs:
            if os.path.isdir(dir_path):
                search_pattern = os.path.join(dir_path, args.pattern)
                dir_files = glob.glob(search_pattern, recursive=False)
                dir_files = [f for f in dir_files if os.path.isfile(f)]
                file_list.extend(dir_files)

        file_list.sort()
        logger.info(f"Found {len(file_list)} files matching pattern '{args.pattern}' in '{args.input_dir}'")
        return file_list


def generate_output_paths(input_paths, output_dir, prefix=""):
    """
    Generate output file paths for the cropped radar SLC files.

    Args:
        input_paths: List of input file paths
        output_dir: Output directory
        prefix: Prefix for output filenames

    Returns:
        List of output file paths
    """
    os.makedirs(output_dir, exist_ok=True)

    output_paths = []
    for input_path in input_paths:
        input_file = Path(input_path)
        # Fixed ISCE2 SLC naming: output must be {prefix}yyyymmdd.slc
        if '.slc' in input_file.name:
            # Strip trailing format extensions (.tif / .h5 / ...) keeping .slc
            stem = input_file.name.split('.slc')[0] + '.slc'
        else:
            stem = f"{input_file.stem}.slc"
        output_filename = f"{prefix}{stem}"

        output_path = Path(output_dir) / output_filename
        output_paths.append(str(output_path))

    return output_paths


def filter_existing_outputs(input_paths, output_paths):
    """
    Filter out files that already exist in the output directory.

    Args:
        input_paths: List of input file paths
        output_paths: List of output file paths

    Returns:
        Tuple of (filtered_inputs, filtered_outputs)
    """
    logger = logging.getLogger(__name__)
    filtered_inputs = []
    filtered_outputs = []

    for input_path, output_path in zip(input_paths, output_paths):
        if not os.path.exists(output_path):
            filtered_inputs.append(input_path)
            filtered_outputs.append(output_path)
        else:
            logger.info(f"Skipping existing output file: {output_path}")

    return filtered_inputs, filtered_outputs


def find_crop_window_from_full_files(lon_full_path, lat_full_path, wsen_bounds):
    """
    Find the minimum bounding rectangle in radar coordinates for a given geographic area.

    Args:
        lon_full_path: Path to longitude full coordinate file
        lat_full_path: Path to latitude full coordinate file
        wsen_bounds: Tuple of (west, south, east, north) bounds

    Returns:
        Tuple of (min_row, max_row, min_col, max_col) in radar coordinates
    """
    logger = logging.getLogger(__name__)

    # Check if files exist
    if not os.path.exists(lon_full_path):
        raise FileNotFoundError(f"Longitude file not found: {lon_full_path}")
    if not os.path.exists(lat_full_path):
        raise FileNotFoundError(f"Latitude file not found: {lat_full_path}")

    west, south, east, north = wsen_bounds

    logger.info("Loading coordinate files to find crop window...")

    # Open the coordinate files
    with rasterio.open(lon_full_path) as lon_src, rasterio.open(lat_full_path) as lat_src:
        # Get file shapes
        lon_shape = lon_src.shape
        lat_shape = lat_src.shape

        if lon_shape != lat_shape:
            raise ValueError(f"Coordinate file shapes don't match: lon={lon_shape}, lat={lat_shape}")

        # Read data efficiently
        logger.info(f"Reading coordinate data (shape: {lon_shape})...")
        lon_data = lon_src.read(1)
        lat_data = lat_src.read(1)

        # Create mask for pixels within the geographic bounds
        logger.info("Finding pixels within geographic bounds...")
        mask = (lon_data >= west) & (lon_data <= east) & \
               (lat_data >= south) & (lat_data <= north)

        if not np.any(mask):
            raise ValueError("No pixels found within the specified geographic bounds")

        # Get indices of all pixels within bounds
        rows, cols = np.where(mask)

        # Find min and max row/col to create bounding rectangle
        min_row, max_row = np.min(rows), np.max(rows)
        min_col, max_col = np.min(cols), np.max(cols)

        # Add a small buffer to ensure we get all edge pixels
        buffer_pixels = 1
        min_row = max(0, min_row - buffer_pixels)
        max_row = min(lon_shape[0] - 1, max_row + buffer_pixels)
        min_col = max(0, min_col - buffer_pixels)
        max_col = min(lon_shape[1] - 1, max_col + buffer_pixels)

        return min_row, max_row, min_col, max_col


def _write_envi_hdr(hdr_path, width, height, bands, dtype):
    """Write an ENVI .hdr file for a binary raster file."""
    _dtype_map = {
        np.dtype('float32'): 4,
        np.dtype('float64'): 5,
        np.dtype('int16'): 2,
        np.dtype('uint16'): 12,
        np.dtype('int32'): 3,
        np.dtype('uint32'): 13,
        np.dtype('byte'): 1,
    }
    envi_dtype = _dtype_map.get(np.dtype(dtype), 4)
    lines = [
        'ENVI',
        'samples = {}'.format(width),
        'lines   = {}'.format(height),
        'bands   = {}'.format(bands),
        'header offset = 0',
        'file type = ENVI Standard',
        'data type = {}'.format(envi_dtype),
        'interleave = bsq',
        'byte order = 0',
        'band names = {',
    ]
    for b in range(1, bands + 1):
        lines.append('Band {},'.format(b))
    lines.append('}')
    # NB: no 'data ignore value' line — for geometry products (lat/lon/
    # height/los) 0 is a valid value and must not be treated as nodata.
    with open(hdr_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    logger = logging.getLogger(__name__)
    logger.info(f'  Created ENVI .hdr: {os.path.basename(hdr_path)}')

def crop_coordinate_files(geom_dir, crop_window, output_geom_dir,
                          verbose=False, no_skip_existing=False):
    """
    Safely crop all .full coordinate files, band by band.
    This ensures complete correctness for multi-band files.
    """
    logger = logging.getLogger(__name__)

    # Find all .full files in the geom directory
    full_files = glob.glob(os.path.join(geom_dir, "*.full"))

    if not full_files:
        logger.warning(f"No .full files found in directory: {geom_dir}")
        return []

    logger.info(f"Found {len(full_files)} .full files to crop")

    min_row, max_row, min_col, max_col = crop_window

    # Ensure output directory exists
    os.makedirs(output_geom_dir, exist_ok=True)

    results = []

    for full_file in full_files:
        try:
            # Generate output filename
            base_name = os.path.basename(full_file)
            output_name = base_name
            output_path = os.path.join(output_geom_dir, output_name)

            # Skip if output already exists (unless forced to recrop)
            if os.path.exists(output_path) and not no_skip_existing:
                logger.info(f"Skipping existing geometry file: {output_name}")
                results.append((full_file, output_path, True, "Skipped (exists)"))
                continue

            # Determine the correct file to open for reading.
            # For ISCE2 .full files without a matching .hdr, GDAL/rasterio
            # may fall back to the wrong .hdr (e.g., los.hdr for los.rdr.full)
            # resulting in corrupted band reads.  Prefer the .vrt if it exists.
            open_path = full_file
            vrt_path = full_file + '.vrt'
            if os.path.isfile(vrt_path) and not os.path.isfile(full_file + '.hdr'):
                logger.info(f"  Using VRT for {base_name} (no matching .hdr)")
                open_path = vrt_path

            logger.info(f"\nProcessing: {base_name}")

            # Open the source file
            with rasterio.open(open_path) as src:
                src_shape = src.shape
                band_count = src.count

                # Calculate window for cropping
                height = max_row - min_row + 1
                width = max_col - min_col + 1

                # Ensure window is within bounds
                if min_row < 0 or max_row >= src_shape[0] or min_col < 0 or max_col >= src_shape[1]:
                    msg = f"Crop window out of bounds. Image shape: {src_shape}"
                    results.append((full_file, output_path, False, msg))
                    continue

                window = rasterio.windows.Window(min_col, min_row, width, height)
                out_transform = rasterio.windows.transform(window, src.transform)

                # Get metadata from first band to use as template
                band1 = src.read(1, window=window)
                out_dtype = band1.dtype

                # Update metadata for the output file
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "ENVI",
                    "height": height,
                    "width": width,
                    "count": band_count,
                    "dtype": out_dtype,
                    "transform": out_transform,
                    "crs": None,
                })

                logger.info(f"  Creating output with {band_count} bands, shape {height}x{width}")

                # Create output file
                with rasterio.open(output_path, 'w', **out_meta) as dest:
                    # Process each band individually
                    for band_idx in range(1, band_count + 1):
                        # Read this band
                        band_data = src.read(band_idx, window=window)

                        # Verify shape
                        if band_data.shape != (height, width):
                            logger.warning(f"  Band {band_idx} shape mismatch: {band_data.shape} != ({height}, {width})")

                        # Write this band
                        dest.write(band_data, band_idx)

                        if verbose:
                            logger.debug(f"    Band {band_idx}: shape={band_data.shape}, dtype={band_data.dtype}")
                            logger.debug(f"    Band {band_idx}: min={band_data.min()}, max={band_data.max()}")

                    logger.info(f"  Successfully wrote {band_count} bands")

                # Ensure ENVI .hdr is created (rasterio ENVI driver does not
                # create .hdr for non-standard extensions like .full).
                # Without a matching .hdr, GDAL may fall back to a wrong .hdr
                # for files sharing the same base name (e.g., los.hdr for los.rdr.full).
                hdr_path = output_path + '.hdr'
                if not os.path.isfile(hdr_path):
                    _write_envi_hdr(hdr_path, width, height, band_count, out_dtype)

                # Create XML file for cropped geometry
                input_xml_path = full_file + ".xml"
                output_xml_path = output_path + ".xml"
                if os.path.exists(input_xml_path):
                    xml_success = create_xml_file(input_xml_path, output_xml_path, crop_window, output_path)
                    if xml_success and verbose:
                        logger.debug("  Created XML file")
                else:
                    from .utils.slc2ifg_utils import create_xml_for_binary
                    create_xml_for_binary(output_path, family='image',
                                          description=f'Cropped {os.path.basename(full_file)}')

                results.append((full_file, output_path, True, f"Success - {band_count} bands"))

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(f"  ERROR: {error_msg}")
            results.append((full_file, output_path, False, error_msg))

    return results


def crop_radar_slc(input_path, output_path, crop_window, verbose=False):
    """
    Crop a SLC file in radar coordinates.

    Returns:
        (input_path, output_path, success_status, error_message)
    """
    logger = logging.getLogger(__name__)

    if not os.path.exists(input_path):
        return (input_path, output_path, False, f"Input file not found: {input_path}")

    # Warn if file extension is unexpected for radar format
    ext = Path(input_path).suffix.lower()
    if ext not in ['.slc', '.rdr', '.full', '.int', '.unw']:
        logger.warning(f"Input file '{input_path}' has unexpected extension for radar format. Expected .slc/.rdr/.full.")

    try:
        if crop_window is None:
            return (input_path, output_path, False, "Crop window required for radar coordinate processing")

        with rasterio.open(input_path) as src:
            src_shape = src.shape

            min_row, max_row, min_col, max_col = crop_window

            if verbose:
                logger.debug(f"Radar coordinate bounds: rows [{min_row}:{max_row}], cols [{min_col}:{max_col}]")

            # Calculate window for cropping
            height = max_row - min_row + 1
            width = max_col - min_col + 1

            # Ensure window is within bounds
            if min_row < 0 or max_row >= src_shape[0] or min_col < 0 or max_col >= src_shape[1]:
                return (input_path, output_path, False, f"Crop window out of bounds. Image shape: {src_shape}")

            # Read the data using window
            window = rasterio.windows.Window(min_col, min_row, width, height)
            out_image = src.read(window=window)

            # Check if the output image has any data
            if out_image.size == 0:
                return (input_path, output_path, False, "Cropped region contains no data")

            # Update transform for the cropped region
            out_transform = rasterio.windows.transform(window, src.transform)

            # Update metadata for the output file
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "ENVI",
                "height": height,
                "width": width,
                "transform": out_transform,
                "crs": None,
            })

            # Write the cropped data to new file
            with rasterio.open(output_path, 'w', **out_meta) as dest:
                dest.write(out_image)

            # Create XML file for the cropped SLC
            input_xml_path = input_path + ".xml"
            output_xml_path = output_path + ".xml"
            xml_success = create_xml_file(input_xml_path, output_xml_path, crop_window, output_path)

            if verbose and xml_success:
                logger.debug(f"Created XML file: {output_xml_path}")

        if verbose:
            logger.debug(f"Successfully cropped radar coordinate SLC file: {output_path}")
        return (input_path, output_path, True, "Success")

    except Exception as e:
        return (input_path, output_path, False, f"Error cropping SLC file: {str(e)}")


def parallel_crop_slc_files(input_paths, output_paths, crop_window,
                            max_workers=None, verbose=False):
    """
    Efficiently crop multiple SLC files in parallel using ThreadPoolExecutor.
    """
    logger = logging.getLogger(__name__)

    if len(input_paths) != len(output_paths):
        logger.error("Input and output path lists must have the same length")
        return []

    tasks = []
    for i in range(len(input_paths)):
        tasks.append((input_paths[i], output_paths[i], crop_window, verbose))

    logger.info(f"Starting parallel processing of {len(tasks)} SLC files with {max_workers or 'auto'} workers")
    logger.info("Processing mode: Radar coordinates")

    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(crop_radar_slc, *task): task
            for task in tasks
        }

        for future in tqdm_progress(concurrent.futures.as_completed(future_to_task),
                                    total=len(tasks),
                                    desc="Processing SLC files"):
            task = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                input_path, output_path, _, _ = task
                results.append((input_path, output_path, False, f"Task generated exception: {exc}"))

    return results


def print_processing_summary(results, file_type="files"):
    """
    Print a summary of the parallel processing results.
    """
    logger = logging.getLogger(__name__)

    successful = [r for r in results if r[2]]
    failed = [r for r in results if not r[2]]

    logger.info(f"\n{file_type} Processing Summary:")
    logger.info(f"  Successful: {len(successful)} files")
    logger.info(f"  Failed: {len(failed)} files")

    if failed:
        logger.error("\nFailed files:")
        for input_path, output_path, _, error_msg in failed:
            logger.error(f"  {os.path.basename(input_path)}: {error_msg}")


def main(args=None):
    """Main function that can be called with parsed arguments."""
    if args is None:
        args = parse_arguments()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Apply buffer to crop bounds
    crop_bounds = list(args.wsen)
    if args.buffer > 0:
        crop_bounds[0] -= args.buffer
        crop_bounds[1] -= args.buffer
        crop_bounds[2] += args.buffer
        crop_bounds[3] += args.buffer

    logger.info(f"Crop bounds: West={crop_bounds[0]}, South={crop_bounds[1]}, East={crop_bounds[2]}, North={crop_bounds[3]}")
    logger.info(f"Buffer: {args.buffer} degrees")

    # Get input files
    input_files = get_radar_files(args)

    if not input_files:
        logger.error("No input files found.")
        return 1

    # Generate output paths
    output_files = generate_output_paths(input_files, args.output_dir, args.prefix)

    # Filter existing outputs if requested (SLC files only; geometry handled below)
    skip_existing = not args.no_skip_existing
    slc_to_process = len(input_files)
    if skip_existing:
        input_files, output_files = filter_existing_outputs(input_files, output_files)
        slc_skipped = slc_to_process - len(input_files)
        if slc_skipped > 0:
            logger.info(f"Skipping {slc_skipped} existing SLC files, {len(input_files)} to process")
    else:
        logger.info(f"Processing {len(input_files)} SLC files")

    # Validate geom directory
    if not os.path.exists(args.geom_dir):
        logger.error(f"Geom directory not found: {args.geom_dir}")
        return 1

    logger.info(f"Geom directory: {args.geom_dir}")

    # Define paths to full coordinate files
    lon_full_path = os.path.join(args.geom_dir, "lon.rdr.full")
    lat_full_path = os.path.join(args.geom_dir, "lat.rdr.full")

    # Calculate crop window from full coordinate files
    try:
        logger.info("\nCalculating crop window from full coordinate files...")
        crop_window = find_crop_window_from_full_files(lon_full_path, lat_full_path, crop_bounds)
        logger.info(f"Crop window: rows [{crop_window[0]}:{crop_window[1]}], cols [{crop_window[2]}:{crop_window[3]}]")

    except Exception as e:
        logger.error(f"Error calculating crop window: {e}")
        return 1

    # Create output directory for SLC files
    os.makedirs(args.output_dir, exist_ok=True)

    # Create output geom directory in the same parent directory as output directory
    output_parent_dir = os.path.dirname(os.path.abspath(args.output_dir))
    output_geom_dir = os.path.join(output_parent_dir, "geom")

    # Crop geometry files (always process, independent of SLC skip status)
    logger.info(f"\nCropping coordinate files to: {output_geom_dir}")
    coord_results = crop_coordinate_files(
        args.geom_dir, crop_window, output_geom_dir, args.verbose,
        no_skip_existing=args.no_skip_existing)
    print_processing_summary(coord_results, "Coordinate files")

    # Process SLC files in parallel
    logger.info("\nProcessing SLC files...")
    slc_results = parallel_crop_slc_files(
        input_paths=input_files,
        output_paths=output_files,
        crop_window=crop_window,
        max_workers=args.max_workers,
        verbose=args.verbose
    )

    # Print summary for SLC files
    print_processing_summary(slc_results, "SLC files")

    # Return error code if any failures
    if any(not r[2] for r in slc_results) or any(not r[2] for r in coord_results):
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logging.error("\nProcessing interrupted by user")
        sys.exit(130)
    except Exception as e:
        logging.error(f"\nUnexpected error: {e}")
        raise