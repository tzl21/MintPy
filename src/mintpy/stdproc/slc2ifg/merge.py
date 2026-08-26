#!/usr/bin/env python3
############################################################
# Program is part of MintPy / slc2ifg (moved from insarflow)  #
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""
Enhanced script to merge various data types with support for .h5 geometry files.
"""

import argparse
import re
import sys
import h5py
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from collections import defaultdict
import traceback
import tempfile

# Import the stitching utilities
from .utils.stitching_utils import merge_images, DEFAULT_TIFF_OPTIONS
from osgeo import gdal, osr
from .utils.slc2ifg_utils import get_input_epsg, convert_bounds_to_target_epsg

# Configure GDAL
gdal.UseExceptions()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments(args_list: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments with enhanced .h5 support."""
    parser = argparse.ArgumentParser(
        description="Merge various data types across multiple burst IDs with enhanced .h5 support."
    )

    parser.add_argument(
        "--processor",
        type=str,
        choices=['isce2', 'isce3'],
        required=True,
        help="Processor type: 'isce2' (radar coordinates, ENVI format) or 'isce3' (geocoded, GeoTIFF/HDF5)"
    )

    parser.add_argument(
        "--burst-ids",
        nargs="+",
        default=None,
        help="List of burst IDs to merge (e.g., t124_264305_iw2). If not provided, auto-detected from directory structure."
    )

    parser.add_argument(
        "--mode",
        nargs="+",
        choices=["baseline", "intf", "slc", "geometry"],
        required=True,
        help="Processing mode(s): baseline, intf, slc, geometry"
    )

    # Path arguments for different data types
    parser.add_argument(
        "--baseline-dir",
        default=".",
        help="Base directory for baseline files (default: current directory)"
    )

    parser.add_argument(
        "--intf-dir",
        default=".",
        help="Base directory for interferogram files (default: current directory)"
    )

    parser.add_argument(
        "--slc-dir",
        default=".",
        help="Base directory for SLC files (default: current directory)"
    )

    parser.add_argument(
        "--geom-dir",
        default=".",
        help="Base directory for geometry files (default: current directory)"
    )

    parser.add_argument(
        "--output-dir",
        default="./merged",
        help="Output directory for merged files (default: ./merged)"
    )

    parser.add_argument(
        "--intf-types",
        nargs="+",
        default=[".int.tif", "_phsig.coh.tif", "_cpx.coh.tif", ".unw.tif", ".unw.conncomp.tif"],
        help="Interferogram file types to process (default: .int.tif _phsig.coh.tif _cpx.coh.tif .unw.tif .unw.conncomp.tif)"
    )

    parser.add_argument(
        "--geom-types",
        nargs="+",
        default=["height.tif", "layover_shadow_mask.tif", "local_incidence_angle.tif",
                "los_east.tif", "los_north.tif"],
        help="Geometry file types to process"
    )

    # Enhanced .h5 specific parameters
    parser.add_argument(
        "--h5-dataset-mapping",
        type=str,
        help="JSON string or path to JSON file mapping output filenames to .h5 dataset paths. "
             "Example: '{\"dem.tif\": \"/data/z\", \"los_east.tif\": \"/data/los_east\"}'"
    )

    parser.add_argument(
        "--h5-group",
        default="/data",
        help="Base HDF5 group path for geometry datasets (default: /data)"
    )

    parser.add_argument(
        "--h5-x-coords",
        default="x_coordinates",
        help="Dataset name for x coordinates in .h5 file (default: x_coordinates)"
    )

    parser.add_argument(
        "--h5-y-coords",
        default="y_coordinates",
        help="Dataset name for y coordinates in .h5 file (default: y_coordinates)"
    )

    parser.add_argument(
        "--h5-projection",
        default="projection",
        help="Dataset name for projection information in .h5 file (default: projection)"
    )

    # Stitching parameters
    parser.add_argument(
        "--out-bounds",
        nargs=4,
        type=float,
        metavar=("LEFT", "BOTTOM", "RIGHT", "TOP"),
        help="Output bounds for merged images in EPSG:4326 (WGS84) coordinates: left bottom right top"
    )

    parser.add_argument(
        "--out-bounds-epsg",
        type=int,
        default=4326,
        help="EPSG code for the output bounds (default: 4326 for WGS84)"
    )

    parser.add_argument(
        "--dest-epsg",
        type=int,
        help="EPSG code for the output projection. If not specified, uses the EPSG of input data."
    )

    parser.add_argument(
        "--out-nodata",
        type=float,
        help="Nodata value for output files"
    )

    parser.add_argument(
        "--in-nodata",
        type=float,
        help="Override input files' nodata value during merging"
    )

    parser.add_argument(
        "--resample-alg",
        default="lanczos",
        choices=["near", "bilinear", "cubic", "cubicspline", "lanczos", "average", "mode"],
        help="Resampling algorithm (default: lanczos)"
    )

    parser.add_argument(
        "--driver",
        default="GTiff",
        help="GDAL driver for output files (default: GTiff)"
    )

    parser.add_argument(
        "--output-suffix",
        default=".tif",
        help="Suffix for output files (default: .tif)"
    )

    parser.add_argument(
        "--output-prefix",
        default="",
        help="Prefix for output files before date"
    )

    parser.add_argument(
        "--options",
        nargs="+",
        default=DEFAULT_TIFF_OPTIONS,
        help="GDAL creation options for output files"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files"
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of parallel workers for stitching (default: 1)"
    )

    parser.add_argument(
        "--target-aligned-pixels",
        action="store_true",
        default=True,
        help="Adjust output image bounds so pixel coordinates are integer multiples of pixel size"
    )

    parser.add_argument(
        "--strides",
        nargs=2,
        type=int,
        metavar=("X_STRIDE", "Y_STRIDE"),
        default=[1, 1],
        help="Subsampling factors: x_stride y_stride (default: 1 1)"
    )

    parser.add_argument(
        "--out-dtype",
        help="Output data type (e.g., float32, uint8)"
    )

    parser.add_argument(
        "--create-only",
        action="store_true",
        help="Create empty output file, do not write data"
    )

    parser.add_argument(
        "--keep-temp-files",
        action="store_true",
        help="Keep temporary extracted .h5 files for debugging"
    )

    parser.add_argument(
        "--force-epsg-conversion",
        action="store_true",
        default=False,
        help="Force EPSG conversion even if data EPSG cannot be detected (default: False)"
    )

    if args_list is None:
        return parser.parse_args()
    else:
        return parser.parse_args(args_list)


def prepare_stitching_parameters_with_epsg_conversion(
    file_list: List[Path],
    args: argparse.Namespace,
    mode: Optional[str] = None,
    out_nodata: Optional[float] = None
) -> Dict[str, Any]:
    """
    Prepare stitching parameters with automatic EPSG conversion.

    Parameters
    ----------
    file_list : List[Path]
        List of input files
    args : argparse.Namespace
        Command line arguments
    mode : str, optional
        Processing mode
    out_nodata : float, optional
        Explicit nodata value for output

    Returns
    -------
    Dict[str, Any]
        Prepared stitching parameters with EPSG-converted bounds
    """
    params = {
        "driver": args.driver,
        "target_aligned_pixels": args.target_aligned_pixels,
        "out_nodata": args.out_nodata,
        "in_nodata": args.in_nodata,
        "resample_alg": args.resample_alg,
        "overwrite": args.overwrite,
        "options": args.options,
        "create_only": args.create_only,
    }

    # Handle strides
    if args.strides:
        params["strides"] = {"x": args.strides[0], "y": args.strides[1]}

    # Handle output dtype
    if args.out_dtype:
        try:
            params["out_dtype"] = getattr(np, args.out_dtype)
        except AttributeError:
            logger.error(f"Invalid data type: {args.out_dtype}")
            raise

    if args.out_nodata is None and out_nodata is not None:
        logger.info(f"Using inferred nodata value: {out_nodata}")
        params["out_nodata"] = out_nodata

    # Handle bounds and EPSG conversion
    if args.out_bounds:
        bounds = tuple(args.out_bounds)

        # Try to get input EPSG from file list
        input_epsg = get_input_epsg(file_list)

        logger.info(f"Input EPSG detected: {input_epsg}")
        logger.info(f"Output bounds EPSG: {args.out_bounds_epsg}")
        logger.info(f"Destination EPSG: {args.dest_epsg}")

        # ``stitch_arrays`` requires the bounds in EPSG:4326 (it reprojects
        # them to the output CRS internally) — convert the user bounds to
        # 4326 here.  Pre-converting to the *input* CRS used to cause a
        # double transformation on UTM inputs, and a conversion failure used
        # to silently drop the crop box (full-union merge).
        try:
            bounds4326 = convert_bounds_to_target_epsg(
                bounds, args.out_bounds_epsg or 4326, 4326
            )
        except Exception as e:
            raise ValueError(
                f"Failed to convert out_bounds {bounds} "
                f"(EPSG:{args.out_bounds_epsg}) to EPSG:4326: {e}"
            ) from e
        params["out_bounds"] = bounds4326
        params["out_bounds_epsg"] = 4326
        if input_epsg:
            params["dest_epsg"] = input_epsg
            logger.info(f"Output bounds (EPSG:4326): {bounds4326}")
        else:
            logger.warning(
                "Cannot determine input EPSG — the merged product CRS will "
                "follow the first input file")

    elif args.dest_epsg is not None:
        params["dest_epsg"] = args.dest_epsg

    return params


def merge_baselines(
    burst_ids: List[str],
    baseline_dir: Path,
    output_dir: Path,
    overwrite: bool = False
) -> Dict[str, Dict[str, float]]:
    """
    Merge baseline files by averaging Bperp and Bpar values.

    Parameters
    ----------
    burst_ids : List[str]
        List of burst IDs
    baseline_dir : Path
        Base directory containing baseline files
    output_dir : Path
        Output directory for merged baselines
    overwrite : bool
        Whether to overwrite existing files

    Returns
    -------
    Dict[str, Dict[str, float]]
        Dictionary mapping date pairs to averaged baseline values
    """
    output_dir = output_dir / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dictionary to store baseline values by date pair
    baseline_values = defaultdict(lambda: {"Bperp": [], "Bpar": []})

    # Pattern to parse baseline values
    bperp_pattern = re.compile(r"Bperp\s*\(m\):\s*(-?\d+\.?\d*)")
    bpar_pattern = re.compile(r"Bpar\s*\(m\):\s*(-?\d+\.?\d*)")

    for burst_id in burst_ids:
        burst_dir = baseline_dir / burst_id

        if not burst_dir.exists():
            logger.warning(f"Baseline directory not found for {burst_id}: {burst_dir}")
            continue

        # Find all baseline files
        for baseline_file in burst_dir.glob("*.txt"):
            # Extract date pair from filename (e.g., 20240107_20240119)
            date_pair = baseline_file.stem

            try:
                with open(baseline_file, 'r') as f:
                    content = f.read()

                # Parse Bperp and Bpar values
                bperp_match = bperp_pattern.search(content)
                bpar_match = bpar_pattern.search(content)

                if bperp_match and bpar_match:
                    bperp = float(bperp_match.group(1))
                    bpar = float(bpar_match.group(1))

                    baseline_values[date_pair]["Bperp"].append(bperp)
                    baseline_values[date_pair]["Bpar"].append(bpar)
                else:
                    logger.warning(f"Could not parse baseline values from {baseline_file}")

            except Exception as e:
                logger.error(f"Error reading {baseline_file}: {e}")

    # Calculate averages and write output files
    results = {}
    for date_pair, values in baseline_values.items():
        if values["Bperp"] and values["Bpar"]:
            bperp_avg = sum(values["Bperp"]) / len(values["Bperp"])
            bpar_avg = sum(values["Bpar"]) / len(values["Bpar"])

            output_file = output_dir / f"{date_pair}.txt"

            if output_file.exists() and not overwrite:
                logger.info(f"{output_file} already exists, skipping")
                continue

            with open(output_file, 'w') as f:
                f.write(f"Bperp average (m): {bperp_avg}\n")
                f.write(f"Bpar average (m): {bpar_avg}\n")

            results[date_pair] = {"Bperp": bperp_avg, "Bpar": bpar_avg}
            logger.info(f"Merged baseline for {date_pair}: Bperp={bperp_avg:.6f}, Bpar={bpar_avg:.6f}")

    return results


def load_h5_dataset_mapping(mapping_arg: Optional[str]) -> Dict[str, str]:
    """
    Load HDF5 dataset mapping from JSON string or file.

    Parameters
    ----------
    mapping_arg : Optional[str]
        JSON string or path to JSON file

    Returns
    -------
    Dict[str, str]
        Mapping from output filename to HDF5 dataset path
    """
    if not mapping_arg:
        # Default mapping based on common naming conventions
        return {
            "height.tif": "z",
            "layover_shadow_mask.tif": "layover_shadow_mask",
            "local_incidence_angle.tif": "local_incidence_angle",
            "los_east.tif": "los_east",
            "los_north.tif": "los_north",
        }

    # Check if argument is a JSON file path
    mapping_path = Path(mapping_arg)
    if mapping_path.exists():
        try:
            with open(mapping_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON file {mapping_path}: {e}")
            raise

    # Try to parse as JSON string
    try:
        return json.loads(mapping_arg)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON string: {e}")
        raise


def extract_h5_geometry(
    h5_file: Path,
    output_dir: Path,
    geom_types: List[str],
    h5_group: str = "/data",
    dataset_mapping: Optional[Dict[str, str]] = None,
    x_coords_name: str = "x_coordinates",
    y_coords_name: str = "y_coordinates",
    projection_name: str = "projection"
) -> Dict[str, Dict[str, Any]]:
    """
    Extract geometry datasets from HDF5 file with enhanced configuration.

    Returns dict with 'file_list' (Path) and 'nodata' (float) per geometry type.
    """
    if dataset_mapping is None:
        dataset_mapping = {}

    extracted_files = defaultdict(lambda: {'file_list': [], 'nodata': None})

    try:
        with h5py.File(h5_file, 'r') as h5f:
            # Navigate to the specified group
            if h5_group:
                if h5_group not in h5f:
                    logger.error(f"Group {h5_group} not found in {h5_file}")
                    return extracted_files
                data_group = h5f[h5_group]
            else:
                data_group = h5f

            # Get the projection information
            epsg = None
            spatial_ref_wkt = None

            if projection_name in data_group:
                projection_ds = data_group[projection_name]

                # Method 1: Check for epsg_code attribute
                if 'epsg_code' in projection_ds.attrs:
                    try:
                        epsg = int(projection_ds.attrs['epsg_code'])
                        logger.info(f"Got EPSG from epsg_code attribute: {epsg}")
                    except Exception as e:
                        logger.warning(f"Failed to read epsg_code attribute: {e}")

                # Method 2: Check for spatial_ref attribute
                if 'spatial_ref' in projection_ds.attrs and epsg is None:
                    try:
                        spatial_ref = projection_ds.attrs['spatial_ref']
                        if isinstance(spatial_ref, bytes):
                            spatial_ref_wkt = spatial_ref.decode('utf-8')
                        else:
                            spatial_ref_wkt = str(spatial_ref)

                        srs_temp = osr.SpatialReference()
                        srs_temp.ImportFromWkt(spatial_ref_wkt)
                        epsg_code = srs_temp.GetAuthorityCode(None)
                        if epsg_code:
                            epsg = int(epsg_code)
                            logger.info(f"Got EPSG from spatial_ref WKT: {epsg}")
                    except Exception as e:
                        logger.warning(f"Failed to parse spatial_ref attribute: {e}")

                # Method 3: Read dataset value
                if epsg is None:
                    try:
                        proj_value = projection_ds[()]
                        if isinstance(proj_value, (int, np.integer)):
                            epsg = int(proj_value)
                            logger.info(f"Got EPSG from dataset value: {epsg}")
                        elif isinstance(proj_value, bytes):
                            proj_str = proj_value.decode('utf-8')
                            epsg_match = re.search(r'EPSG["\']?\s*:\s*(\d+)', proj_str)
                            if epsg_match:
                                epsg = int(epsg_match.group(1))
                                logger.info(f"Got EPSG from decoded string: {epsg}")
                    except Exception as e:
                        logger.warning(f"Failed to read dataset value: {e}")

            # If still no EPSG, use default 4326
            if epsg is None:
                epsg = 4326
                logger.warning(f"Could not determine EPSG, using default: {epsg}")
            else:
                logger.info(f"Determined EPSG: {epsg}")

            for geom_type in geom_types:
                # Check if this geometry type is in our mapping
                dataset_path = dataset_mapping.get(geom_type)

                if dataset_path is None:
                    # Try to guess dataset name from filename
                    dataset_name = Path(geom_type).stem
                    if dataset_name in data_group:
                        dataset_path = dataset_name
                    else:
                        logger.warning(f"No mapping found for {geom_type} and could not guess dataset name")
                        continue

                # Navigate through nested groups if needed
                if '/' in dataset_path:
                    parts = dataset_path.strip('/').split('/')
                    current = data_group
                    found = True
                    for part in parts:
                        if part in current:
                            current = current[part]
                        else:
                            logger.warning(f"Dataset path {dataset_path} not found in {h5_file}")
                            found = False
                            break
                    if not found:
                        continue
                    dataset = current
                else:
                    if dataset_path in data_group:
                        dataset = data_group[dataset_path]
                    else:
                        logger.warning(f"Dataset {dataset_path} not found in {h5_file}")
                        continue

                # Read data
                data = dataset[:]
                data_dtype = dataset.dtype
                processed_data = data.copy()

                # Find nodata value
                nodata_value = None

                if '_FillValue' in dataset.attrs:
                    fill_value = dataset.attrs['_FillValue']
                    if hasattr(fill_value, 'item'):
                        nodata_value = fill_value.item()
                    else:
                        nodata_value = fill_value
                    logger.info(f"{geom_type}: Found _FillValue attribute: {nodata_value}")

                if nodata_value is None:
                    if np.issubdtype(data_dtype, np.floating):
                        nodata_value = np.nan
                        logger.info(f"{geom_type}: Using default nodata for float: NaN")
                    elif np.issubdtype(data_dtype, np.integer):
                        dtype_info = np.iinfo(data_dtype)
                        nodata_value = dtype_info.max
                        logger.info(f"{geom_type}: Using default nodata for integer: {nodata_value}")
                    else:
                        nodata_value = 0
                        logger.info(f"{geom_type}: Using default nodata: {nodata_value}")

                # Get geotransform from coordinates
                geotransform = None
                if x_coords_name in data_group and y_coords_name in data_group:
                    try:
                        x_coords = data_group[x_coords_name][:]
                        y_coords = data_group[y_coords_name][:]

                        if len(x_coords) > 1 and len(y_coords) > 1:
                            dx = abs(x_coords[1] - x_coords[0])
                            dy = abs(y_coords[1] - y_coords[0])
                            left = x_coords[0] - dx/2
                            top = y_coords[0] + dy/2

                            geotransform = (left, dx, 0, top, 0, -dy)

                            logger.info(f"Computed geotransform for {h5_file.stem}/{dataset_path}:")
                            logger.info(f"  Left: {left:.2f}, Top: {top:.2f}")
                            logger.info(f"  Pixel size: {dx:.2f} x {abs(dy):.2f}")
                            logger.info(f"  Grid size: {len(x_coords)} x {len(y_coords)}")
                    except Exception as e:
                        logger.warning(f"Could not extract coordinates from {h5_file}: {e}")

                # Create unique subdirectory based on HDF5 filename
                h5_stem = h5_file.stem
                file_output_dir = output_dir / h5_stem
                file_output_dir.mkdir(parents=True, exist_ok=True)

                output_file = file_output_dir / geom_type

                # Save as GeoTIFF using GDAL
                try:
                    driver = gdal.GetDriverByName('GTiff')
                    if driver is None:
                        raise RuntimeError("GTiff driver not available")

                    rows, cols = processed_data.shape

                    # Convert numpy dtype to GDAL dtype
                    if processed_data.dtype == np.float32:
                        gdal_dtype = gdal.GDT_Float32
                    elif processed_data.dtype == np.float64:
                        gdal_dtype = gdal.GDT_Float64
                    elif processed_data.dtype == np.int8:
                        gdal_dtype = gdal.GDT_Int8
                    elif processed_data.dtype == np.int16:
                        gdal_dtype = gdal.GDT_Int16
                    elif processed_data.dtype == np.int32:
                        gdal_dtype = gdal.GDT_Int32
                    elif processed_data.dtype == np.uint8:
                        gdal_dtype = gdal.GDT_Byte
                    elif processed_data.dtype == np.uint16:
                        gdal_dtype = gdal.GDT_UInt16
                    elif processed_data.dtype == np.uint32:
                        gdal_dtype = gdal.GDT_UInt32
                    else:
                        processed_data = processed_data.astype(np.float32)
                        gdal_dtype = gdal.GDT_Float32
                        if nodata_value is not None:
                            nodata_value = float(nodata_value)

                    ds = driver.Create(
                        str(output_file),
                        cols,
                        rows,
                        1,
                        gdal_dtype,
                        options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER']
                    )

                    if ds is None:
                        raise RuntimeError(f"Failed to create {output_file}")

                    if geotransform is not None:
                        ds.SetGeoTransform(geotransform)

                    srs = osr.SpatialReference()
                    srs.ImportFromEPSG(epsg)
                    ds.SetProjection(srs.ExportToWkt())

                    band = ds.GetRasterBand(1)
                    if band is None:
                        raise RuntimeError(f"Failed to get raster band 1 from {output_file}")

                    write_result = band.WriteArray(processed_data)
                    if write_result != gdal.CE_None:
                        error_msg = gdal.GetLastErrorMsg()
                        raise RuntimeError(f"Failed to write array to {output_file}: {error_msg}")

                    if nodata_value is not None:
                        try:
                            if gdal_dtype in [gdal.GDT_Float32, gdal.GDT_Float64]:
                                if np.isnan(nodata_value):
                                    band.SetNoDataValue(float('nan'))
                                else:
                                    band.SetNoDataValue(float(nodata_value))
                            elif gdal_dtype in [gdal.GDT_Byte, gdal.GDT_UInt16, gdal.GDT_UInt32]:
                                band.SetNoDataValue(int(nodata_value))
                            else:
                                band.SetNoDataValue(int(nodata_value))
                            logger.info(f"Set nodata value to {nodata_value} for {geom_type}")
                        except Exception as e:
                            logger.warning(f"Failed to set nodata value for {geom_type}: {e}")

                    band.FlushCache()
                    band = None

                    if 'description' in dataset.attrs:
                        desc = dataset.attrs['description']
                        if isinstance(desc, bytes):
                            desc = desc.decode('utf-8')
                        ds.SetMetadataItem('DESCRIPTION', desc)

                    ds = None

                    extracted_files[geom_type]['file_list'] = output_file
                    extracted_files[geom_type]['nodata'] = nodata_value
                    logger.info(f"Extracted {geom_type} from {h5_file} (dataset: {dataset_path}, EPSG: {epsg}), shape: {data.shape}, dtype: {data_dtype}")

                except Exception as e:
                    logger.error(f"Error saving {geom_type} to GeoTIFF: {e}")
                    logger.error(traceback.format_exc())
                    if output_file.exists():
                        try:
                            output_file.unlink()
                        except Exception:
                            pass

    except Exception as e:
        logger.error(f"Error processing HDF5 file {h5_file}: {e}")
        logger.error(traceback.format_exc())

    return extracted_files


def merge_geometry_files(
    burst_ids: List[str],
    geom_dir: Path,
    output_dir: Path,
    geom_types: List[str],
    args: argparse.Namespace
) -> Dict[str, Path]:
    """Merge geometry files across multiple burst IDs."""
    output_dir = output_dir / "geom"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.keep_temp_files:
        temp_dir = output_dir / "temp_extracted"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir
    else:
        temp_dir = tempfile.TemporaryDirectory()
        temp_path = Path(temp_dir.name)

    try:
        geometry_files = {geom_type: {'file_list': [], 'nodata': None} for geom_type in geom_types}
        for burst_id in burst_ids:
            burst_main_dir = geom_dir / burst_id
            if not burst_main_dir.exists():
                logger.warning(f"Burst directory not found: {burst_main_dir}")
                continue

            h5_files = []
            for date_dir in sorted(burst_main_dir.iterdir()):
                if date_dir.is_dir():
                    found = list(date_dir.glob("static_layers*.h5"))
                    if found:
                        h5_files.append(found[0])
                        break

            if not h5_files:
                logger.warning(f"No static_layers HDF5 found for {burst_id} in {burst_main_dir}")
                continue

            if h5_files:
                for h5_file in h5_files:
                    extracted = extract_h5_geometry(
                        h5_file, temp_path, geom_types,
                        h5_group=args.h5_group,
                        dataset_mapping=load_h5_dataset_mapping(args.h5_dataset_mapping),
                        x_coords_name=args.h5_x_coords,
                        y_coords_name=args.h5_y_coords,
                        projection_name=args.h5_projection
                    )
                    for geom_type, data in extracted.items():
                        file_path = data['file_list']
                        if geometry_files[geom_type]['nodata'] is None:
                            geometry_files[geom_type]['nodata'] = data['nodata']
                        geometry_files[geom_type]['file_list'].append(file_path)
            else:
                for geom_type in geom_types:
                    geom_file = burst_main_dir / geom_type
                    if geom_file.exists():
                        geometry_files[geom_type]['file_list'].append(geom_file)
                    else:
                        logger.warning(f"Geometry file {geom_type} not found in {burst_main_dir}")

        merged_files = {}
        for geom_type, data in geometry_files.items():
            file_list = data['file_list']
            nodata = data['nodata']
            if not file_list:
                logger.warning(f"No files found for geometry type: {geom_type}")
                continue

            output_file = output_dir / geom_type

            if output_file.exists() and not args.overwrite:
                logger.info(f"{output_file} already exists, skipping")
                merged_files[geom_type] = output_file
                continue

            try:
                stitching_params = prepare_stitching_parameters_with_epsg_conversion(
                    file_list, args, "geometry", nodata
                )
                stitching_params = {k: v for k, v in stitching_params.items() if v is not None}

                merge_images(
                    file_list=file_list,
                    outfile=output_file,
                    **stitching_params
                )
                logger.info(f"Merged {len(file_list)} files for {geom_type} to {output_file}")
            except Exception as e:
                logger.error(f"Error merging {geom_type} files: {e}")
                continue

            merged_files[geom_type] = output_file

    finally:
        if not args.keep_temp_files and 'temp_dir' in locals() and isinstance(temp_dir, tempfile.TemporaryDirectory):
            temp_dir.cleanup()

    return merged_files


def merge_interferograms(
    burst_ids: List[str],
    intf_dir: Path,
    output_dir: Path,
    intf_types: List[str],
    args: argparse.Namespace,
    processor: str
) -> Dict[str, Path]:
    """Merge interferogram files across multiple burst IDs.

    Input structure (fixed): ``intf_dir/{burst_id}/ifgrams/{date1}_{date2}/xxx.ext``
    Output structure (fixed): ``output_dir/ifgrams/{date1}_{date2}/xxx.ext``
    """
    output_dir = output_dir / "ifgrams"
    output_dir.mkdir(parents=True, exist_ok=True)

    intf_files = defaultdict(lambda: defaultdict(list))

    # Expected extensions based on processor
    expected_exts = ['.int', '.unw', '.coh', '.conncomp'] if processor == 'isce2' else ['.int.tif', '.unw.tif', '.coh.tif', '.unw.conncomp.tif', '.conncomp.tif']

    for burst_id in burst_ids:
        burst_intf_dir = intf_dir / burst_id / "ifgrams"

        if not burst_intf_dir.exists():
            logger.warning(f"Interferogram directory not found for {burst_id}: {burst_intf_dir}")
            continue

        for intf_type in intf_types:
            pattern = f"**/*{intf_type}"
            for intf_file in sorted(burst_intf_dir.glob(pattern)):
                # Warn if filename doesn't end with a processor-expected extension
                if not any(intf_file.name.endswith(e) for e in expected_exts):
                    logger.warning(
                        f"Processor '{processor}' expects extensions {expected_exts}, "
                        f"but found '{intf_file.name}'"
                    )

                # date pair from the parent (date-pair) directory
                date_pair = intf_file.parent.name
                intf_files[intf_type][date_pair].append(intf_file)

    merged_files = {}
    for intf_type, date_files in intf_files.items():
        for date_pair, file_list in date_files.items():
            if not file_list:
                continue

            out_pair_dir = output_dir / date_pair
            out_pair_dir.mkdir(parents=True, exist_ok=True)
            output_file = out_pair_dir / Path(file_list[0]).name

            if output_file.exists() and not args.overwrite:
                logger.info(f"{output_file} already exists, skipping")
                merged_files[str(output_file)] = output_file
                continue

            try:
                stitching_params = prepare_stitching_parameters_with_epsg_conversion(
                    file_list, args, "interferogram"
                )
                stitching_params = {k: v for k, v in stitching_params.items() if v is not None}

                merge_images(
                    file_list=file_list,
                    outfile=output_file,
                    **stitching_params
                )
                logger.info(f"Merged {len(file_list)} {intf_type} files for {date_pair} to {output_file}")
                merged_files[str(output_file)] = output_file
            except Exception as e:
                logger.error(f"Error merging {intf_type} files for {date_pair}: {e}")

    return merged_files


def merge_slc_files(
    burst_ids: List[str],
    slc_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    processor: str
) -> Dict[str, Path]:
    """Merge SLC files across multiple burst IDs."""
    output_dir = output_dir / "slc"
    output_dir.mkdir(parents=True, exist_ok=True)

    slc_files = defaultdict(list)
    expected_ext = '.slc' if processor == 'isce2' else '.slc.tif'

    for burst_id in burst_ids:
        burst_slc_dir = slc_dir / burst_id / "slc"

        if not burst_slc_dir.exists():
            logger.warning(f"SLC directory not found for {burst_id}: {burst_slc_dir}")
            continue

        pattern = "*.slc.tif" if processor == 'isce3' else "*.slc"
        for slc_file in burst_slc_dir.glob(pattern):
            ext = slc_file.suffix
            if ext != expected_ext:
                logger.warning(
                    f"Processor '{processor}' expects extension '{expected_ext}', "
                    f"but found '{ext}' for file {slc_file.name}"
                )

            filename = slc_file.name
            date_str = filename.split('.')[0]
            slc_files[date_str].append(slc_file)

    merged_files = {}
    for date_str, file_list in slc_files.items():
        if not file_list:
            continue

        output_file = output_dir / f"{date_str}{expected_ext}"

        if output_file.exists() and not args.overwrite:
            logger.info(f"{output_file} already exists, skipping")
            merged_files[date_str] = output_file
            continue

        try:
            stitching_params = prepare_stitching_parameters_with_epsg_conversion(
                file_list, args, "slc"
            )
            stitching_params = {k: v for k, v in stitching_params.items() if v is not None}

            merge_images(
                file_list=file_list,
                outfile=output_file,
                **stitching_params
            )
            logger.info(f"Merged {len(file_list)} SLC files for {date_str} to {output_file}")
            merged_files[date_str] = output_file
        except Exception as e:
            logger.error(f"Error merging SLC files for {date_str}: {e}")

    return merged_files



def discover_burst_ids_from_dirs(*dirs) -> list:
    """Auto-discover burst IDs from subdirectory names in given directories.

    Scans each directory for subdirectories whose name matches the burst ID
    pattern ``t<track>_<burst>_iw<swath>`` (e.g. ``t124_264305_iw2``).

    Parameters
    ----------
    *dirs : str or Path
        One or more base directories to scan.

    Returns
    -------
    list of str
        Sorted, deduplicated list of burst ID strings.
    """
    burst_id_pattern = re.compile(r'^t\d+_\d+_iw\d+$')
    all_burst_ids = set()

    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for entry in d.iterdir():
            if entry.is_dir() and burst_id_pattern.match(entry.name):
                all_burst_ids.add(entry.name)

    return sorted(all_burst_ids)



def main(args: Optional[argparse.Namespace] = None) -> int:
    """Main function to coordinate merging of different data types."""
    if args is None:
        args = parse_arguments()

    # Convert paths to Path objects
    baseline_dir = Path(args.baseline_dir)
    intf_dir = Path(args.intf_dir)
    slc_dir = Path(args.slc_dir)
    geom_dir = Path(args.geom_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    burst_ids = args.burst_ids
    if burst_ids is None:
        logger.info("No burst IDs specified, auto-detecting from directory structure...")
        burst_ids = discover_burst_ids_from_dirs(
            baseline_dir, intf_dir, slc_dir, geom_dir
        )
        if not burst_ids:
            logger.error("No burst IDs found. Ensure at least one of --baseline-dir, --intf-dir, --slc-dir, or --geom-dir contains burst ID subdirectories.")
            return 1
        logger.info(f"Auto-detected {len(burst_ids)} burst IDs: {burst_ids}")

    logger.info(f"Processor: {args.processor}")
    logger.info(f"Processing burst IDs: {burst_ids}")
    logger.info(f"Modes: {args.mode}")
    logger.info(f"Output directory: {output_dir}")

    if args.out_bounds:
        logger.info(f"Input bounds (EPSG:{args.out_bounds_epsg}): {args.out_bounds}")

    results = {}

    if "baseline" in args.mode:
        logger.info("Merging baseline files...")
        baseline_results = merge_baselines(
            burst_ids, baseline_dir, output_dir, args.overwrite
        )
        results["baselines"] = baseline_results
        logger.info(f"Merged {len(baseline_results)} baseline date pairs")

    if "intf" in args.mode:
        logger.info("Merging interferogram files...")
        intf_results = merge_interferograms(
            burst_ids, intf_dir, output_dir,
            args.intf_types, args, args.processor
        )
        results["interferograms"] = intf_results
        logger.info(f"Merged {len(intf_results)} interferogram files")

    if "slc" in args.mode:
        logger.info("Merging SLC files...")
        slc_results = merge_slc_files(
            burst_ids, slc_dir, output_dir,
            args, args.processor
        )
        results["slc"] = slc_results
        logger.info(f"Merged {len(slc_results)} SLC dates")

    if "geometry" in args.mode:
        logger.info("Merging geometry files...")
        geom_results = merge_geometry_files(
            burst_ids, geom_dir, output_dir,
            args.geom_types, args
        )
        results["geometry"] = geom_results
        logger.info(f"Merged {len(geom_results)} geometry files")

    logger.info(f"All processing complete. Results saved to {output_dir}")

    # Print summary via logger
    summary_lines = ["=" * 50, "MERGE PROCESSING SUMMARY", "=" * 50]
    for line in summary_lines:
        logger.info(line)

    for mode, mode_results in results.items():
        logger.info(f"\n{mode.upper()}:")
        if isinstance(mode_results, dict):
            for key, value in mode_results.items():
                if isinstance(value, Path):
                    logger.info(f"  {key}: {value}")
                else:
                    logger.info(f"  {key}: {len(value) if isinstance(value, list) else value}")
        else:
            logger.info(f"  Results: {mode_results}")

    return 0


if __name__ == "__main__":
    sys.exit(main())