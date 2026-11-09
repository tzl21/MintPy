import glob
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path


# Ensure PROJ/GDAL data paths are set when running without conda activation,
# otherwise osr.ImportFromEPSG() fails with "proj_create_from_database" errors.
# This must run BEFORE "from osgeo import gdal, osr" so that the PROJ context
# is initialized with the correct data directory.
def _setup_gdal_proj_data():
    import sys
    for env_key, subdir, probe in [('PROJ_DATA', 'share/proj', 'proj.db'),
                                   ('GDAL_DATA', 'share/gdal', None)]:
        if env_key in os.environ or (env_key == 'PROJ_DATA' and 'PROJ_LIB' in os.environ):
            continue
        data_dir = os.path.join(sys.prefix, *subdir.split('/'))
        if os.path.isdir(data_dir) and (probe is None or os.path.isfile(os.path.join(data_dir, probe))):
            os.environ[env_key] = data_dir

_setup_gdal_proj_data()

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from mintpy.constants import (  # noqa: E402
    GEOMETRY_FILENAMES as GEOMETRY_FILENAMES,
)
from mintpy.objects import sensor  # noqa: E402
from mintpy.utils import readfile  # noqa: E402

try:
    from scipy.interpolate import CubicHermiteSpline
except ImportError:
    CubicHermiteSpline = None
    print("Warning: scipy not available. Orbit interpolation will fall back to linear.")


# Default geometry layers are defined in mintpy.constants (single source of
# truth), so that the CLI can use them without importing the GDAL/h5py stack.

GEOMETRY_DSET_MAPPING = {
    "height.tif": "z",
    "layover_shadow_mask.tif": "layover_shadow_mask",
    "local_incidence_angle.tif": "local_incidence_angle",
    "los_east.tif": "los_east",
    "los_north.tif": "los_north",
}


def extract_isce3_metadata(meta_file: str) -> dict:
    """Extract common metadata from an ISCE3/Dolphin burst XML file.

    Parameters
    ----------
    meta_file : str
        Path to the reference burst XML file.

    Returns
    -------
    dict
        Common metadata dictionary with keys required by MintPy.

    """
    # NOTE: the XML file is a local, MintPy/ISCE3-generated metadata file.
    # xml.etree.ElementTree does not resolve external entities, but still
    # expands internal entities, so do not feed it untrusted input.
    tree = ET.parse(meta_file)
    root = tree.getroot()
    burst_elem = root.find('burst_attributes')
    if burst_elem is None:
        raise ValueError(f'Missing <burst_attributes> in {meta_file}')

    # Helper to extract text from element
    def get_value(tag):
        elem = burst_elem.find(tag)
        if elem is not None:
            return (elem.text or '').strip()
        return None

    meta = {}

    # Basic radar parameters
    meta['prf'] = get_value('prf')
    meta['startUTC'] = get_value('burstStartUTC')
    meta['stopUTC'] = get_value('burstStopUTC')
    meta['radarWavelength'] = get_value('radarWavelength')
    meta['startingRange'] = get_value('startingRange')
    meta['passDirection'] = get_value('passDirection')
    meta['polarization'] = get_value('polarization')
    meta['trackNumber'] = get_value('trackNumber')
    meta['orbitNumber'] = get_value('orbitNumber')

    # Platform name (Sentinel-1)
    meta['PLATFORM'] = 'sen'

    # Sensing mid time and center line UTC
    sensing_mid = get_value('sensingMid')
    if sensing_mid:
        try:
            dt = datetime.strptime(sensing_mid, '%Y-%m-%d %H:%M:%S.%f')
            seconds_of_day = dt.hour * 3600.0 + dt.minute * 60.0 + dt.second + dt.microsecond / 1e6
            meta['CENTER_LINE_UTC'] = str(seconds_of_day)
        except (ValueError, TypeError):
            meta['CENTER_LINE_UTC'] = '0'
    else:
        meta['CENTER_LINE_UTC'] = '0'

    # Pixel sizes
    az_time_interval = get_value('azimuthTimeInterval')
    range_pixel_size = get_value('rangePixelSize')
    satellite_speed = get_value('satelliteSpeed')

    if az_time_interval and satellite_speed:
        try:
            az_pixel_size = float(satellite_speed) * float(az_time_interval)
            meta['azimuthPixelSize'] = str(az_pixel_size)
        except ValueError:
            meta['azimuthPixelSize'] = '0.0'
    else:
        meta['azimuthPixelSize'] = '0.0'

    if range_pixel_size:
        meta['rangePixelSize'] = range_pixel_size
    else:
        meta['rangePixelSize'] = '0.0'

    # Spatial resolution from sensor database (Sentinel-1 IW)
    # Determine swath (e.g., 'IW2')
    swath_num = get_value('swathNumber')
    if swath_num:
        iw_str = f'IW{swath_num}'
    else:
        # fallback: try to infer from filename
        base = os.path.basename(meta_file)
        if base.startswith('IW'):
            iw_str = base.split('.')[0]
        else:
            iw_str = 'IW2'
    try:
        meta['azimuthResolution'] = str(sensor.SENSOR_DICT['sen'][iw_str]['azimuth_resolution'])
        meta['rangeResolution']   = str(sensor.SENSOR_DICT['sen'][iw_str]['range_resolution'])
    except KeyError:
        meta['azimuthResolution'] = '0.0'
        meta['rangeResolution']   = '0.0'

    # Heading, earth radius, altitude
    meta['HEADING'] = get_value('HEADING')
    meta['earthRadius'] = get_value('earthRadius')
    meta['altitude'] = get_value('altitude')

    # Beam mode and swath
    meta['beam_mode'] = 'IW'
    meta['swathNumber'] = swath_num if swath_num else '2'

    # Frame numbers (may be 0 for ISCE3 products)
    meta['firstFrameNumber'] = get_value('firstFrameNumber') or '0'
    meta['lastFrameNumber'] = get_value('lastFrameNumber') or '0'

    # Antenna side (default to -1, right-looking)
    meta['ANTENNA_SIDE'] = '-1'

    # Processor
    meta['PROCESSOR'] = 'isce3'

    # Height / Earth radius (map from lowercase keys used in the XML)
    if meta.get('altitude') and not meta.get('HEIGHT'):
        meta['HEIGHT'] = meta['altitude']
    if meta.get('earthRadius') and not meta.get('EARTH_RADIUS'):
        meta['EARTH_RADIUS'] = meta['earthRadius']

    # Compute center incidence angle if possible.
    # NOTE: the old formula (look_angle = asin(R/(R+H)) followed by
    # asin((R+H)/R * sin(look_angle))) always collapses to exactly 90 degrees.
    # Use the standard law-of-cosines solution in the triangle formed by the
    # satellite, the target and the earth center, i.e. the same convention as
    # ut.incidence_angle(atr, dimension=0) in utils0.py (measured from the
    # local vertical; near-/mid-range slant range).
    if meta.get('HEIGHT') and meta.get('EARTH_RADIUS') and meta.get('startingRange'):
        try:
            H = float(meta['HEIGHT'])
            R = float(meta['EARTH_RADIUS'])
            rho = float(meta['startingRange'])
            # use the mid-range slant range when the scene width is known
            if meta.get('WIDTH') and meta.get('rangePixelSize'):
                rho += float(meta['WIDTH']) / 2.0 * float(meta['rangePixelSize'])
            cos_inc = ((R + H) ** 2 - R ** 2 - rho ** 2) / (2.0 * R * rho)
            inc_angle = np.rad2deg(np.arccos(np.clip(cos_inc, -1.0, 1.0)))
            meta['CENTER_INCIDENCE_ANGLE'] = str(inc_angle)
        except (ValueError, TypeError):
            pass

    # Looks will be computed later from burst vs interferogram dimensions
    meta['ALOOKS'] = '1'
    meta['RLOOKS'] = '1'

    # Convert all values to strings (MintPy standard)
    for key, value in meta.items():
        if value is not None:
            meta[key] = str(value)
        else:
            meta[key] = ''

    # Standardize keys
    meta = readfile.standardize_metadata(meta)

    # Note: LENGTH and WIDTH are not in this XML; they will be added later
    # from geometry or interferogram files.

    return meta


#################################################################
# baseline computation / reading (ISCE3 / OPERA CSLC, multi-burst)
#################################################################
BASELINE_TXT_GLOB = '[0-9]*_[0-9]*.txt'
CSLC_BURST_ID_PATTERN = r'(t\d{3}_\d{6}_iw\d)'


def is_auto_value(value):
    """Check if a template value is unset/auto."""
    return value is None or str(value).lower() in ['auto', 'none', 'no', '']


def get_burst_id_from_path(path):
    """Get the Sentinel-1 burst ID (e.g. t124_264305_iw2) from a file/dir path."""
    for part in reversed(Path(str(path)).parts):
        match = re.search(CSLC_BURST_ID_PATTERN, part)
        if match:
            return match.group(1)
    return None


def scan_cslc_h5_files(cslc_input):
    """Find the OPERA CSLC HDF5 files matching a path, directory or glob pattern.

    Parameters: cslc_input - str or list of str, glob pattern (e.g.
                             "../../CSLC/t124*/2*/t124*.h5"), directory, or paths
    Returns:    list of str, sorted CSLC HDF5 file paths
    """
    if isinstance(cslc_input, (list, tuple)):
        files = []
        for item in cslc_input:
            files += scan_cslc_h5_files(item)
        return sorted(set(files))

    pattern = str(cslc_input)
    if os.path.isdir(pattern):
        candidates = sorted(glob.glob(os.path.join(pattern, '**', '*.h5'), recursive=True))
    else:
        candidates = sorted(glob.glob(pattern))

    return [fname for fname in candidates
            if os.path.isfile(fname)
            and fname.endswith('.h5')
            and not os.path.basename(fname).startswith('static_layers')
            and get_burst_id_from_path(fname) is not None]


def group_cslc_files_by_burst(cslc_files):
    """Group CSLC HDF5 files by burst ID and acquisition date.

    Parameters: cslc_files - list of str, CSLC HDF5 file paths
    Returns:    dict, burst_id -> {'dir': str, 'dates': {date: h5_file}}
    """
    bursts = defaultdict(lambda: {'dirs': set(), 'dates': {}})
    for fname in cslc_files:
        burst_id = get_burst_id_from_path(fname)
        if burst_id is None:
            continue

        path = Path(fname)
        date = None
        for part in reversed(path.parts[:-1]):
            if len(part) == 8 and part.isdigit():
                date = part
                break
        if date is None:
            match = re.search(r'_(\d{8})(?:_|\.)', path.name)
            date = match.group(1) if match else None
        if date is None:
            continue

        parts = list(path.parts)
        if burst_id in parts:
            idx = parts.index(burst_id)
            burst_dir = str(Path(*parts[:idx + 1]))
        else:
            burst_dir = str(path.parent.parent) if path.parent.name == date else str(path.parent)

        bursts[burst_id]['dirs'].add(burst_dir)
        bursts[burst_id]['dates'][date] = str(path)

    return {bid: {'dir': sorted(info['dirs'])[0], 'dates': info['dates']}
            for bid, info in bursts.items()}


def is_cslc_baseline_input(baseline_dir):
    """Check if mintpy.load.baselineDir contains CSLC HDF5 products (to be computed)."""
    return len(scan_cslc_h5_files(baseline_dir)) > 0


def get_burst_ids_from_cslc(cslc_input):
    """Get the sorted burst IDs from the CSLC files matched by the input pattern."""
    return sorted(group_cslc_files_by_burst(scan_cslc_h5_files(cslc_input)))


def get_burst_ids_from_geom_dirs(geom_dirs):
    """Get the sorted burst IDs from geometry dirs containing static_layers*.h5 files."""
    burst_ids = set()
    if isinstance(geom_dirs, str):
        geom_dirs = [geom_dirs]

    for geom_dir in geom_dirs or []:
        if not os.path.isdir(geom_dir):
            continue
        if glob.glob(os.path.join(geom_dir, 'static_layers*.h5')):
            burst_ids.add(get_burst_id_from_path(geom_dir))
            continue
        for sub_dir in sorted(glob.glob(os.path.join(geom_dir, '*/'))):
            if glob.glob(os.path.join(sub_dir, 'static_layers*.h5')):
                burst_ids.add(get_burst_id_from_path(sub_dir))
                continue
            # burst dir with one subdir per date, e.g. <burst>/<YYYYMMDD>/static_layers.h5
            for date_dir in sorted(glob.glob(os.path.join(sub_dir, '*/'))):
                if glob.glob(os.path.join(date_dir, 'static_layers*.h5')):
                    burst_ids.add(get_burst_id_from_path(date_dir))
                    break

    return sorted(bid for bid in burst_ids if bid)


def _load_orbit_from_cslc(h5_file):
    """Load the isce3 Orbit and reference epoch datetime from a CSLC HDF5 file."""
    from isce3.core import DateTime, Orbit, StateVector

    with h5py.File(h5_file, 'r') as f:
        orbit_ds = f['metadata/orbit']
        times = orbit_ds['time'][:]
        px = orbit_ds['position_x'][:]
        py = orbit_ds['position_y'][:]
        pz = orbit_ds['position_z'][:]
        vx = orbit_ds['velocity_x'][:]
        vy = orbit_ds['velocity_y'][:]
        vz = orbit_ds['velocity_z'][:]
        ref_str = orbit_ds['reference_epoch'][()].decode('utf-8')

    ref_dt = datetime.strptime(ref_str, '%Y-%m-%d %H:%M:%S.%f')
    ref_epoch = DateTime(ref_dt.strftime('%Y-%m-%dT%H:%M:%S.%f'))

    state_vectors = []
    for i in range(len(times)):
        sv_dt = ref_dt + timedelta(seconds=float(times[i]))
        state_vectors.append(StateVector(
            DateTime(sv_dt.strftime('%Y-%m-%dT%H:%M:%S.%f')),
            [px[i], py[i], pz[i]],
            [vx[i], vy[i], vz[i]],
        ))

    return Orbit(state_vectors, ref_epoch), ref_dt


def _get_cslc_center_coords(h5_file, ref_dt):
    """Get the center azimuth time / slant range and the wavelength from a CSLC HDF5."""
    with h5py.File(h5_file, 'r') as f:
        burst_meta = f['metadata/processing_information/input_burst_metadata']
        sensing_start = datetime.strptime(
            burst_meta['sensing_start'][()].decode().strip(), '%Y-%m-%d %H:%M:%S.%f')
        start_sec = (sensing_start - ref_dt).total_seconds()
        shape = burst_meta['shape'][:]
        num_az, num_rg = int(shape[0]), int(shape[1])
        az_interval = float(burst_meta['azimuth_time_interval'][()])
        range_spacing = float(burst_meta['range_pixel_spacing'][()])
        range_start = float(burst_meta['starting_range'][()])
        wavelength = float(burst_meta['wavelength'][()])

    az_mid = start_sec + (num_az // 2) * az_interval
    range_mid = range_start + (num_rg // 2) * range_spacing
    return az_mid, range_mid, wavelength


def _prepare_baseline_reference(h5_ref):
    """Prepare the reference target position for baseline computation (per burst)."""
    from isce3.core import Ellipsoid, LookSide
    from isce3.geometry import rdr2geo

    ref_orbit, ref_dt = _load_orbit_from_cslc(h5_ref)
    az_mid, range_mid, wavelength = _get_cslc_center_coords(h5_ref, ref_dt)

    # zero-Doppler geometry, matching the OPERA CSLC burst grid
    lon, lat, height = rdr2geo(
        az_mid, range_mid, ref_orbit, LookSide.Right,
        doppler=0.0, wavelength=wavelength,
    )
    target_xyz = np.array(Ellipsoid().lon_lat_to_xyz(
        [float(lon), float(lat), float(height)]))

    return {
        'orbit': ref_orbit,
        'az_mid': az_mid,
        'wavelength': wavelength,
        'target_xyz': target_xyz,
    }


def _compute_baseline_pair(ref_cache, h5_secondary):
    """Compute (B_par, B_perp) in meters for one reference/secondary CSLC pair."""
    from isce3.core import LookSide, LUT2d
    from isce3.geometry import geo2rdr_bracket

    sec_orbit, _ = _load_orbit_from_cslc(h5_secondary)
    secondary_time, _ = geo2rdr_bracket(
        ref_cache['target_xyz'], sec_orbit, LUT2d(),
        ref_cache['wavelength'], LookSide.Right)

    ref_pos, ref_vel = ref_cache['orbit'].interpolate(ref_cache['az_mid'])
    ref_pos = np.array(ref_pos)
    ref_vel = np.array(ref_vel)
    sec_pos = np.array(sec_orbit.interpolate(float(secondary_time))[0])

    line_of_sight = ref_cache['target_xyz'] - ref_pos
    los_unit = line_of_sight / np.linalg.norm(line_of_sight)
    baseline = sec_pos - ref_pos

    b_par = float(np.dot(baseline, los_unit))
    b_perp = float(np.sqrt(max(0.0, np.linalg.norm(baseline) ** 2 - b_par ** 2)))
    b_perp *= np.sign(np.dot(np.cross(line_of_sight, baseline), ref_vel))
    return b_par, b_perp


def compute_baselines_isce3(cslc_input, out_dir=None, update_mode=True, print_msg=True):
    """Compute the baseline time series from OPERA CSLC HDF5 products.

    For each burst, the earliest date is used as the reference and one
    YYYYMMDD_YYYYMMDD.txt file is written per date pair, with the same content
    format as the ISCE2 topsStack / Dolphin baselines:
        Bperp (m): <value>
        Bpar (m): <value>

    Parameters: cslc_input  - str or list of str, CSLC glob pattern / directory / files,
                              e.g. "../../CSLC/t124*/2*/t124*.h5"
                out_dir     - str, output baselines directory; default is
                              <parent of the burst dir>/baselines/<burst_id>/
                update_mode - bool, skip the date pairs whose .txt already exists
                print_msg   - bool, print progress
    Returns:    list of str, the baselines parent directory(/ies) to be read
    """
    cslc_files = scan_cslc_h5_files(cslc_input)
    if not cslc_files:
        raise FileNotFoundError(f'no CSLC HDF5 file found for: {cslc_input}')

    bursts = group_cslc_files_by_burst(cslc_files)
    if not bursts:
        raise ValueError(f'no burst ID / acquisition date recognized from: {cslc_input}')

    baseline_dirs = []
    for burst_id in sorted(bursts):
        dates = bursts[burst_id]['dates']
        if len(dates) < 2:
            print(f'WARNING: need >= 2 dates to compute baselines, got {len(dates)} '
                  f'for burst {burst_id}')
            continue

        date_list = sorted(dates)
        ref_date = date_list[0]
        parent_dir = os.path.dirname(bursts[burst_id]['dir'])
        burst_out_dir = (os.path.join(out_dir, burst_id) if out_dir
                         else os.path.join(parent_dir, 'baselines', burst_id))
        os.makedirs(burst_out_dir, exist_ok=True)
        baseline_dirs.append(os.path.dirname(burst_out_dir))

        pair_list = [(ref_date, date2) for date2 in date_list[1:]]
        pairs2compute = [(d1, d2) for d1, d2 in pair_list
                         if not (update_mode
                                 and os.path.isfile(os.path.join(burst_out_dir, f'{d1}_{d2}.txt')))]
        if print_msg:
            print(f'computing baselines for burst {burst_id}: {len(date_list)} dates, '
                  f'{len(pairs2compute)} / {len(pair_list)} pairs to compute')
        if len(pairs2compute) == 0:
            continue

        ref_cache = _prepare_baseline_reference(dates[ref_date])
        for d1, d2 in pairs2compute:
            b_par, b_perp = _compute_baseline_pair(ref_cache, dates[d2])
            out_file = os.path.join(burst_out_dir, f'{d1}_{d2}.txt')
            with open(out_file, 'w') as f:
                f.write(f'Bperp (m): {b_perp}\n')
                f.write(f'Bpar (m): {b_par}\n')
            if print_msg:
                print(f'  {d1}-{d2}: Bpar={b_par:+9.3f} m, Bperp={b_perp:+9.3f} m')

    return sorted(set(baseline_dirs))


def _read_baseline_burst(txt_files):
    """Read the baselines of a single burst from its YYYYMMDD_YYYYMMDD.txt files."""
    pairs = []
    for fname in sorted(txt_files):
        date_pair = os.path.basename(fname)[:-4].split('_')
        if len(date_pair) != 2:
            continue
        date1, date2 = date_pair
        if len(date1) != 8 or len(date2) != 8 or not date1.isdigit() or not date2.isdigit():
            continue

        bperp, bpar = None, None
        with open(fname) as f:
            for line in f:
                line = line.strip()
                match = re.match(r'[Bb]perp\b.*?:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
                if match:
                    bperp = float(match.group(1))
                match = re.match(r'[Bb]par\b.*?:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
                if match:
                    bpar = float(match.group(1))
        if bperp is None:
            continue
        pairs.append((date1, date2, bperp, bpar))

    if not pairs:
        return None

    ref_date = Counter(pair[0] for pair in pairs).most_common(1)[0][0]
    # keep only the pairs that are relative to the common reference date
    dates = {}
    for date1, date2, bperp, bpar in pairs:
        if date1 == ref_date:
            dates[date2] = [bperp, bpar if bpar is not None else np.nan]
    return {'ref_date': ref_date, 'dates': dates}


def _collect_baseline_burst_files(baseline_dir):
    """Collect the baseline .txt files per burst from a directory or glob pattern."""
    if isinstance(baseline_dir, (list, tuple)):
        paths = [str(item) for item in baseline_dir]
    else:
        paths = sorted(glob.glob(str(baseline_dir))) or [str(baseline_dir)]

    burst_files = {}
    for path in paths:
        if not os.path.isdir(path):
            continue
        txt_files = sorted(glob.glob(os.path.join(path, BASELINE_TXT_GLOB)))
        if txt_files:
            burst_id = get_burst_id_from_path(path) or os.path.basename(os.path.abspath(path))
            burst_files.setdefault(burst_id, []).extend(txt_files)
            continue
        for sub_dir in sorted(glob.glob(os.path.join(path, '*/'))):
            if not os.path.isdir(sub_dir):
                continue
            txt_files = sorted(glob.glob(os.path.join(sub_dir, BASELINE_TXT_GLOB)))
            if txt_files:
                burst_id = (get_burst_id_from_path(sub_dir)
                            or os.path.basename(os.path.abspath(sub_dir)))
                burst_files.setdefault(burst_id, []).extend(txt_files)

    if not burst_files:
        # last-resort recursive search for deeply nested layouts
        for path in paths:
            if not os.path.isdir(path):
                continue
            for fname in sorted(glob.glob(os.path.join(path, '**', BASELINE_TXT_GLOB),
                                          recursive=True)):
                burst_id = get_burst_id_from_path(fname) or 'default'
                burst_files.setdefault(burst_id, []).append(fname)

    return burst_files


def read_baseline_timeseries_isce3(baseline_dir, burst_ids=None):
    """Read the baseline time series from an ISCE3/Dolphin (multi-burst) baselines dir.

    Accepted layouts, either for one or multiple burst IDs:
        <baseline_dir>/<burst-id>/YYYYMMDD_YYYYMMDD.txt
        <baseline_dir>/YYYYMMDD_YYYYMMDD.txt
    Each .txt file contains one `Bperp (m):` line (and optionally `Bpar (m):`).

    For multiple burst IDs, the burst with the most dates is used as the
    reference: if the other bursts have the same date list, their baselines are
    averaged; otherwise a warning is printed and only the reference burst is
    used. If `burst_ids` is given, only those bursts are read, so that the
    baselines always come from the same frames as the CSLC stack.

    Parameters: baseline_dir - str or list of str, baselines directory / glob pattern
                burst_ids    - list of str, optional, expected CSLC burst IDs
    Returns:    dict, date (YYYYMMDD) -> [Bperp_top, Bperp_bottom] in meters
    """
    baseline_dict = {}
    if is_auto_value(baseline_dir):
        return baseline_dict

    burst_files = _collect_baseline_burst_files(baseline_dir)
    if not burst_files:
        print(f'WARNING: no baseline text files found in {baseline_dir}')
        return baseline_dict

    # keep the baselines of the same frames as the CSLC stack
    if burst_ids:
        expected = set(burst_ids)
        missing = sorted(expected - set(burst_files))
        extra = sorted(set(burst_files) - expected)
        if missing:
            print(f'WARNING: no baselines found for burst ID(s): {missing}')
        if extra:
            print(f'WARNING: ignore baselines of burst ID(s) not in the CSLC stack: {extra}')
        burst_files = {bid: files for bid, files in burst_files.items() if bid in expected}
        if not burst_files:
            print('WARNING: no baseline file matches the CSLC burst ID(s), skip baselines')
            return baseline_dict

    burst_dict = {}
    for burst_id, txt_files in sorted(burst_files.items()):
        parsed = _read_baseline_burst(txt_files)
        if parsed:
            burst_dict[burst_id] = parsed
    if not burst_dict:
        return baseline_dict

    # burst with the most dates is the reference
    ref_burst = max(burst_dict, key=lambda bid: (len(burst_dict[bid]['dates']), bid))
    ref_info = burst_dict[ref_burst]

    inconsistent = []
    for burst_id, info in burst_dict.items():
        if burst_id == ref_burst:
            continue
        if (info['ref_date'] != ref_info['ref_date']
                or set(info['dates']) != set(ref_info['dates'])):
            inconsistent.append(burst_id)

    if inconsistent:
        print(f'WARNING: baselines of burst ID(s) {sorted(inconsistent)} have a different '
              f'date/ref list than burst {ref_burst}; using only burst {ref_burst}')
        for date2, (bperp, _) in ref_info['dates'].items():
            baseline_dict[date2] = [bperp, bperp]
    else:
        for date2 in ref_info['dates']:
            bperp = float(np.mean([burst_dict[bid]['dates'][date2][0]
                                   for bid in burst_dict]))
            baseline_dict[date2] = [bperp, bperp]
        if len(burst_dict) > 1:
            print(f'averaged baselines over {len(burst_dict)} burst ID(s): '
                  f'{sorted(burst_dict)}')

    baseline_dict[ref_info['ref_date']] = [0.0, 0.0]
    return baseline_dict
