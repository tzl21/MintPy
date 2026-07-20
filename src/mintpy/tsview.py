#!/usr/bin/env python3
#############################################################
# Program is part of MintPy                                 #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi          #
# Author: Zhang Yunjun, Joshua Zahner, Heresh Fattahi, 2013 #
#############################################################

import argparse
import os
import re

import numpy as np
from matplotlib import patches, pyplot as plt, widgets, dates as mdates
from scipy import linalg, stats

from mintpy import subset, timeseries2velocity as ts2vel, view
from mintpy.multilook import multilook_data
from mintpy.objects import HDFEOS, giantTimeseries, timeseries, gnss as gnss_mod
from mintpy.utils import plot as pp, ptime, readfile, time_func, utils as ut

# Module-level default for vprint, so standalone function calls
# (e.g., read_init_info, read_timeseries_data) work without
# going through timeseriesViewer.open().
vprint = print


###########################################################################################
def read_init_info(inps):
    """Read and initialize metadata and parameters.

    Parameters
    ----------
    inps : Namespace
        Input arguments from the command line parser.

    Returns
    -------
    inps : Namespace
        Updated input arguments with additional attributes.
    atr : dict
        Metadata dictionary of the first input file.
    """

    # Time Series Info
    atr = readfile.read_attribute(inps.file[0])
    atr['DATA_TYPE'] = atr.get('DATA_TYPE', 'float32')

    inps.key = atr['FILE_TYPE']
    if inps.key == 'timeseries':
        obj = timeseries(inps.file[0])
    elif inps.key == 'giantTimeseries':
        obj = giantTimeseries(inps.file[0])
    elif inps.key == 'HDFEOS':
        obj = HDFEOS(inps.file[0])
    else:
        raise ValueError(f'input file is {inps.key}, not timeseries.')
    obj.open(print_msg=inps.print_msg)
    inps.seconds = atr.get('CENTER_LINE_UTC', 0)

    if not inps.file_label:
        inps.file_label = []
        for fname in inps.file:
            fbase = os.path.splitext(os.path.basename(fname))[0]
            fbase = fbase.replace('timeseries', 'TS')
            inps.file_label.append(fbase)

    # default mask file
    if not inps.mask_file and 'msk' not in inps.file[0]:
        dir_name = os.path.dirname(inps.file[0])
        if 'Y_FIRST' in atr.keys():
            inps.mask_file = os.path.join(dir_name, 'geo_maskTempCoh.h5')
        else:
            inps.mask_file = os.path.join(dir_name, 'maskTempCoh.h5')
        if not os.path.isfile(inps.mask_file):
            inps.mask_file = None

    ## date info
    inps.date_list = obj.dateList
    inps.num_date = len(inps.date_list)
    if inps.start_date:
        inps.date_list = [i for i in inps.date_list if int(i) >= int(inps.start_date)]
    if inps.end_date:
        inps.date_list = [i for i in inps.date_list if int(i) <= int(inps.end_date)]
    inps.num_date = len(inps.date_list)
    inps.dates, inps.yearList = ptime.date_list2vector(inps.date_list)

    (inps.ex_date_list,
     inps.ex_dates,
     inps.ex_flag) = read_exclude_date(inps.ex_date_list, inps.date_list)

    # reference date/index
    if not inps.ref_date:
        inps.ref_date = atr.get('REF_DATE', None)
    if inps.ref_date:
        inps.ref_idx = inps.date_list.index(inps.ref_date)
    else:
        inps.ref_idx = None

    # date/index of interest for initial display
    if not inps.idx:
        if (not inps.ref_idx) or (inps.ref_idx < inps.num_date / 2.):
            inps.idx = inps.num_date - 2
        else:
            inps.idx = 2

    # Display Unit
    inps.disp_unit, inps.unit_fac = pp.scale_data2disp_unit(
        metadata=atr, disp_unit=inps.disp_unit)[1:3]

    # Read Error List
    inps.ts_plot_func = plot_ts_scatter
    inps.error_ts = None
    inps.ex_error_ts = None
    if inps.error_file:
        # assign plot function
        inps.ts_plot_func = plot_ts_errorbar

        # read error file
        error_fc = np.loadtxt(inps.error_file, dtype=bytes).astype(str)
        inps.error_ts = error_fc[:, 1].astype(np.float32)*inps.unit_fac

        # update error file with exclude date
        if inps.ex_date_list:
            e_ts = inps.error_ts[:]
            inps.ex_error_ts = e_ts[inps.ex_flag == 0]
            inps.error_ts = e_ts[inps.ex_flag == 1]

    # Zero displacement for 1st acquisition
    if inps.zero_first:
        inps.zero_idx = min(0, np.min(np.where(inps.ex_flag)[0]))

    # default lookup table file and coordinate object
    if not inps.lookup_file:
        inps.lookup_file = ut.get_lookup_file('./inputs/geometryRadar.h5')
    inps.coord = ut.coordinate(atr, inps.lookup_file)

    ## size and lalo info
    inps.pix_box, inps.geo_box = subset.subset_input_dict2box(vars(inps), atr)
    inps.pix_box = inps.coord.check_box_within_data_coverage(inps.pix_box)
    inps.geo_box = inps.coord.box_pixel2geo(inps.pix_box)
    data_box = (0, 0, int(atr['WIDTH']), int(atr['LENGTH']))
    vprint('data   coverage in y/x: '+str(data_box))
    vprint('subset coverage in y/x: '+str(inps.pix_box))
    vprint('data   coverage in lat/lon: '+str(inps.coord.box_pixel2geo(data_box)))
    vprint('subset coverage in lat/lon: '+str(inps.geo_box))
    vprint('------------------------------------------------------------------------')

    # Map info - coordinate unit
    inps.coord_unit = atr.get('Y_UNIT', 'degrees').lower()
    inps.lalo_digit = ut.get_lalo_digit4display(atr, coord_unit=inps.coord_unit)
    inps = view.check_map_projection(inps, metadata=atr, print_msg=inps.print_msg)

    # calculate multilook_num
    # ONLY IF:
    #   inps.multilook is True (no --nomultilook input) AND
    #   inps.multilook_num ==1 (no --multilook-num input)
    # Note: inps.multilook is used for this check ONLY
    # Note: multilooking is only applied to the 3D data cubes and their related operations:
    # e.g. spatial indexing, referencing, etc. All the other variables are in the original grid
    # so that users get the same result as the non-multilooked version.
    if inps.multilook and inps.multilook_num == 1:
        inps.multilook_num = pp.auto_multilook_num(
            inps.pix_box, inps.num_date,
            max_memory=inps.maxMemory,
            print_msg=inps.print_msg,
        )

    ## reference pixel
    if not inps.ref_lalo and 'REF_LAT' in atr.keys():
        inps.ref_lalo = (float(atr['REF_LAT']), float(atr['REF_LON']))
    if inps.ref_lalo:
        # set longitude to [-180, 180)
        if inps.coord_unit.lower().startswith('deg') and inps.ref_lalo[1] >= 180.:
            inps.ref_lalo[1] -= 360.
        # ref_lalo --> ref_yx if not set in cmd
        if not inps.ref_yx:
            # Integer coordinates for data flow (mask, array indexing, backward compat)
            inps.ref_yx = inps.coord.geo2radar(inps.ref_lalo[0], inps.ref_lalo[1],
                                               print_msg=False)[0:2]
            # Float coordinates for bilinear interpolation (when --precise is set)
            if getattr(inps, 'precise', False):
                inps.ref_yx_float = inps.coord.geo2radar(inps.ref_lalo[0], inps.ref_lalo[1],
                                                          print_msg=False, precise=True)[0:2]
                vprint(f'reference point in y/x (precise): ({inps.ref_yx_float[0]:.4f}, {inps.ref_yx_float[1]:.4f})')

    # use REF_Y/X if ref_yx not set in cmd
    if not inps.ref_yx and 'REF_Y' in atr.keys():
        inps.ref_yx = (int(atr['REF_Y']), int(atr['REF_X']))

    # print/plot ref_yx/lalo info
    if inps.ref_yx:
        vprint(f'reference point in y/x: ({inps.ref_yx[0]:.2f}, {inps.ref_yx[1]:.2f})')
        # ref_yx --> ref_lalo if in geo-coord [for plotting purpose only]
        if 'Y_FIRST' in atr.keys():
            inps.ref_lalo = inps.coord.radar2geo(inps.ref_yx[0], inps.ref_yx[1], print_msg=False)[0:2]
            vprint(f'reference point in lat/lon: ({inps.ref_lalo[0]:.6f}, {inps.ref_lalo[1]:.6f})')
        if hasattr(inps, 'ref_lalo') and inps.ref_lalo:
            pass  # info already shown below

    # do not plot native reference point if it's out of the coverage due to subset
    if (inps.ref_yx and 'Y_FIRST' in atr.keys()
        and is_native_reference_point(inps.ref_yx, atr)
        and not (    inps.pix_box[0] <= inps.ref_yx[1] < inps.pix_box[2]
                 and inps.pix_box[1] <= inps.ref_yx[0] < inps.pix_box[3])):
        inps.disp_ref_pixel = False
        vprint('WARNING: the native REF_Y/X is out of subset box, thus do not display')

    ## initial pixel coord
    if inps.lalo:
        inps.yx = inps.coord.geo2radar(inps.lalo[0], inps.lalo[1], print_msg=False)[0:2]
    if inps.yx:
        try:
            inps.lalo = inps.coord.radar2geo(inps.yx[0], inps.yx[1], print_msg=False)[0:2]
        except FileNotFoundError:
            inps.lalo = None

    ## figure settings
    # Flip up-down / left-right
    if inps.auto_flip:
        inps.flip_lr, inps.flip_ud = pp.auto_flip_direction(atr, print_msg=inps.print_msg)

    # Transparency - Alpha
    if not inps.transparency:
        # Auto adjust transparency value when showing shaded relief DEM
        if inps.dem_file and inps.disp_dem_shade:
            inps.transparency = 0.7
        else:
            inps.transparency = 1.0

    ## display unit and wrap
    # if wrap_step == 2*np.pi (default value), set disp_unit_img = radian;
    # otherwise set disp_unit_img = disp_unit
    inps.disp_unit_img = inps.disp_unit
    if inps.wrap:
        inps.vlim = inps.wrap_range

        if (inps.wrap_range[1] - inps.wrap_range[0]) == 2*np.pi:
            inps.disp_unit_img = 'radian'

        if inps.disp_unit_img == 'radian':
            inps.range2phase = -4. * np.pi / float(atr['WAVELENGTH'])
            if   'cm' == inps.disp_unit.split('/')[0]:   inps.range2phase /= 100.
            elif 'mm' == inps.disp_unit.split('/')[0]:   inps.range2phase /= 1000.
            elif 'm'  == inps.disp_unit.split('/')[0]:   inps.range2phase /= 1.
            else:
                raise ValueError(f'un-recognized display unit: {inps.disp_unit}')

    inps.cbar_label = 'Amplitude' if atr['DATA_TYPE'].startswith('complex') else 'Displacement'
    inps.cbar_label += f' [{inps.disp_unit_img}]'

    ## fit a suite of time func to the time series
    inps.model = time_func.inps2model(inps, date_list=inps.date_list, print_msg=inps.print_msg)

    # dense TS for plotting
    inps.date_list_fit = ptime.get_date_range(inps.date_list[0], inps.date_list[-1])
    inps.dates_fit = ptime.date_list2vector(inps.date_list_fit)[0]
    inps.G_fit = time_func.get_design_matrix4time_func(
        date_list=inps.date_list_fit,
        model=inps.model,
        seconds=inps.seconds)

    return inps, atr


def subset_and_multilook_yx(yx, pix_box=None, multilook_num=1):
    """Update row/col number of one pixel due to subset and multilooking."""
    y, x = yx
    if pix_box is not None:
        y -= pix_box[1]
        x -= pix_box[0]
    if multilook_num > 1:
        y = int((y - int(multilook_num / 2)) / multilook_num)
        x = int((x - int(multilook_num / 2)) / multilook_num)
    return (y, x)


def _bilinear_interpolate_ts(ts_data_3d, y_f, x_f):
    """Bilinear interpolation of 3D time series at sub-pixel location (y_f, x_f).

    Parameters
    ----------
    ts_data_3d : np.ndarray, shape (n_date, n_row, n_col)
        3D InSAR time series data.
    y_f : float
        Floating-point row coordinate in the local (subsetted) data frame.
    x_f : float
        Floating-point column coordinate in the local (subsetted) data frame.

    Returns
    -------
    ts_interp : 1D np.ndarray, shape (n_date,)
        Interpolated time series at the sub-pixel location.
    """
    ndate, n_rows, n_cols = ts_data_3d.shape

    y_f = max(0.0, min(y_f, n_rows - 1.001))
    x_f = max(0.0, min(x_f, n_cols - 1.001))

    y0 = int(np.floor(y_f))
    x0 = int(np.floor(x_f))
    y1 = min(y0 + 1, n_rows - 1)
    x1 = min(x0 + 1, n_cols - 1)

    wy = y_f - y0
    wx = x_f - x0

    w00 = (1 - wy) * (1 - wx)
    w01 = (1 - wy) * wx
    w10 = wy * (1 - wx)
    w11 = wy * wx

    ts_interp = (w00 * ts_data_3d[:, y0, x0] +
                 w01 * ts_data_3d[:, y0, x1] +
                 w10 * ts_data_3d[:, y1, x0] +
                 w11 * ts_data_3d[:, y1, x1])

    return ts_interp


def is_native_reference_point(ref_yx, atr, max_err=0):
    """Check if the given ref_yx is the native reference point or not.

    Parameters: ref_yx  - list of int, input reference point in row/col
                atr     - dict, attributes, to retrieve the native REF_Y/X
                max_err - int, maximum allowable error to account for
                          potential geo2radar coordinate conversion error
    """
    if 'REF_Y' not in atr.keys():
        return False

    ref_x = int(atr['REF_X'])
    ref_y = int(atr['REF_Y'])
    x0, x1 = ref_x - max_err, ref_x + max_err
    y0, y1 = ref_y - max_err, ref_y + max_err

    return x0 <= ref_yx[1] < x1 and y0 <= ref_yx[0] < y1


def read_exclude_date(input_ex_date, dateListAll):
    """Read exclude list of dates
    Parameters: input_ex_date : list of string in YYYYMMDD or filenames for excluded dates
                dateListAll   : list of string in YYYYMMDD for all dates
    Returns:    ex_date_list  : list of string in YYYYMMDD for excluded dates
                ex_dates      : list of datetime.datetime objects for excluded dates
                ex_flag       : 1D np.ndarray in size of (num_date),
                                1/True for kept, 0/False for excluded
    """
    # default value
    ex_date_list = []
    ex_dates = []
    ex_flag = np.ones((len(dateListAll)), np.bool_)

    ex_date_list = ptime.read_date_list(input_ex_date, date_list_all=dateListAll)
    if ex_date_list:
        ex_dates = ptime.date_list2vector(ex_date_list)[0]
        for i in ex_date_list:
            ex_flag[dateListAll.index(i)] = False
        vprint('exclude date:'+str(ex_date_list))
    return ex_date_list, ex_dates, ex_flag


def read_timeseries_data(inps):
    """Read data of time-series files
    Parameters: inps : Namespace of input arguments
    Returns:    ts_data : list of 3D np.array in size of (num_date, length, width)
                mask : 2D np.array in size of (length, width)
                inps : Namespace of input arguments
    """
    ## read list of 3D time-series
    ts_data = []
    for fname in inps.file:
        msg = f'reading timeseries from file {fname}'
        msg += f' with step of {inps.multilook_num} by {inps.multilook_num}' if inps.multilook_num > 1 else ''
        vprint(msg)

        data, atr = readfile.read(
            fname,
            datasetName=inps.date_list,
            box=inps.pix_box,
            xstep=inps.multilook_num,
            ystep=inps.multilook_num)

        if atr['DATA_TYPE'].startswith('complex'):
            vprint('input data is complex, calculate its amplitude and continue')
            data = np.abs(data)

        if inps.ref_yx and not is_native_reference_point(inps.ref_yx, atr):
            # Bilinear interpolation for sub-pixel accuracy at the reference point.
            # Prefer float ref_yx_float (from --precise) over integer ref_yx.
            ref_coord = getattr(inps, 'ref_yx_float', None) or inps.ref_yx
            y_f, x_f = ref_coord[0], ref_coord[1]
            if inps.pix_box is not None:
                y_f -= inps.pix_box[1]
                x_f -= inps.pix_box[0]
            if inps.multilook_num > 1:
                half_ml = inps.multilook_num // 2
                y_f = (y_f - half_ml) / inps.multilook_num
                x_f = (x_f - half_ml) / inps.multilook_num
            ref_phase = _bilinear_interpolate_ts(data, y_f, x_f)
            data -= np.tile(ref_phase.reshape(-1, 1, 1), (1, data.shape[-2], data.shape[-1]))
            vprint(f'spatial referencing (bilinear interp): local yx=({y_f:.2f}, {x_f:.2f}), '
                   f'global yx=({inps.ref_yx[0]:.2f}, {inps.ref_yx[1]:.2f})')
        elif inps.ref_yx:
            vprint(f'spatial referencing: using native REF_Y/X (within 2 px), no re-interpolation')

        if inps.ref_idx is not None:
            vprint(f'reference to date: {inps.date_list[inps.ref_idx]}')
            data -= np.tile(data[inps.ref_idx, :, :], (inps.num_date, 1, 1))

        # Display Unit
        data, inps.disp_unit, inps.unit_fac = pp.scale_data2disp_unit(
            data,
            metadata=atr,
            disp_unit=inps.disp_unit)
        ts_data.append(data)

    ## mask file: input mask file + non-zero ts pixels - ref_point
    mask = pp.read_mask(
        inps.file[0],
        mask_file=inps.mask_file,
        datasetName='displacement',
        box=inps.pix_box,
        xstep=inps.multilook_num,
        ystep=inps.multilook_num,
        print_msg=inps.print_msg,
    )[0]

    if mask is None:
        mask = np.ones(ts_data[0].shape[-2:], np.bool_)

    ts_stack = np.nansum(ts_data[0], axis=0)
    mask[np.isnan(ts_stack)] = False
    # keep all-zero value for unwrapError time-series
    if atr['UNIT'] not in ['cycle']:
        mask[ts_stack == 0.] = False
    del ts_stack

    # do not mask the reference point
    x0, y0, x1, y1 = inps.pix_box
    if inps.ref_yx and (x0 <= inps.ref_yx[1] < x1) and (y0 <= inps.ref_yx[0] < y1):
        (ry, rx) = subset_and_multilook_yx(inps.ref_yx, inps.pix_box, inps.multilook_num)
        mask[ry, rx] = True

    ## default vlim
    inps.dlim = [np.nanmin(ts_data[0]), np.nanmax(ts_data[0])]
    if not inps.vlim:
        inps.cmap_lut, inps.vlim = pp.auto_adjust_colormap_lut_and_disp_limit(
            ts_data[0], num_multilook=10, print_msg=inps.print_msg,
        )[:2]
    vprint(f'data    range: {inps.dlim} {inps.disp_unit}')
    vprint(f'display range: {inps.vlim} {inps.disp_unit}')

    ## default ylim
    num_file = len(inps.file)
    if not inps.ylim:
        ts_data_mli = multilook_data(np.squeeze(ts_data[-1]), 4, 4)
        if inps.zero_first:
            ts_data_mli -= np.tile(ts_data_mli[inps.zero_idx, :, :], (inps.num_date, 1, 1))
        ymin, ymax = (np.nanmin(ts_data_mli[inps.ex_flag != 0]),
                      np.nanmax(ts_data_mli[inps.ex_flag != 0]))
        ybuffer = (ymax - ymin) * 0.05
        inps.ylim = [ymin - ybuffer, ymax + ybuffer]
        if inps.offset:
            inps.ylim[1] += inps.offset * (num_file - 1)
        del ts_data_mli

    return ts_data, mask, inps


def plot_ts_errorbar(ax, dis_ts, inps, ppar):
    """Plot displacement time series with error bars."""

    # make a local copy
    dates = np.array(inps.dates)
    d_ts = dis_ts[:]

    kwargs = dict(
        fmt='-o', ms=ppar.ms, lw=0, alpha=1,
        elinewidth=inps.edge_width,
        capsize=ppar.ms*0.5, mew=inps.edge_width,
    )

    if inps.ex_date_list:
        # Update displacement time-series
        d_ts = dis_ts[inps.ex_flag == 1]
        dates = dates[inps.ex_flag == 1]

        # Plot excluded dates
        ex_d_ts = dis_ts[inps.ex_flag == 0]
        ax.errorbar(
            inps.ex_dates, ex_d_ts,
            yerr=inps.ex_error_ts,
            color='gray',
            ecolor='gray',
            **kwargs,
        )

    # Plot kept dates
    ax.errorbar(
        dates, d_ts,
        yerr=inps.error_ts,
        label=ppar.label,
        color=ppar.mfc,
        ecolor=ppar.mfc,
        **kwargs
    )

    handles = ax.get_legend_handles_labels()[0]

    return ax, handles[-1]


def plot_ts_scatter(ax, dis_ts, inps, ppar):
    """Plot displacement time series as a scatter plot."""

    # make a local copy
    dates = np.array(inps.dates)
    d_ts = dis_ts[:]

    kwargs = dict(ms=ppar.ms, marker=inps.marker,
                  markerfacecolor='none', markeredgecolor=ppar.mfc, markeredgewidth=1.5)

    if inps.ex_date_list:
        # Update displacement time-series
        d_ts = dis_ts[inps.ex_flag == 1]
        dates = dates[inps.ex_flag == 1]

        # Plot excluded dates
        ex_d_ts = dis_ts[inps.ex_flag == 0]
        ax.plot(inps.ex_dates, ex_d_ts, color='gray', lw=0, **kwargs)

    # Plot kept dates
    handle, = ax.plot(dates, d_ts, color=ppar.mfc, label=ppar.label, lw=inps.linewidth, **kwargs)
    return ax, handle


def plot_ts_fit(ax, ts_fit, inps, ppar, m_strs=None, ts_fit_lim=None):
    """Plot time series fitting results."""
    # plot the model prediction uncertainty boundaries
    h0 = None
    if ts_fit_lim is not None:
        h0 = ax.fill_between(
            inps.dates_fit, ts_fit_lim[0], ts_fit_lim[1],
            fc=ppar.color, ec='none', alpha=0.2,
        )

    # plot the model prediction curve in a fine date_lists
    h1, = ax.plot(inps.dates_fit, ts_fit, color=ppar.color, lw=ppar.linewidth, alpha=0.8)

    # legend
    handles = []
    labels = []
    kwargs = dict(loc='best', fontsize=ppar.fontsize)
    if h0 is not None:
        handles.append((h0, h1))
        labels.append('time func. pred. (w. 95% conf. interval.)')

    # print model parameters on the plot
    # link for auto text loc: https://stackoverflow.com/questions/7045729
    if m_strs:
        if len(labels) == 0:
            kwargs['handlelength'] = 0
            kwargs['handletextpad'] = 0
            kwargs['borderaxespad'] = 1.0

        # remove multiple spaces for better display since matplotlib does not give good alignment as in the terminal
        handles += [patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)] * len(m_strs)
        labels += [re.sub(' +', ' ', x) for x in m_strs]

    # do not print model parameters for multi files [temporarily]
    # solutions:
    # 1. plot multiple legends, need to find out how to layout them, otherwise they would overlap on top of each other
    #    link: https://matplotlib.org/stable/tutorials/intermediate/legend_guide.html#multiple-legends-on-the-same-axes
    # 2. update the same legend, by saving the handles and labels for all files
    if len(labels) > 0 and len(inps.file) == 1:
        ax.legend(handles, labels, **kwargs)

    return ax


def get_model_param_str(model, ds_dict, disp_unit='cm'):
    """Summary model parameters in a str paragraph.
    Parameters: model     - dict, model dictionary
                ds_dict   - dict, est. time func. param
                disp_unit - float, display unit for length, which can be scaled
    Returns:    ds_strs   - list of strings, each to summary the result of one time func
    """

    # dataset unit dict
    ds_unit_dict = ts2vel.model2hdf5_dataset(model)[2]
    ds_names = list(ds_unit_dict.keys())

    # update ds_unit_dict based on disp_unit
    for ds_name in ds_names:
        units = ds_unit_dict[ds_name].split('/')
        if units[0] == 'm' and disp_unit != 'm':
            units[0] = disp_unit
            ds_unit_dict[ds_name] = '/'.join(units)

    # list of dataset names
    ds_names = [x for x in ds_dict.keys() if not x.endswith('Std') and x not in ['intercept']]
    w_key = max(len(x) for x in ds_names)
    w_val = max(len(f'{x[0]:.2f}') for x in ds_dict.values())

    ds_strs = []
    for ds_name in ds_names:
        # get param info
        ds_value = ds_dict[ds_name]
        ds_unit = ds_unit_dict[ds_name]
        ds_std_value = ds_dict.get(ds_name+'Std', None)

        # compose string
        ds_str = f'{ds_name:<{w_key}}: {ds_value[0]:>{w_val}.2f}'
        ds_str += f' +/- {ds_std_value[0]:>{w_val}.2f}' if ds_std_value is not None else ''
        ds_str += f' {ds_unit}'
        ds_strs.append(ds_str)

    return ds_strs


def fit_time_func(model, date_list, ts_dis, disp_unit='cm', G_fit=None, conf_level=0.95, seconds=0):
    """Fit a suite of time functions to the time series.
    Equations:  Gm = d
    Parameters: model      - dict of time functions, check utils.time_func.estimate_time_func() for details.
                date_list  - list of dates in YYYYMMDD format
                ts_dis     - 1D np.ndarray, displacement time series
                disp_unit  - str, display unit for length, which can be scaled
                G_fit      - 2D np.ndarray, design matrix for the dense time series prediction plot
                conf_level - float in [0,1], confidence level of the plotted confidence intervals
    Returns:    m_strs     - dict, dictionary in {ds_name: ds_value}
                ts_fit     - 1D np.ndarray, dense time series fit for plotting
                ts_fit_lim - list of 1D np.ndarray, the lower and upper
                             boundaries of dense time series fit for plotting
    """
    # init output
    m_strs = []
    ts_fit = None
    ts_fit_lim = None

    if np.all(np.isnan(ts_dis)):
        return m_strs, ts_fit, ts_fit_lim

    # 1.1 estimate time func parameter via least squares (OLS)
    G, m, e2 = time_func.estimate_time_func(
        model=model,
        date_list=date_list,
        dis_ts=ts_dis,
        seconds=seconds)

    # 1.2 calc the precision of time func parameters
    # using the OLS estimation residues e2 = sum((d - Gm) ** 2)
    # assuming obs errors following normal distribution in time
    num_obs = len(date_list)
    num_param = G.shape[1]
    G_inv = linalg.inv(np.dot(G.T, G))
    m_var_sum = e2.flatten() / (num_obs - num_param)
    m_std = np.sqrt(np.dot(np.diag(G_inv).reshape(-1, 1), m_var_sum))

    # 1.3 translate estimation result into HDF5 ready datasets
    # AND compose list of strings for printout
    m_dict = ts2vel.model2hdf5_dataset(model, m, m_std)[0]
    m_strs = get_model_param_str(model, m_dict, disp_unit=disp_unit)

    # 2. reconstruct the fine resolution function
    if G_fit is not None:
        ts_fit = np.matmul(G_fit, m)
        ts_fit_std = np.sqrt(np.diag(G_fit.dot(np.diag(m_std**2)).dot(G_fit.T)))

        # calc confidence interval
        # references:
        # 1. Exercise 6.4 OMT: Interpretation from Hanssen et al. (2017) EdX online course.
        #    Hanssen, R., Verhagen, S. and Samiei-Esfahany, S., (2017) Observation Theory: Estimating the Unknown,
        #    Available at: https://www.edx.org/course/observation-theory-estimating-the-unknown
        # 2. https://stackoverflow.com/questions/20626994
        alpha = 1 - conf_level                                # level of significance
        conf_int_scale = stats.norm.ppf(1 - alpha / 2)        # scaling factor for confidence interval
        ts_fit_lim = [ts_fit - conf_int_scale * ts_fit_std,
                      ts_fit + conf_int_scale * ts_fit_std]

    return m_strs, ts_fit, ts_fit_lim


def get_point_coord_str(y, x, coord_obj, lalo_digit=5, use_float=False):
    """Get the string of the point coordinates.

    Parameters: y / x      - float or int, row / column number
                coord_obj  - mintpy.objects.coordinate object
                lalo_digit - int, digit of the decimal place for lat/lon
                use_float  - bool, show fractional y/x for sub-pixel locations
    Returns:    pts_str    - str, point coordinate string
    """
    if use_float:
        coord_str = f'Y/X = {y:.2f}, {x:.2f}'
    else:
        coord_str = f'Y/X = {y}, {x}'
    try:
        lat, lon = coord_obj.radar2geo(y, x, print_msg=False)[0:2]
        coord_str += f', lat/lon = {lat:.{lalo_digit}f}, {lon:.{lalo_digit}f}'
    except FileNotFoundError:
        pass
    return coord_str


def save_ts_data_and_plot(yx, d_ts, m_strs, inps):
    """Save TS data and plots into files."""
    y, x = yx
    vprint(f'save info on pixel ({y}, {x})')

    # output file name
    if inps.outfile:
        inps.outfile_base, fext = os.path.splitext(inps.outfile[0])
        if fext != '.pdf':
            msg = 'Output file extension is fixed to .pdf,'
            msg += f' input extension {fext} is ignored.'
            vprint(msg)
    else:
        inps.outfile_base = f'y{y}x{x}'

    # TXT - point time-series and time func param
    outName = f'{inps.outfile_base}_ts.txt'
    header = f'time-series file = {inps.file[0]}\n'
    header += f'{get_point_coord_str(y, x, inps.coord, inps.lalo_digit)}\n'
    header += f'reference pixel: y={inps.ref_yx[0]}, x={inps.ref_yx[1]}\n' if inps.ref_yx else ''
    header += f'reference date: {inps.date_list[inps.ref_idx]}\n' if inps.ref_idx else ''
    header += f'exclude date: {inps.ex_date_list}\n' if inps.ex_date_list else ''
    header += 'estimated time function parameters:\n'
    for m_str in m_strs:
        header += f'    {m_str}\n'
    header += f'unit: {inps.disp_unit}'

    # prepare data
    data = np.hstack((np.array(inps.date_list).reshape(-1, 1), d_ts.reshape(-1, 1)))

    # write
    np.savetxt(outName, data, fmt='%s', delimiter='\t', header=header)
    print('save displacement time-series to file: '+outName)

    # Figure - point time-series
    outName = f'{inps.outfile_base}_ts.pdf'
    inps.fig_pts.savefig(outName, bbox_inches='tight', transparent=True, dpi=inps.fig_dpi)
    print('save time-series plot to file: '+outName)

    # Figure - map
    outName = f'{inps.outfile_base}_{inps.date_list[inps.idx]}.png'
    inps.fig_img.savefig(outName, bbox_inches='tight', transparent=True, dpi=inps.fig_dpi)
    print('save map plot to file: '+outName)
    return


class timeseriesViewer():
    """Class for tsview.py

    Example:
        from mintpy.cli.tsview import cmd_line_parse
        from mintpy.tsview import timeseriesViewer

        cmd = 'timeseries.h5 --yx 273 271 --figsize 8 4'
        inps = cmd_line_parse(cmd.split())
        obj = timeseriesViewer(inps)
        obj.open()
        obj.plot()
    """

    def __init__(self, inps):

        # figure variables
        self.figname_img = 'Cumulative Displacement Map'
        self.figsize_img = None
        self.fig_img = None
        self.ax_img = None
        self.cbar_img = None
        self.img = None

        self.ax_tslider = None
        self.tslider = None

        self.figname_pts = 'Point Displacement Time-series'
        self.figsize_pts = None
        self.fig_pts = None
        self.ax_pts = None

        # track whether user explicitly set --ylim
        self.ylim_user_set = (inps.ylim is not None)

        # copy inps to self object
        for key, value in inps.__dict__.items():
            setattr(self, key, value)

    def open(self):
        global vprint
        vprint = print if self.print_msg else lambda *args, **kwargs: None

        # print command line
        if self.argv is not None:
            print(f'{os.path.basename(__file__)} ' + ' '.join(self.argv))

        # matplotlib backend setting
        # Auto-detect headless environment: if no DISPLAY and user
        # didn't explicitly disable display, switch to Agg backend
        # to avoid silent crash or hang from plt.show().
        # When running in Jupyter with %matplotlib widget, respect the
        # user-configured interactive backend.
        if not self.disp_fig:
            plt.switch_backend('Agg')
        elif not os.environ.get('DISPLAY'):
            current_backend = plt.get_backend().lower()
            if 'agg' in current_backend or 'template' in current_backend:
                print('WARNING: No DISPLAY detected, auto-switching to headless mode. '
                      'Use --save to save figures to files.')
                self.disp_fig = False
                plt.switch_backend('Agg')

        self, self.atr = read_init_info(self)

        # input figsize for the image/point time-series plot
        self.figsize_img = self.fig_size_img
        self.figsize_pts = self.fig_size
        self.pts_marker = 'r^'
        self.pts_marker_size = 6.

    def _ensure_view_attrs(self):
        """Set default attributes required by view.plot_slice and related plotting functions."""
        defaults = {
            # shapefile / line overlay
            'shp_file': None,
            'lalo_file': None,
            'gmt_file': None,
            'shp_color': 'k',
            'shp_linewidth': 0.5,
            'shp_min_dist': 0.1,
            # coastline
            'coastline': None,
            'coastline_linewidth': 1.0,
            # lat/lon labels
            'lalo_step': None,
            'lalo_loc': [1, 0, 0, 1],
            'lalo_max_num': 3,
            'lalo_offset': None,
            'lalo_font_size': None,
            'lalo_label': False,
            # scale bar
            'scalebar': [0.2, 0.2, 0.1],
            'disp_scalebar': True,
            'scalebar_pad': 0.05,
            'scalebar_linewidth': 2.0,
            # axis & tick
            'disp_axis': True,
            'disp_tick': True,
            'tick_right': False,
            'ylabel_rot': None,
            # title
            'disp_title': True,
            'title_in': False,
            'fig_title': '',
            'font_color': 'k',
            'font_size': 12,
            # figure
            'disp_whitespace': True,
            'fig_ext': '.png',
            'fig_dpi': 300,
            'outfile': None,
            'save_fig': False,
            # DEM
            'dem_file': None,
            'disp_dem_shade': False,
            'disp_dem_blend': False,
            'disp_dem_contour': False,
            # reference pixel
            'disp_ref_pixel': True,
            'ref_marker': 'ks',
            'ref_marker_size': 6,
            # points of interest
            'pts_yx': None,
            'pts_lalo': None,
            'pts_marker': 'r^',
            'pts_marker_size': 6.,
            # GNSS
            'disp_gnss': False,
            'gnss_component': None,
            'gnss_source': 'UNR',
            'ref_gnss_site': None,
            'gnss_start_date': None,
            'gnss_end_date': None,
            'horz_az': -90.,
            'gnss_redo': False,
            'gnss_label': False,
            'mask_gnss': False,
            'gnss_marker_size': 6.,
            'gnss_median_remove': False,
            # misc
            'msk': self.mask if hasattr(self, 'mask') else None,
            'style': 'image',
            'interpolation': 'nearest',
            'colormap': 'jet',
            'cmap_lut': 256,
            'cmap_vlist': [0.0, 0.7, 1.0],
            'vlim': None,
            'wrap': False,
            'wrap_range': [-np.pi, np.pi],
        }
        for attr, default in defaults.items():
            if not hasattr(self, attr):
                setattr(self, attr, default)


    def _get_gnss_stations_info(self):
        """Retrieve GNSS station names and coordinates within the map extent."""
        from pyproj import CRS, Transformer
        if not (self.geo_box and self.fig_coord == 'geo' and self.disp_gnss):
            self.gnss_site_names = []
            self.gnss_site_lats = np.array([])
            self.gnss_site_lons = np.array([])
            return

        # geo_box is (W, N, E, S) as returned by MintPy
        west, north, east, south = self.geo_box

        if self.coord_unit.startswith('meter'):
            # Convert UTM bounds to lat/lon
            utm_zone = self.atr.get('UTM_ZONE', None)
            if utm_zone is None:
                print('WARNING: UTM_ZONE not found in metadata, cannot convert to lat/lon for GNSS.')
                self.gnss_site_names = []
                self.gnss_site_lats = np.array([])
                self.gnss_site_lons = np.array([])
                return

            zone_num = int(''.join(filter(str.isdigit, utm_zone)))
            hemisphere = 'north' if utm_zone[-1].upper() == 'N' else 'south'
            utm_crs = CRS.from_proj4(
                f'+proj=utm +zone={zone_num} +{hemisphere} +ellps=WGS84 +datum=WGS84 +units=m +no_defs'
            )
            wgs84 = CRS.from_epsg(4326)
            transformer = Transformer.from_crs(utm_crs, wgs84, always_xy=True)

            # Build the four corners: SW, SE, NE, NW
            xs = [west, east, east, west]
            ys = [south, south, north, north]
            lons, lats = transformer.transform(xs, ys)
            SNWE_deg = (min(lats), max(lats), min(lons), max(lons))
        else:
            # Already in degrees; geo_box is (W, N, E, S)
            SNWE_deg = (south, north, west, east)

        # Search GNSS stations
        site_names, site_lats, site_lons = gnss_mod.search_gnss(
            SNWE=SNWE_deg,
            start_date=self.gnss_start_date,
            end_date=self.gnss_end_date,
            source=self.gnss_source,
            print_msg=self.print_msg,
        )

        self.gnss_site_names = site_names
        self.gnss_site_lats = site_lats
        self.gnss_site_lons = site_lons
        vprint(f'GNSS stations in view: {len(site_names)}')

        # Determine the effective GNSS reference site for spatial double-differencing.
        # If ref_gnss_site is set explicitly, use it. Otherwise, find the GNSS station
        # nearest to ref_lalo (the InSAR spatial reference point).
        self.eff_ref_site = self.ref_gnss_site
        self.eff_ref_lat = None
        self.eff_ref_lon = None
        if (not self.eff_ref_site
                and hasattr(self, 'ref_lalo')
                and self.ref_lalo is not None
                and len(site_lats) > 0):
            ref_lat, ref_lon = self.ref_lalo[0], self.ref_lalo[1]
            if abs(ref_lat) > 90 or abs(ref_lon) > 180:
                from mintpy.utils import utils0 as ut0
                ref_lat, ref_lon = ut0.utm2latlon(self.atr, ref_lon, ref_lat)
            dist = np.sqrt((site_lats - ref_lat)**2 + (site_lons - ref_lon)**2)
            self.eff_ref_site = site_names[int(np.argmin(dist))]
            vprint(f'Implicit GNSS ref site: {self.eff_ref_site} '
                   f'(nearest to InSAR ref point)')
        if self.eff_ref_site:
            for j, name in enumerate(site_names):
                if name == self.eff_ref_site:
                    self.eff_ref_lat = site_lats[j]
                    self.eff_ref_lon = site_lons[j]
                    break

    def _get_gnss_timeseries(self, site_name, ignore_ref_site=False, _force_ref_site=None):
        """Get GNSS LOS displacement, properly aligned to InSAR reference date
        and using geometry file for accurate incidence / azimuth angles.

        Parameters
        ----------
        site_name : str
            GNSS site name.
        ignore_ref_site : bool
            If True, ignore self.ref_gnss_site when computing LOS displacement.
        _force_ref_site : str or None
            If provided, use this as the ref_site for GNSS spatial differencing,
            overriding self.ref_gnss_site. Used internally for consistent
            double-difference comparison across stations.
        """
        # Determine reference site name early (also needed for cache-hit bias)
        ref_site_name = (_force_ref_site if _force_ref_site
                         else self.ref_gnss_site
                         or getattr(self, 'eff_ref_site', None))

        # Return cached result if available (composite key includes ref_site)
        # Apply per-station bias on top of cached bias-free data
        cache_key = (site_name, _force_ref_site or '__default__')
        if (hasattr(self, '_gnss_ts_cache')
                and cache_key in self._gnss_ts_cache):
            dates, dis = self._gnss_ts_cache[cache_key]
            sb = getattr(self, 'gnss_station_biases', {}).get(site_name, 0.0)
            if abs(sb) > 1e-9 and not (ref_site_name and site_name == ref_site_name):
                dis = dis + sb
            return dates, dis

        gnss_cls = gnss_mod.get_gnss_class(self.gnss_source)
        gnss_obj = gnss_cls(site=site_name)

        use_ref_site = (ref_site_name and ref_site_name != site_name
                        and not ignore_ref_site)

        # Geometry source: always prefer the user‑provided lookup file
        geom_obj = self.lookup_file if self.lookup_file else self.atr
        if isinstance(geom_obj, str):
            vprint(f'Using geometry file: {geom_obj} for GNSS LOS projection')
        else:
            vprint('WARNING: No geometry file provided, using metadata HEADING '
                '(may be inaccurate for LOS projection)')

        # 1. Get raw GNSS LOS displacement (absolute, no spatial differencing yet).
        #    Pre-compute geometry to enable caching across stations.
        if not hasattr(self, '_gnss_geom_cache'):
            self._gnss_geom_cache = {}
        geom_key = (site_name, geom_obj if isinstance(geom_obj, str) else '__dict__')
        if geom_key not in self._gnss_geom_cache:
            self._gnss_geom_cache[geom_key] = gnss_obj.get_los_geometry(geom_obj)
        inc_angle, az_angle = self._gnss_geom_cache[geom_key]

        dates = gnss_obj.read_displacement(
            self.gnss_start_date, self.gnss_end_date, print_msg=self.print_msg)[0]
        dis, std = gnss_obj.displacement_enu2los(
            inc_angle, az_angle,
            gnss_comp=self.gnss_component,
            horz_az_angle=getattr(self, 'horz_az', -90.),
        )

        # 2. Spatial differencing: for each target date, subtract the mean of
        #    reference station values within +-1 day to reduce noise amplification.
        if use_ref_site and len(dis) > 0:
            # Cache ref station data to avoid re-reading geometry + GNSS files
            if not hasattr(self, '_gnss_ref_cache'):
                self._gnss_ref_cache = {}
            ref_cache_key = (ref_site_name, geom_obj if isinstance(geom_obj, str) else '__dict__')
            if ref_cache_key in self._gnss_ref_cache:
                ref_dates, ref_dis = self._gnss_ref_cache[ref_cache_key]
            else:
                ref_obj = gnss_cls(site=ref_site_name)
                # Use cached geometry for ref site too
                ref_geom_key = (ref_site_name, geom_obj if isinstance(geom_obj, str) else '__dict__')
                if ref_geom_key not in self._gnss_geom_cache:
                    self._gnss_geom_cache[ref_geom_key] = ref_obj.get_los_geometry(geom_obj)
                ref_inc, ref_az = self._gnss_geom_cache[ref_geom_key]
                ref_dates = ref_obj.read_displacement(
                    self.gnss_start_date, self.gnss_end_date, print_msg=self.print_msg)[0]
                ref_dis, ref_std = ref_obj.displacement_enu2los(
                    ref_inc, ref_az,
                    gnss_comp=self.gnss_component,
                    horz_az_angle=getattr(self, 'horz_az', -90.),
                )
                ref_dis = ref_dis
                self._gnss_ref_cache[ref_cache_key] = (ref_dates, ref_dis)
            if len(ref_dates) > 0 and len(ref_dis) > 0:
                for i, d in enumerate(dates):
                    day_diffs = np.array([abs((rd - d).total_seconds()) / 86400.0
                                         for rd in ref_dates])
                    within_1day = day_diffs <= 1.0
                    if np.sum(within_1day) > 0:
                        dis[i] -= np.mean(ref_dis[within_1day])
                    else:
                        # fallback: closest ref date (skip if NaN)
                        fallback_idx = int(np.argmin(day_diffs))
                        if not np.isnan(ref_dis[fallback_idx]):
                            dis[i] -= ref_dis[fallback_idx]

        # 3. Time alignment: use mean of GNSS values within +-3 days of
        #    InSAR reference date (InSAR interval is 6 or 12 days, so
        #    a 3-day window captures the nearest InSAR acquisition).
        if self.ref_date and len(dis) > 0:
            ref_dt = ptime.date_list2vector([self.ref_date])[0][0]
            max_diff_days = 3
            day_diffs = np.array([abs((d - ref_dt).total_seconds()) / 86400.0
                                  for d in dates])
            within_window = day_diffs <= max_diff_days
            if np.sum(within_window) > 0:
                ref_value = np.mean(dis[within_window])
                dis -= ref_value
                vprint(f'Temporal alignment: GNSS {site_name} shifted by {ref_value:.3f} '
                       f'(mean of {np.sum(within_window)} pts within +-{max_diff_days}d)')
            else:
                vprint(f'WARNING: No GNSS observation within {max_diff_days} days of '
                       f'InSAR reference date {self.ref_date} for station {site_name}. '
                       f'Alignment skipped.')

        # 4. If the site is the reference station itself, force zero (to match map display)
        if (ref_site_name and site_name == ref_site_name
                and not ignore_ref_site):
            dis = np.zeros_like(dis)

        # 5. Apply unit conversion
        dis *= self.unit_fac

        # 6. Apply per-station median bias (if already computed) to align
        #    GNSS with InSAR at this specific station.
        #    bias = median(InSAR - GNSS), so ADD it to GNSS:
        #    GNSS_corrected = GNSS + bias = GNSS + median(InSAR - GNSS)
        #    Skip the reference station itself — its value is force-zeroed.
        sb = getattr(self, 'gnss_station_biases', {}).get(site_name, 0.0)
        if abs(sb) > 1e-9 and not (ref_site_name and site_name == ref_site_name):
            dis = dis + sb

        return dates, dis

    def _interpolate_insar_at_site(self, lat, lon, ts_data_3d=None):
        """Bilinear interpolation of InSAR time series at exact GNSS site coordinates.

        Parameters
        ----------
        lat : float
            Latitude of the GNSS station in degrees.
        lon : float
            Longitude of the GNSS station in degrees.
        ts_data_3d : np.ndarray or None, optional
            3D InSAR time series data (n_date, n_row, n_col) for the current file.
            If None, uses self.ts_data[0].

        Returns
        -------
        ts_interp : 1D np.ndarray (n_date,)
            Interpolated InSAR displacement time series at the GNSS location.
        (y_f, x_f) : tuple of float
            Floating-point radar coordinates in local (subsetted) data frame.
        """
        if ts_data_3d is None:
            ts_data_3d = self.ts_data[0]

        # Convert geo-coordinates to exact (floating-point) radar coordinates
        y_global, x_global = self.coord.geo2radar(lat, lon, print_msg=False,
                                                   precise=getattr(self, 'precise', False))[0:2]

        # Apply subset offset and multilook scaling (keeping floats for interpolation)
        y_f = y_global
        x_f = x_global
        if self.pix_box is not None:
            y_f -= self.pix_box[1]
            x_f -= self.pix_box[0]
        if self.multilook_num > 1:
            # multilook averaging: center pixel maps to (y - n//2) / n
            half_ml = self.multilook_num // 2
            y_f = (y_f - half_ml) / self.multilook_num
            x_f = (x_f - half_ml) / self.multilook_num

        n_rows, n_cols = ts_data_3d.shape[1], ts_data_3d.shape[2]

        # Clamp to valid range
        y_f = max(0.0, min(y_f, n_rows - 1.001))
        x_f = max(0.0, min(x_f, n_cols - 1.001))

        # Integer pixel anchors for bilinear interpolation
        y0 = int(np.floor(y_f))
        x0 = int(np.floor(x_f))
        y1 = min(y0 + 1, n_rows - 1)
        x1 = min(x0 + 1, n_cols - 1)

        wy = y_f - y0  # fractional weight in y
        wx = x_f - x0  # fractional weight in x

        # Extract the 2×2 pixel neighborhood and apply bilinear weights
        # ts_data_3d has shape (n_date, n_row, n_col)
        # We interpolate each date independently
        w00 = (1 - wy) * (1 - wx)
        w01 = (1 - wy) * wx
        w10 = wy * (1 - wx)
        w11 = wy * wx

        # For efficiency, compute once per date using broadcasting
        ts_interp = (w00 * ts_data_3d[:, y0, x0] +
                     w01 * ts_data_3d[:, y0, x1] +
                     w10 * ts_data_3d[:, y1, x0] +
                     w11 * ts_data_3d[:, y1, x1])

        return ts_interp, (y_f, x_f)


    def _load_or_build_gnss_cache(self):
        """Build GNSS data cache, reusing a disk cache when the InSAR file
        has not changed (avoids re-downloading and re-computing on re-run).

        Populates:
          self._gnss_ts_cache      - {(site, ref_key): (dates, dis)}
        """
        import pickle

        self._gnss_ts_cache = {}

        if (not self.disp_gnss
                or not hasattr(self, 'gnss_site_names')
                or len(self.gnss_site_names) == 0):
            return

        # Disk cache path: next to the first InSAR file
        cache_dir = os.path.dirname(os.path.abspath(self.file[0]))
        cache_file = os.path.join(cache_dir, '.gnss_cache.pkl')

        # Try loading from disk (invalidate if InSAR file is newer)
        if os.path.exists(cache_file):
            try:
                cache_mtime = os.path.getmtime(cache_file)
                insar_mtime = os.path.getmtime(self.file[0])
                if cache_mtime >= insar_mtime:
                    with open(cache_file, 'rb') as fh:
                        data = pickle.load(fh)
                    self._gnss_ts_cache = data.get('gnss_ts', {})
                    vprint(f'Loaded GNSS cache from disk '
                           f'({len(self._gnss_ts_cache)} entries)')
                    return
                else:
                    vprint('Disk GNSS cache is stale, rebuilding...')
            except Exception as e:
                vprint(f'Failed to load GNSS disk cache: {e}, rebuilding...')

        # Build cache: pre-compute GNSS time series with both default
        # spatial reference (for click handler) and with eff_ref_site
        # (for per-station bias), all in one pass.
        eff_ref_site = getattr(self, 'eff_ref_site', None)

        vprint(f'Pre-computing GNSS time series for '
               f'{len(self.gnss_site_names)} stations...')
        # The default spatial reference for the click handler already uses
        # self.ref_gnss_site or self.eff_ref_site.  If that equals eff_ref_site,
        # the explicit ref=eff_ref_site call is redundant -- skip it.
        default_ref = (self.ref_gnss_site
                       or getattr(self, 'eff_ref_site', None)
                       or '__default__')
        skip_explicit = (default_ref == eff_ref_site)

        for i, site_name in enumerate(self.gnss_site_names):
            try:
                # Compute with default spatial reference
                key_default = (site_name, '__default__')
                if key_default not in self._gnss_ts_cache:
                    vprint(f'  [{i+1}/{len(self.gnss_site_names)}] {site_name} (default ref)...')
                    dates, dis = self._get_gnss_timeseries(site_name)
                    self._gnss_ts_cache[key_default] = (dates, dis)

                # Compute with explicit eff_ref_site (for median bias)
                if eff_ref_site and eff_ref_site != site_name:
                    key_ref = (site_name, eff_ref_site)
                    if skip_explicit:
                        # Default ref already equals eff_ref_site -- reuse
                        if key_ref not in self._gnss_ts_cache:
                            self._gnss_ts_cache[key_ref] = (dates, dis)
                    elif key_ref not in self._gnss_ts_cache:
                        vprint(f'  [{i+1}/{len(self.gnss_site_names)}] '
                               f'{site_name} (ref={eff_ref_site})...')
                        dates2, dis2 = self._get_gnss_timeseries(
                            site_name, ignore_ref_site=False,
                            _force_ref_site=eff_ref_site)
                        self._gnss_ts_cache[key_ref] = (dates2, dis2)
            except Exception as e:
                vprint(f'  [{i+1}/{len(self.gnss_site_names)}] '
                       f'failed: {site_name} - {e}')

        vprint(f'GNSS cache built ({len(self._gnss_ts_cache)} entries).')

        # Save to disk for next run (bias-free GNSS data only;
        # per-station biases depend on InSAR data and are recomputed each run)
        try:
            cache_data = {
                'gnss_ts': self._gnss_ts_cache,
            }
            with open(cache_file, 'wb') as fh:
                pickle.dump(cache_data, fh)
            vprint(f'Saved GNSS cache to {cache_file}')
        except Exception as e:
            vprint(f'Failed to save GNSS disk cache: {e}')

    def _compute_per_station_biases(self):
        """Compute per-station median(InSAR - GNSS) bias for each GNSS station.

        For each station, interpolates InSAR at the station location,
        compares with the double-differenced GNSS time series across all
        valid epochs, and stores the per-station median bias.

        This replaces the old global median-of-medians approach with
        station-specific biases, since GNSS and InSAR measurement noise
        cannot be uniformly removed across stations. Each double-differenced
        GNSS station has ONE uniform bias subtracted across all time periods.

        Populates:
          self.gnss_station_biases - dict {site_name: bias_value}
        """
        self.gnss_station_biases = {}

        if (not hasattr(self, 'gnss_site_names')
                or len(self.gnss_site_names) == 0
                or not hasattr(self, 'ts_data')):
            return

        eff_ref_site = getattr(self, 'eff_ref_site', None)

        vprint(f'Computing per-station GNSS-InSAR biases for '
               f'{len(self.gnss_site_names)} stations...')

        for site_name, site_lat, site_lon in zip(
                self.gnss_site_names, self.gnss_site_lats, self.gnss_site_lons):

            # Reference station: bias is zero (already force-zeroed)
            if site_name == eff_ref_site:
                self.gnss_station_biases[site_name] = 0.0
                continue

            try:
                # Get GNSS data (double-differenced, bias-free)
                cache_key = (site_name, eff_ref_site)
                if (hasattr(self, '_gnss_ts_cache')
                        and cache_key in self._gnss_ts_cache):
                    dates_gnss, dis_gnss = self._gnss_ts_cache[cache_key]
                else:
                    dates_gnss, dis_gnss = self._get_gnss_timeseries(
                        site_name, ignore_ref_site=False,
                        _force_ref_site=eff_ref_site)

                if len(dates_gnss) == 0 or len(dis_gnss) == 0:
                    self.gnss_station_biases[site_name] = 0.0
                    continue

                # Interpolate InSAR at station location
                ts_insar, _ = self._interpolate_insar_at_site(site_lat, site_lon)
                if np.all(np.isnan(ts_insar)):
                    self.gnss_station_biases[site_name] = 0.0
                    vprint(f'  {site_name}: InSAR all NaN, bias = 0')
                    continue

                # Collect matched InSAR-GNSS pairs across all valid epochs.
                # For each InSAR date, find the closest GNSS observation
                # within +-7 days and compute InSAR - GNSS.
                diffs = []
                for i_date, insar_val in zip(self.dates, ts_insar):
                    if np.isnan(insar_val):
                        continue
                    day_diffs = np.abs(np.array(
                        [(d - i_date).total_seconds() / 86400.0
                         for d in dates_gnss]))
                    closest_idx = int(np.argmin(day_diffs))
                    if day_diffs[closest_idx] <= 7:
                        gnss_val = dis_gnss[closest_idx]
                        if not np.isnan(gnss_val):
                            diffs.append(insar_val - gnss_val)

                if len(diffs) >= 3:
                    bias = float(np.median(diffs))
                    self.gnss_station_biases[site_name] = bias
                    vprint(f'  {site_name}: bias = {bias:+.3f} {self.disp_unit} '
                           f'(from {len(diffs)} epochs)')
                else:
                    self.gnss_station_biases[site_name] = 0.0
                    vprint(f'  {site_name}: insufficient epochs ({len(diffs)}), '
                           f'bias = 0')
            except Exception as e:
                self.gnss_station_biases[site_name] = 0.0
                vprint(f'  {site_name}: bias computation failed - {e}')

    def plot(self):
        # read 3D time-series
        self.ts_data, self.mask = read_timeseries_data(self)[0:2]

        # ensure attributes needed by view.plot_slice exist
        self._ensure_view_attrs()

        # Collect GNSS station info BEFORE map rendering so per-station biases
        # can be applied to the displayed data (both map and timeseries).
        self._get_gnss_stations_info()
        self._load_or_build_gnss_cache()
        if getattr(self, 'gnss_median_remove', False):
            self._compute_per_station_biases()
        if hasattr(self, 'gnss_station_biases') and any(
                abs(v) > 1e-9 for v in self.gnss_station_biases.values()):
            vprint('Per-station GNSS biases applied '
                   '(GNSS corrected to match InSAR at each station)')

        # Figure 1 - Cumulative Displacement Map
        if not self.figsize_img:
            if self.geo_box and self.fig_coord == 'geo':
                w, n, e, s = self.geo_box
                ds_shape = (e - w, n - s)
            else:
                ds_shape = self.ts_data[0].shape[-2:]
            self.figsize_img = pp.auto_figure_size(
                ds_shape=ds_shape,
                disp_cbar=True,
                disp_slider=True,
                print_msg=False)
        vprint(f'create figure for map in size of [{self.figsize_img[0]:.1f}, {self.figsize_img[1]:.1f}]')
        subplot_kw = dict(projection=self.map_proj_obj) if self.map_proj_obj is not None else {}
        self.fig_img, self.ax_img = plt.subplots(figsize=self.figsize_img, subplot_kw=subplot_kw)

        # Figure 1 - Axes 1 - Displacement Map
        img_data = np.array(self.ts_data[0][self.idx, :, :])
        img_data[self.mask == 0] = np.nan
        self.plot_init_image(img_data)

        # Figure 1 - Axes 2 - Time Slider
        self.plot_init_time_slider(init_idx=self.idx, ref_idx=self.ref_idx)
        self.tslider.on_changed(self.update_time_slider)

        # Figure 2 - Time Series Displacement - Point
        vprint(f'create figure for point in size of [{self.figsize_pts[0]:.1f}, {self.figsize_pts[1]:.1f}]')
        self.fig_pts, self.ax_pts = plt.subplots(num=self.figname_pts, figsize=self.figsize_pts)
        if self.yx:
            d_ts, m_strs = self.plot_point_timeseries(self.yx)

            # save figures and data to files
            if self.save_fig:
                save_ts_data_and_plot(self.yx, d_ts, m_strs, self)

        # Final linking of the canvas to the plots.
        self.fig_img.canvas.mpl_connect('button_press_event', self.update_point_timeseries)
        self.fig_img.canvas.mpl_connect('key_press_event', self.update_image)
        if self.disp_fig:
            vprint('showing ...')
            msg = '\n------------------------------------------------------------------------'
            msg += '\nTo scroll through the image sequence:'
            msg += '\n1) Move the slider, OR'
            msg += '\n2) Press left or right arrow key (if not responding, click the image and try again).'
            msg += '\n------------------------------------------------------------------------'
            vprint(msg)

            # --no-show-map option
            # requires --yx/lalo input
            if self.yx and not self.disp_fig_img:
                plt.close(self.fig_img)

            plt.show()
        return


    ##---------- event functions
    def update_point_timeseries(self, event):
        """Event function to get y/x from button press.
        If the click is near a GNSS station, also retrieve its time series
        and use the station's exact coordinates for the InSAR pixel.
        """
        from pyproj import CRS, Transformer

        if event.inaxes != self.ax_img:
            return

        if self.fig_coord == 'geo':
            gnss_data = None
            click_x, click_y = event.xdata, event.ydata

            # If map uses UTM meters, convert the click coordinates to lat/lon
            if (self.coord_unit.startswith('meter')
                    and self.disp_gnss
                    and hasattr(self, 'gnss_site_lats')
                    and self.gnss_site_lats.size > 0):

                utm_zone = self.atr.get('UTM_ZONE', None)
                if utm_zone:
                    zone_num = int(''.join(filter(str.isdigit, utm_zone)))
                    hemisphere = 'north' if utm_zone[-1].upper() == 'N' else 'south'
                    utm_crs = CRS.from_proj4(
                        f'+proj=utm +zone={zone_num} +{hemisphere} '
                        '+ellps=WGS84 +datum=WGS84 +units=m +no_defs'
                    )
                    wgs84 = CRS.from_epsg(4326)
                    transformer = Transformer.from_crs(utm_crs, wgs84, always_xy=True)
                    click_lon, click_lat = transformer.transform(click_x, click_y)
                else:
                    click_lon, click_lat = None, None
            else:
                click_lon, click_lat = click_x, click_y

            # Check for a nearby GNSS station
            gnss_site_coords = None   # (lat, lon) of the selected station
            if (self.disp_gnss
                    and hasattr(self, 'gnss_site_lats')
                    and self.gnss_site_lats.size > 0
                    and click_lon is not None):

                dist = np.sqrt(
                    (self.gnss_site_lons - click_lon) ** 2 +
                    (self.gnss_site_lats - click_lat) ** 2
                )
                tol_deg = 0.005
                if np.any(dist < tol_deg):
                    idx = np.argmin(dist)
                    site = self.gnss_site_names[idx]
                    # Save the exact lat/lon of the station
                    gnss_site_coords = (self.gnss_site_lats[idx], self.gnss_site_lons[idx])

                    if getattr(self, 'gnss_component', None):
                        vprint(f'Clicked near GNSS station: {site}')
                        try:
                            dates_gnss, dis_gnss = self._get_gnss_timeseries(site)
                            gnss_data = {
                                'dates': dates_gnss,
                                'dis': dis_gnss,
                                'label': f'GNSS {site}'
                            }
                        except Exception as e:
                            vprint(f'Failed to get GNSS data for {site}: {e}')
                            gnss_data = None
                    else:
                        vprint(f'Clicked near GNSS station {site}, '
                            'but --gnss-comp is not set.')

            # Convert to radar coordinates: if a GNSS station was selected, use its
            # exact lat/lon; otherwise use the original click position.
            if gnss_site_coords is not None:
                lat, lon = gnss_site_coords
            else:
                # Use the original coordinates (in map units) for geo2radar
                lat, lon = event.ydata, event.xdata

            precise = getattr(self, 'precise', False) or (gnss_site_coords is not None)
            y, x = self.coord.geo2radar(lat, lon, print_msg=False, precise=precise)[0:2]

        else:
            # Radar coordinate view (no GNSS overlay)
            y, x = int(event.ydata + 0.5), int(event.xdata + 0.5)
            gnss_data = None

        self.plot_point_timeseries((y, x), gnss_data=gnss_data)
        return


    def update_image(self, event):
        """Slide images with left/right key on keyboard"""
        if event.inaxes and event.inaxes.figure == self.fig_img:
            idx = None
            if event.key == 'left':
                idx = max(self.idx - 1, 0)
            elif event.key == 'right':
                idx = min(self.idx + 1, self.num_date - 1)

            if idx is not None and idx != self.idx:
                # update title
                disp_date = self.dates[idx].strftime(self.disp_date_format)
                sub_title = f'N = {idx}, Time = {disp_date}'
                self.ax_img.set_title(sub_title, fontsize=self.font_size)

                # read data
                data_img = np.array(self.ts_data[0][idx, :, :])
                data_img[self.mask == 0] = np.nan
                if self.wrap:
                    if self.disp_unit_img == 'radian':
                        data_img *= self.range2phase
                    data_img = ut.wrap(data_img, wrap_range=self.wrap_range)

                # update
                self.tslider.eventson = False
                self.tslider.set_val(idx)
                self.tslider.eventson = True
                self.img.set_data(data_img)       # update image
                self.idx = idx
                self.fig_img.canvas.draw_idle()
                self.fig_img.canvas.flush_events()
        return


    def update_time_slider(self, val):
        """Update Displacement Map using Slider"""
        self.idx = self.tslider.val

        # update title
        disp_date = self.dates[self.idx].strftime(self.disp_date_format)
        sub_title = f'N = {self.idx}, Time = {disp_date}'
        self.ax_img.set_title(sub_title, fontsize=self.font_size)

        # read/update 2D image data
        data_img = np.array(self.ts_data[0][self.idx, :, :])
        data_img[self.mask == 0] = np.nan
        if self.wrap:
            if self.disp_unit_img == 'radian':
                data_img *= self.range2phase
            data_img = ut.wrap(data_img, wrap_range=self.wrap_range)
        self.img.set_data(data_img)

        # update figure
        self.fig_img.canvas.draw_idle()
        
        return


    ##---------- plot functions
    def plot_init_image(self, img_data):
        """Plot the initial 2D image."""
        # prepare data
        if self.wrap:
            if self.disp_unit_img == 'radian':
                img_data *= self.range2phase
            img_data = ut.wrap(img_data, wrap_range=self.wrap_range)

        # Title and Axis Label
        self.disp_date_format = ptime.get_compact_isoformat(self.date_list[0])
        disp_date = self.dates[self.idx].strftime(self.disp_date_format)
        self.fig_title = f'N = {self.idx}, Time = {disp_date}'

        # Initial Pixel of interest
        self.pts_yx = None
        self.pts_lalo = None
        if self.yx and self.yx != self.ref_yx:
            self.pts_yx = np.array(self.yx).reshape(-1, 2)
            if self.lalo:
                self.pts_lalo = np.array(self.lalo).reshape(-1, 2)

        # call view.py to plot
        self.img, self.cbar_img = view.plot_slice(self.ax_img, img_data, self.atr, self)[2:4]
        self.fig_img.canvas.manager.set_window_title(self.figname_img)
        self.fig_img.tight_layout(rect=(0, 0.16, 1, 0.97))

        return self.img, self.cbar_img


    def plot_init_time_slider(self, init_idx=-1, ref_idx=None):
        """Plot the initial slider."""
        # initiate axes
        #self.fig_img.subplots_adjust(bottom=0.16)
        self.ax_tslider = self.fig_img.add_axes([0.125, 0.05, 0.75, 0.03])

        # plot slider
        self.tslider = widgets.Slider(
            ax=self.ax_tslider,
            label='Image',
            valinit=init_idx,
            valmin=0,
            valmax=self.num_date-1,
            valstep=1)

        # plot reference date:
        # as a gray dot on the slider AND
        # as x-axis label
        if ref_idx is not None:
            self.tslider.ax.scatter(ref_idx, 0.5, s=8**2, marker='o', color='gray', edgecolors='w')
            disp_date = self.dates[ref_idx].strftime(self.disp_date_format)
            self.ax_tslider.set_title(f'Reference: N = {ref_idx}, Time = {disp_date}', fontsize=self.font_size)

        return self.tslider


    def plot_point_timeseries(self, yx, gnss_data=None):
        """Plot point displacement time-series at pixel [y, x]
        Parameters: yx     : list of 2 int
                    gnss_data : dict with 'dates', 'dis', 'label', or None
        Returns:    ts_dis : 1D np.array in size of (num_date,) for the 1st file
                    m_strs : list of strings for the est. time func. param. for the 1st file
        """
        ax = self.ax_pts
        ax.cla()

        # plot scatter in different size for different files
        num_file = len(self.ts_data)
        if   num_file <= 2: ms_step = 4
        elif num_file == 3: ms_step = 3
        elif num_file == 4: ms_step = 2
        elif num_file >= 5: ms_step = 1

        # get local Y/X coord for the subsetted and multilooked 3D data cube
        use_interpolation = (gnss_data is not None
                             and isinstance(yx[0], (int, float))
                             and isinstance(yx[1], (int, float)))

        if use_interpolation:
            # GNSS sub-pixel: keep floating-point coordinates for bilinear interpolation
            y_f, x_f = yx[0], yx[1]
            if self.pix_box is not None:
                y_f -= self.pix_box[1]
                x_f -= self.pix_box[0]
            if self.multilook_num > 1:
                half_ml = self.multilook_num // 2
                y_f = (y_f - half_ml) / self.multilook_num
                x_f = (x_f - half_ml) / self.multilook_num
            y, x = y_f, x_f  # keep floats for interpolation
            y_int, x_int = int(np.round(y_f)), int(np.round(x_f))  # integers for mask indexing
        else:
            (y, x) = subset_and_multilook_yx(yx, self.pix_box, self.multilook_num)
            # ensure integer indices for array subscripting
            y, x = int(round(y)), int(round(x))
            y_int, x_int = y, x

        handles, labels = [], []
        for i in range(num_file-1, -1, -1):
            if use_interpolation:
                ts_dis = _bilinear_interpolate_ts(self.ts_data[i], y, x)
            else:
                ts_dis = self.ts_data[i][:, y, x]

            # fit time func
            m_strs, ts_fit, ts_fit_lim = fit_time_func(
                model=self.model,
                date_list=np.array(self.date_list)[self.ex_flag].tolist(),
                ts_dis=ts_dis[self.ex_flag],
                disp_unit=self.disp_unit,
                G_fit=self.G_fit,
                seconds=self.seconds)

            if self.zero_first:
                off = ts_dis[self.zero_idx]
                ts_dis -= off
                ts_fit -= off

            if self.offset:
                ts_dis += self.offset * (num_file - 1 - i)
                ts_fit += self.offset * (num_file - 1 - i)

            # plot
            if not np.all(np.isnan(ts_dis)):
                ppar = argparse.Namespace()
                ppar.label = self.file_label[i]
                ppar.mfc = f'C{num_file-1-i}' if self.mask[y_int, x_int] != 0 else 'gray'
                ppar.ms = self.marker_size - ms_step * (num_file - 1 - i)
                # use smaller marker size for very long time series
                ppar.ms /= 10 if self.num_date > 1e3 else 1

                handle = self.ts_plot_func(ax, ts_dis, self, ppar)[1]
                handles.append(handle)
                labels.append(ppar.label)

                # plot model prediction
                if self.plot_model:
                    fpar = argparse.Namespace()
                    fpar.linewidth = 3
                    fpar.color = 'C1' if num_file == 1 else ppar.mfc
                    fpar.fontsize = self.font_size

                    if not self.plot_model_conf_int:
                        ts_fit_lim = None

                    plot_ts_fit(
                        ax, ts_fit, self, fpar,
                        m_strs=m_strs,
                        ts_fit_lim=ts_fit_lim,
                    )

        # overlay GNSS time series if available
        if gnss_data is not None and len(gnss_data['dis']) > 0:
            gnss_dates_num = [mdates.date2num(d) for d in gnss_data['dates']]
            ax.plot(gnss_dates_num, gnss_data['dis'],
                    color='darkred', marker='d', ms=3, lw=0,
                    zorder=1, alpha=0.8,
                    label=gnss_data['label'],
                    clip_on=True)
            # update handles/labels for the legend
            handles, labels = ax.get_legend_handles_labels()

            # Compute RMSE: interpolate GNSS to each InSAR date.
            # Also collect overlapping dates for velocity regression.
            insar_dates_kept = np.array(self.dates)[self.ex_flag == 1]
            insar_dis_kept = ts_dis[self.ex_flag == 1]
            gnss_dates_num = np.array(gnss_dates_num)
            gnss_dis_arr = np.array(gnss_data['dis'])
            overlap_insar = []
            overlap_gnss = []
            overlap_dnums = []
            for i_date, i_dis in zip(insar_dates_kept, insar_dis_kept):
                if np.isnan(i_dis):
                    continue
                i_num = mdates.date2num(i_date)
                # find bracketing GNSS dates
                idx = np.searchsorted(gnss_dates_num, i_num)
                if idx == 0 or idx == len(gnss_dates_num):
                    continue
                g0, g1 = gnss_dis_arr[idx-1], gnss_dis_arr[idx]
                if np.isnan(g0) or np.isnan(g1):
                    continue
                t0, t1 = gnss_dates_num[idx-1], gnss_dates_num[idx]
                frac = (i_num - t0) / (t1 - t0)
                interp_gnss = g0 + frac * (g1 - g0)
                overlap_insar.append(i_dis)
                overlap_gnss.append(interp_gnss)
                overlap_dnums.append(i_num)
            # constrain x-axis to InSAR date range (GNSS may extend beyond)
            ax.set_xlim(mdates.date2num(self.dates[0]),
                        mdates.date2num(self.dates[-1]))
            # RMSE
            if len(overlap_gnss) > 1:
                diff_arr = np.array(overlap_insar) - np.array(overlap_gnss)
                rmse_val = np.sqrt(np.nanmean(diff_arr**2))
                rmse_str = f'RMSE = {rmse_val:.2f} {self.disp_unit}'
                ax.text(0.02, 0.02, rmse_str, transform=ax.transAxes,
                        fontsize=self.font_size - 2, verticalalignment='bottom',
                        bbox=dict(boxstyle='round', facecolor='white',
                                  edgecolor='black', alpha=0.8))

            # Compute avg velocities via linear regression over the
            # COMMON overlapping time period (same span, same method).
            if len(overlap_gnss) > 3:
                dnums = np.array(overlap_dnums)
                yrs = (dnums - dnums[0]) / 365.25

                insar_arr = np.array(overlap_insar)
                gnss_arr = np.array(overlap_gnss)

                # Linear regression slope: cov(t, y) / var(t)
                t_mean = np.mean(yrs)
                t_demean = yrs - t_mean
                var_t = np.sum(t_demean ** 2)
                if var_t > 1e-12:
                    insar_vel = (np.sum(t_demean * (insar_arr - np.mean(insar_arr)))
                                 / var_t)
                    gnss_vel = (np.sum(t_demean * (gnss_arr - np.mean(gnss_arr)))
                                / var_t)

                    vel_diff = abs(insar_vel - gnss_vel)
                    vel_str = (
                        f"Vel diff = {vel_diff:.2f} {self.disp_unit}/yr "
                        f"(InSAR={insar_vel:.2f}, GNSS={gnss_vel:.2f})")
                    ax.text(0.02, 0.10, vel_str, transform=ax.transAxes,
                            fontsize=self.font_size - 2, verticalalignment='bottom',
                            bbox=dict(boxstyle='round', facecolor='white',
                                      edgecolor='darkred', alpha=0.8))

        # axis format
        ax.tick_params(which='both', direction='in', labelsize=self.font_size,
                       bottom=True, top=True, left=True, right=True)
        pp.auto_adjust_xaxis_date(ax, self.yearList, fontsize=self.font_size)
        ax.set_ylabel(self.cbar_label, fontsize=self.font_size)
        # auto ylim: use plotted data range when user did not set --ylim explicitly
        if self.ylim_user_set:
            ax.set_ylim(self.ylim)
        else:
            all_y = []
            for line in ax.get_lines():
                all_y.extend(line.get_ydata().tolist())
            for coll in ax.collections:
                offsets = coll.get_offsets()
                if len(offsets) > 0:
                    all_y.extend(offsets[:, 1].tolist())
            if all_y:
                ymin, ymax = np.nanmin(all_y), np.nanmax(all_y)
                ybuffer = max((ymax - ymin) * 0.1, 1e-6)
                ax.set_ylim(ymin - ybuffer, ymax + ybuffer)

        if self.tick_right:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")

        # title
        title = get_point_coord_str(yx[0], yx[1], self.coord, self.lalo_digit,
                                     use_float=use_interpolation)
        title += ' (masked out)' if self.mask[y_int, x_int] == 0 else ''
        if self.disp_title:
            ax.set_title(title, fontsize=self.font_size)

        # legend (unchanged behavior but now includes GNSS if present)
        if len(handles) > 0:
            ax.legend(handles, labels)

        # Print to terminal
        vprint('\n---------------------------------------')
        vprint(title.replace(',',''))    # remove "," in the print out msg for easy reuse in cmd
        float_formatter = lambda x: [float(f'{i:.2f}') for i in x]
        if self.num_date <= 1e3:
            vprint(float_formatter(ts_dis))

        if not np.all(np.isnan(ts_dis)):
            # min/max displacement
            ts_min, ts_max = np.nanmin(ts_dis), np.nanmax(ts_dis)
            vprint(f'time-series range: [{ts_min:.2f}, {ts_max:.2f}] {self.disp_unit}')

            # time func param
            vprint('time function parameters:')
            for m_str in m_strs:
                vprint(f'    {m_str}')

            # update figure
            # use fig.canvas.draw_idel() instead of fig.canvas.draw()
            # reference: https://stackoverflow.com/questions/64789437
            self.fig_pts.canvas.draw_idle()
            self.fig_pts.canvas.flush_events()

        return ts_dis, m_strs