# GIS packages
import rasterio
import shapely as shp

# deepbiosphere packages
import deepbiosphere.Run as run
import deepbiosphere.Utils as utils
import deepbiosphere.Models as mods
from deepbiosphere.Utils import paths
from deepbiosphere.Losses import Loss as losses
from deepbiosphere.Models import Model as mods
import deepbiosphere.NAIP_Utils as naip
import deepbiosphere.Build_Data as build
import deepbiosphere.Make_Maps as maps

# ML + statistics packages
import torch
import numpy as np
import pandas as pd
import geopandas as gpd

# miscellaneous packages
import os
import csv
import copy
import glob
import time
import json
import warnings
import argparse
import multiprocessing
from tqdm import tqdm
from datetime import date
from functools import partial
from typing import List, Tuple
from types import SimpleNamespace


# ---------- Distance functions ---------- #

def euc_distance(bef, aft):
    a = torch.tensor(bef)
    b = torch.tensor(aft)
    # sigmoid tensors
    a = torch.sigmoid(a)
    b = torch.sigmoid(b)
    # euclidean distance
    dist = (a - b).pow(2).sum(0).sqrt()
    dist = dist.numpy()
    return dist

# change in total size of prob mass distro
def pmass_distance(bef, aft):
    a = torch.tensor(bef)
    b = torch.tensor(aft)
    # sigmoid tensors
    a = torch.sigmoid(a)
    b = torch.sigmoid(b)
    dist = (a.sum(axis=0)) / (b.sum(axis=0))
    dist = dist.numpy()
    return dist

# ---------- Types ---------- #

# legal distance functions to use
# when calculating biodiversity change
class Change(utils.FuncEnum, metaclass=utils.MetaEnum):
    EUC_DISTANCE = partial(euc_distance)
    P_MASS  = partial(pmass_distance)

# ---------- Change calculation ---------- #

def calculate_change(oldf, newf, start_year, end_year, change_fn, save_name=None):
    old = rasterio.open(oldf)
    new = rasterio.open(newf)
    # get the interiors of the bounding boxes
    ob = naip.bounding_box_to_polygon(old.bounds)
    nb = naip.bounding_box_to_polygon(new.bounds)
    overlap = shp.geometry.box(*ob.intersection(nb).bounds)
    # generate intersecting window
    xs, ys = overlap.exterior.coords.xy
    oldat = old.read(window=naip.get_window(old.transform, xs, ys))
    # this below relies on default nearest interpolation, could cause some errors
    newdat = new.read(window=naip.get_window(new.transform, xs, ys), out_shape=oldat.shape)
    assert (oldat.shape[0] == newdat.shape[0]), f"shapes don't line up! {oldat.shape} vs {newdat.shape}"
    # A. check types
    change_fn = Change[change_fn]
    dist = change_fn(oldat, newdat)
    # and finally save out to file
    profile = copy.copy(old.profile)
    ntransf = rasterio.transform.from_bounds(*overlap.bounds, dist.shape[1], dist.shape[0])
    profile['count'] = 1
    profile['transform'] = ntransf
    profile['width'] = dist.shape[1]
    profile ['height'] = dist.shape[0]
    save_dir = oldf.rsplit('/', 1)[0]
    if save_name is None:
        save_name = oldf.split('/')[-1].rsplit('_', 3)[0]
    fname = f"{save_dir}/{save_name}_{start_year}to{end_year}_change.tif"
    dist = np.expand_dims(dist, axis=0)
    with rasterio.open(fname, 'w', **profile) as dst:
        dst.write(dist, range(1,2), masked=False)
        dst.descriptions = ['change']
    return fname


def calculate_change_parallel(procid, lock, overlapping_ras, rasters_start, rasters_end, start_year, end_year, change_fn):

    with lock:
        prog = tqdm(total=len(overlapping_ras), desc=f"Change tiff group #{procid}", unit=' tiffs', position=procid)
    for i, key in enumerate(overlapping_ras):
        calculate_change(rasters_start[key], rasters_end[key], start_year, end_year, change_fn)
        with lock:
            prog.update(1)
    with lock:
        prog.close()


# ---------- Driver code ---------- #

# driver code that takes raster list from two timepionts and drives below
# also add in taking differences and whatnot
# Basically, takes the difference as seen before
def predict_raster_intime(pred_outline : gpd.GeoDataFrame,
                        start_dir : str,
                        end_dir : str,
                        pred_types : List[str],
                        alpha_type : str,
                        cfg : SimpleNamespace,
                        epoch : int,
                        band : int,
                        state : str,
                        start_year : int,
                        end_year : int,
                        device: int,
                        n_processes : int, # whether to use parallel or not
                        batch_size : int,
                        pred_res: int, # meter resolution of predictions to make with DBS
                        change_fn = str,
                        sat_res : int = 1.0, # resolution to upsample sat imagery to, in meters
                        impute_climate = True,
                        generate_preds=False,
                        bioclim_train_dir=paths.RASTERS):
    if generate_preds:
        # get climate data first
        clim_rasters = build.get_bioclim_rasters(state=state)
        # 1. predict rasters in start_year
        pred_types = [naip.Prediction[pred_type] for pred_type in pred_types]
        if naip.Prediction.RAW not in pred_types:
            pred_types.insert(0, naip.Prediction.RAW)
        start_ras = maps.predict_rasters_list(pred_outline=pred_outline,
                                    pred_types=pred_types,
                                    parent_dir=start_dir,
                                    alpha_type = alpha_type,
                                    cfg=cfg,
                                    epoch=epoch,
                                    band=band,
                                    state=state,
                                    pred_year=start_year,
                                    device=device,
                                    n_processes=n_processes, # whether to use parallel or not
                                    batch_size=batch_size,
                                    pred_res=pred_res, # meter resolution of predictions to make with DBS
                                    sat_res=sat_res, # resolution to upsample sat imagery to
                                    impute_climate=impute_climate,
                                    clim_rasters = clim_rasters,
                                    train_dir=bioclim_train_dir)
        # 2. predict rasters in end_year
        end_ras = maps.predict_rasters_list(pred_outline=pred_outline,
                                    pred_types=pred_types,
                                    parent_dir=end_dir,
                                    alpha_type = alpha_type,
                                    cfg=cfg,
                                    epoch=epoch,
                                    band=band,
                                    state=state,
                                    pred_year=end_year,
                                    device=device,
                                    n_processes=n_processes, # whether to use parallel or not
                                    batch_size=batch_size,
                                    pred_res=pred_res, # meter resolution of predictions to make with DBS
                                    sat_res=sat_res, # resolution to upsample sat imagery to
                                    impute_climate=impute_climate,
                                    clim_rasters = clim_rasters,
                                    train_dir=bioclim_train_dir)

    # 3. calculate change between the years
    start_ras = glob.glob(f"{paths.RASTERS}{start_dir}/{pred_res}m_{start_year}_{band}_{cfg.exp_id}_{epoch}/*/*raw*.tif")
    end_ras = glob.glob(f"{paths.RASTERS}{end_dir}/{pred_res}m_{end_year}_{band}_{cfg.exp_id}_{epoch}/*/*raw*.tif")
    rasters_start = {'_'.join(f.split('/')[-1].split('_')[:4]) : f for f in start_ras}
    rasters_end = {'_'.join(f.split('/')[-1].split('_')[:4]) : f for f in end_ras}
    # only keep rasters that match
    overlapping_ras = list(set(rasters_start.keys()) & set(rasters_end.keys()))

    if n_processes > 1:
        # set up parallel
        ras_pars = utils.partition(overlapping_ras, n_processes)
        lock = multiprocessing.Manager().Lock()
        pool =  multiprocessing.Pool(n_processes)

        res_async = [pool.apply_async(calculate_change_parallel, args=(i, lock, ras, rasters_start, rasters_end, start_year, end_year, change_fn)) for i, ras in enumerate(ras_pars)]
        res_files = [r.get() for r in res_async]
        pool.close()
        pool.join()

    else:
        for key in tqdm(overlapping_ras, total=len(overlapping_ras), desc=f"change tiffs", unit=' tiffs'):
            calculate_change(rasters_start[key], rasters_end[key], start_year, end_year, change_fn)

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument('--shape_pth', type=str, help='relative path to location of shapely file storing tiffs to predict with', required=True)
    args.add_argument('--pred_types', nargs = '+', help='What type/s of predictions to make', choices = naip.Prediction.valid(), default=['RAW'])
    args.add_argument('--start_dir', type=str, help='what parent directory start year tiffs are saved in. Full path expansion is: {paths.RASTERS}{start_dir}/{file_name}', required=True)
    args.add_argument('--end_dir', type=str, help='what parent directory end year tffs are saved in, if different. Full path expansion is: {paths.RASTERS}{end_dir}/{file_name}', required=True)
    args.add_argument('--start_year', type=int, help='What year of imagery to make before prediction on', default=2012)
    args.add_argument('--end_year', type=int, help='What year of imagery to make after prediction on', default=2014)
    args.add_argument('--exp_id', type=str, help='Experiment ID for model to use for mapmaking', required=True)
    args.add_argument('--band', type=str, help='Band which model to use for mapmaking was trained on', required=True)
    args.add_argument('--loss', type=str, help='Loss function used to train mapmaking model', required=True, choices=losses.valid())
    args.add_argument('--architecture', type=str, help='Architecture of mapmaking model', required=True, choices=mods.valid())
    args.add_argument('--epoch', type=int, help='what model epoch to use for making maps', required=True)
    args.add_argument('--batch_size', type=int, help='what size batch to use for making map inference', required=True)
    args.add_argument('--pred_resolution', type=int, help='what meter resolution to make map', default=utils.IMG_SIZE)
    args.add_argument('--change_fn', type=str, help='what distance function to use to calculate change', choices = Change.valid(), default='EUC_DISTANCE')
    args.add_argument('--sat_resolution', type=float, help='what meter resolution to up / downsample base imagery to', default=1.0)
    args.add_argument('--device', type=int, help="Which CUDA device to use. Set -1 for CPU", default=-1)
    args.add_argument('--state', type=str, help='What state predictions are being made int', default='ca')
    args.add_argument('--alpha_type', type = str, help='What type of alpha prediction to make', choices = naip.Alpha.valid(), default='SUM')
    args.add_argument('--bioclim_train_dir', type=str, help='Location of bioclim rasters used in training if different from inference time', default=paths.RASTERS)
    args.add_argument('--processes', type=int, help="How many worker processes to use for mapmaking", default=1)
    args.add_argument('--impute_climate', action='store_true', help="whether to impute the climate for locations with no bioclim coverage")
    args.add_argument('--generate_preds', action='store_true', help="whether to generate predictions for both years if not already done")
    args, _ = args.parse_known_args()
    if args.processes > 1:
        multiprocessing.set_start_method('spawn')
    # load config
    cnn = {
        'exp_id': args.exp_id,
        'band' : args.band,
        'loss': args.loss,
        'model': args.architecture
    }
    cfg = run.load_config(**cnn)
    # read in polygon
    bound_shp = gpd.read_file(f"{paths.SHPFILES}{args.shape_pth}")
    predict_raster_intime(pred_outline = bound_shp,
                        start_dir  = args.start_dir,
                        end_dir = args.end_dir,
                        pred_types = args.pred_types,
                        alpha_type = args.alpha_type,
                        cfg  = cfg,
                        epoch = args.epoch,
                        band = args.band,
                        state = args.state,
                        start_year = args.start_year,
                        end_year = args.end_year,
                        device = args.device,
                        n_processes = args.processes,
                        batch_size = args.batch_size,
                        pred_res = args.pred_resolution,
                        change_fn = args.change_fn,
                        sat_res = args.sat_resolution,
                        impute_climate = args.impute_climate,
                        generate_preds = args.generate_preds,
                        bioclim_train_dir = args.bioclim_train_dir)

