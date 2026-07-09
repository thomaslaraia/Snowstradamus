from scripts.imports import *

import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rioxarray

from rasterio.enums import Resampling
from pyproj import Transformer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize


class Tee:
    """Write to console + file at the same time for stdout/stderr."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


# ---------------------------------------------------------------------
# ARGUMENTS
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument("-E", type=int, default=80)

parser.add_argument(
    "-aspect",
    type=float,
    default=0.0,
    help=(
        "0 disables low-variability aspect split. If >0, compute topography "
        "on grouped cells and make north/south OOB summaries using only rows "
        "where aspect_variability <= this value."
    )
)

parser.add_argument(
    "-aspect_high",
    type=float,
    default=0.0,
    help=(
        "0 disables high-aspect-variability analysis. If >0, compute OOB "
        "summaries using rows where aspect_variability > this value, without "
        "considering north/south aspect dominance."
    )
)

args = parser.parse_args()

E = args.E

ASPECT_VARIABILITY_THRESHOLD = float(args.aspect)
HIGH_ASPECT_VARIABILITY_THRESHOLD = float(args.aspect_high)

if ASPECT_VARIABILITY_THRESHOLD < 0:
    raise ValueError(
        "-aspect must be >= 0. Use 0 to disable the low-variability aspect split."
    )

if HIGH_ASPECT_VARIABILITY_THRESHOLD < 0:
    raise ValueError(
        "-aspect_high must be >= 0. Use 0 to disable the high-variability analysis."
    )

USE_ASPECT_SPLIT = ASPECT_VARIABILITY_THRESHOLD > 0
USE_HIGH_ASPECT_VARIABILITY = HIGH_ASPECT_VARIABILITY_THRESHOLD > 0

# Topography is needed if either analysis is active
USE_TOPOGRAPHY = USE_ASPECT_SPLIT or USE_HIGH_ASPECT_VARIABILITY


suffix = "nw_wc_nolof_aspect_test"
BIN_W_PARAM = 0

remove_cams = []
num_cameras = 18 - len(remove_cams)

ASPECT_COL = "north_facing_fraction"
ASPECT_VARIABILITY_COL = "aspect_variability"
ASPECT_GROUP_COL = "aspect_group"

ASPECT_GROUPS = (
    "north_dominated",
    "south_dominated",
)

HIGH_ASPECT_GROUP = "high_aspect_variability"


# ---------------------------------------------------------------------
# TOPOGRAPHY SETTINGS
# ---------------------------------------------------------------------
MASKING_FOLDER = Path("../scratch/data/landsat_masking/")

FOREST_LANDCOVER_MIN = 111
FOREST_LANDCOVER_MAX = 126

CELL_SIZE_M = 1000
HALF_CELL_M = CELL_SIZE_M / 2

# Tied to -E, so -E 80 means +/-80 m elevation mask.
ELEVATION_TOLERANCE_M = E

LAT_COL = "lat"
LON_COL = "lon"
ALT_COL = "altitude"
CAM_COL = "camera"

TOPOGRAPHY_COLS = [
    "north_facing_fraction",
    "south_facing_fraction",
    "aspect_variability",
    "gradient_p16",
    "gradient_p50",
    "gradient_p84",
    "topography_valid_pixel_count",
    "topography_cell_pixel_count",
]

CAM_FILE_CANDIDATES = {
    "sodankyla_full": ["sodankyla_full", "sodankyla"],
    "sodankyla": ["sodankyla", "sodankyla_full"],
    "marcell_MN": ["marcell_MN", "marcell"],
    "marcell": ["marcell", "marcell_MN"],
    "oregon_yp": ["oregon_yp", "oregon"],
    "oregon": ["oregon", "oregon_yp"],
}


# ---------------------------------------------------------------------
# OUTPUT FOLDER + LOGGING SETUP
# ---------------------------------------------------------------------
out_dir = os.path.join(
    ".",
    "bootstrap_images",
    f"{E}m_{suffix}"
)

os.makedirs(out_dir, exist_ok=True)

log_path = os.path.join(
    out_dir,
    f"{E}m_{suffix}_log.txt"
)

log_fh = open(
    log_path,
    "w",
    encoding="utf-8"
)

log_fh.write(
    f"Run started: {datetime.now().isoformat(timespec='seconds')}\n"
)

log_fh.write(f"E = {E}\n")
log_fh.write(f"suffix = {suffix}\n")
log_fh.write(f"BIN_W_PARAM = {BIN_W_PARAM}\n")

log_fh.write(
    f"USE_ASPECT_SPLIT = {USE_ASPECT_SPLIT}\n"
)

log_fh.write(
    f"ASPECT_VARIABILITY_THRESHOLD = "
    f"{ASPECT_VARIABILITY_THRESHOLD}\n"
)

log_fh.write(
    f"USE_HIGH_ASPECT_VARIABILITY = "
    f"{USE_HIGH_ASPECT_VARIABILITY}\n"
)

log_fh.write(
    f"HIGH_ASPECT_VARIABILITY_THRESHOLD = "
    f"{HIGH_ASPECT_VARIABILITY_THRESHOLD}\n"
)

log_fh.write("-" * 60 + "\n\n")
log_fh.flush()

_orig_stdout = sys.stdout
_orig_stderr = sys.stderr

sys.stdout = Tee(
    _orig_stdout,
    log_fh
)

sys.stderr = Tee(
    _orig_stderr,
    log_fh
)


try:

    # -----------------------------------------------------------------
    # LOAD RAW DATA
    # -----------------------------------------------------------------
    df = pd.read_pickle(
        f"dataset_lcforest_noLOF_bin15_th3_"
        f"{E}m_1kmsmallbox_noprior_ta_wc1_v7.pkl"
    )

    df["Eg_strong"] = np.where(
        (df["beam_str"] == "strong") &
        (df["outlier"] == 1),
        df["Eg"],
        np.nan
    )

    df["Ev_strong"] = np.where(
        (df["beam_str"] == "strong") &
        (df["outlier"] == 1),
        df["Ev"],
        np.nan
    )

    df["Eg_weak"] = np.where(
        (df["beam_str"] == "weak") &
        (df["outlier"] == 1),
        df["Eg"],
        np.nan
    )

    df["Ev_weak"] = np.where(
        (df["beam_str"] == "weak") &
        (df["outlier"] == 1),
        df["Ev"],
        np.nan
    )


    # -----------------------------------------------------------------
    # GROUP DATA FIRST
    # -----------------------------------------------------------------
    agg_dict = {
        "pvpg": "mean",
        "pv": "max",
        "pg": "max",
        "Eg_strong": "median",
        "Ev_strong": "median",
        "Eg_weak": "median",
        "Ev_weak": "median",
        "data_quantity": "max",
        "snr": "mean",
        "FSC": "mean",
        "TreeSnow": "mean",
        "layer_flag": "mean",
        "file_index": "mean",
        "msw_flag": "mean",
        "pv_ratio_mean": "mean",
        "pv_ratio_max": "mean",
        "altitude": "median",
    }

    df_grouped = (
        df
        .groupby(
            [
                "camera",
                "date",
                "lat",
                "lon",
            ]
        )
        .agg(agg_dict)
        .reset_index()
    )

    df_grouped = df_grouped[
        df_grouped["Eg_strong"] >= 0
    ].copy()

    df_grouped["JointSnow"] = (
        df_grouped["FSC"] +
        df_grouped["TreeSnow"]
    )

    df_grouped = df_grouped[
        ~df_grouped["camera"].isin(remove_cams)
    ].copy()

    df_grouped["cell_id"] = (
        df_grouped["camera"].astype(str)
        + "|"
        + df_grouped["date"].astype(str)
        + "|"
        + df_grouped["lat"].round(6).astype(str)
        + "|"
        + df_grouped["lon"].round(6).astype(str)
    )

    df_grouped = df_grouped.reset_index(drop=True)

    print(
        "Grouped dataframe shape:",
        df_grouped.shape
    )


    # -----------------------------------------------------------------
    # TOPOGRAPHY HELPERS
    # -----------------------------------------------------------------
    def local_utm_crs(lon, lat):

        zone = int(
            (lon + 180) // 6
        ) + 1

        epsg = (
            32600 + zone
            if lat >= 0
            else 32700 + zone
        )

        return f"EPSG:{epsg}"


    def find_topography_files(
        masking_folder,
        cam
    ):

        landcover_fp = None
        elevation_fp = None
        gradient_fp = None
        aspect_fp = None

        cam_candidates = (
            CAM_FILE_CANDIDATES.get(
                str(cam),
                [str(cam)]
            )
        )

        cam_candidates = [
            c.lower()
            for c in cam_candidates
        ]

        for fp in masking_folder.glob("*.tif"):

            name_lower = fp.name.lower()

            if not any(
                c in name_lower
                for c in cam_candidates
            ):
                continue

            if "landcover" in name_lower:
                landcover_fp = fp

            elif "alos_aw3d30_elevation" in name_lower:
                elevation_fp = fp

            elif "alos_aw3d30_gradient" in name_lower:
                gradient_fp = fp

            elif "alos_aw3d30_aspect" in name_lower:
                aspect_fp = fp

        return (
            landcover_fp,
            elevation_fp,
            gradient_fp,
            aspect_fp,
        )


    def add_topography_to_grouped_df(df_grouped):

        df_grouped = df_grouped.copy()

        for col in TOPOGRAPHY_COLS:
            df_grouped[col] = np.nan

        df_grouped[
            "topography_valid_pixel_count"
        ] = 0

        df_grouped[
            "topography_cell_pixel_count"
        ] = 0

        for cam, cam_df in df_grouped.groupby(CAM_COL):

            print("\n" + "=" * 80)
            print(
                f"Computing topography for camera: "
                f"{cam}"
            )
            print(
                f"Grouped rows: {len(cam_df)}"
            )

            (
                landcover_fp,
                elevation_fp,
                gradient_fp,
                aspect_fp,
            ) = find_topography_files(
                MASKING_FOLDER,
                cam
            )

            if any(
                x is None
                for x in [
                    landcover_fp,
                    elevation_fp,
                    gradient_fp,
                    aspect_fp,
                ]
            ):

                print(
                    f"Skipping {cam}: "
                    f"missing one or more required rasters"
                )

                print(
                    f"  landcover: "
                    f"{landcover_fp}"
                )

                print(
                    f"  elevation: "
                    f"{elevation_fp}"
                )

                print(
                    f"  gradient:  "
                    f"{gradient_fp}"
                )

                print(
                    f"  aspect:    "
                    f"{aspect_fp}"
                )

                continue

            print(
                f"  landcover: "
                f"{landcover_fp.name}"
            )

            print(
                f"  elevation: "
                f"{elevation_fp.name}"
            )

            print(
                f"  gradient:  "
                f"{gradient_fp.name}"
            )

            print(
                f"  aspect:    "
                f"{aspect_fp.name}"
            )

            elevation_raw = (
                rioxarray
                .open_rasterio(
                    elevation_fp,
                    masked=True
                )
                .squeeze()
            )

            elevation_raw.name = "elevation"

            landcover_raw = (
                rioxarray
                .open_rasterio(
                    landcover_fp,
                    masked=True
                )
                .squeeze()
            )

            landcover_raw.name = "landcover"

            gradient_raw = (
                rioxarray
                .open_rasterio(
                    gradient_fp,
                    masked=True
                )
                .squeeze()
            )

            gradient_raw.name = "gradient"

            aspect_raw = (
                rioxarray
                .open_rasterio(
                    aspect_fp,
                    masked=True
                )
                .squeeze()
            )

            aspect_raw.name = "aspect"

            if elevation_raw.rio.crs is None:

                print(
                    f"Skipping {cam}: "
                    f"elevation raster has no CRS"
                )

                continue

            centre_lon = float(
                cam_df[LON_COL].median()
            )

            centre_lat = float(
                cam_df[LAT_COL].median()
            )

            target_crs = local_utm_crs(
                centre_lon,
                centre_lat
            )

            print(
                f"Original elevation CRS: "
                f"{elevation_raw.rio.crs}"
            )

            print(
                f"Using local metre CRS:  "
                f"{target_crs}"
            )

            elevation = elevation_raw.rio.reproject(
                target_crs,
                resolution=30,
                resampling=Resampling.bilinear
            )

            elevation.name = "elevation"

            landcover = (
                landcover_raw
                .rio
                .reproject_match(
                    elevation,
                    resampling=Resampling.nearest
                )
            )

            landcover.name = "landcover"

            gradient = (
                gradient_raw
                .rio
                .reproject_match(
                    elevation,
                    resampling=Resampling.bilinear
                )
            )

            gradient.name = "gradient"

            aspect = (
                aspect_raw
                .rio
                .reproject_match(
                    elevation,
                    resampling=Resampling.nearest
                )
            )

            aspect.name = "aspect"

            transformer = Transformer.from_crs(
                "EPSG:4326",
                target_crs,
                always_xy=True
            )

            processed = 0
            skipped = 0

            for idx, row in cam_df.iterrows():

                lon = float(row[LON_COL])
                lat = float(row[LAT_COL])
                altitude = float(row[ALT_COL])

                if (
                    not np.isfinite(lon)
                    or not np.isfinite(lat)
                    or not np.isfinite(altitude)
                ):

                    skipped += 1
                    continue

                centre_x, centre_y = (
                    transformer.transform(
                        lon,
                        lat
                    )
                )

                minx = (
                    centre_x
                    - HALF_CELL_M
                )

                maxx = (
                    centre_x
                    + HALF_CELL_M
                )

                miny = (
                    centre_y
                    - HALF_CELL_M
                )

                maxy = (
                    centre_y
                    + HALF_CELL_M
                )

                try:

                    elev_cell = (
                        elevation
                        .rio
                        .clip_box(
                            minx=minx,
                            miny=miny,
                            maxx=maxx,
                            maxy=maxy
                        )
                    )

                    lc_cell = (
                        landcover
                        .rio
                        .clip_box(
                            minx=minx,
                            miny=miny,
                            maxx=maxx,
                            maxy=maxy
                        )
                    )

                    grad_cell = (
                        gradient
                        .rio
                        .clip_box(
                            minx=minx,
                            miny=miny,
                            maxx=maxx,
                            maxy=maxy
                        )
                    )

                    aspect_cell = (
                        aspect
                        .rio
                        .clip_box(
                            minx=minx,
                            miny=miny,
                            maxx=maxx,
                            maxy=maxy
                        )
                    )

                except Exception:

                    skipped += 1
                    continue

                if elev_cell.size == 0:

                    skipped += 1
                    continue

                elev_vals = elev_cell.values
                lc_vals = lc_cell.values
                grad_vals_2d = grad_cell.values
                aspect_vals_2d = aspect_cell.values

                elevation_mask = (
                    np.abs(
                        elev_vals - altitude
                    )
                    <= ELEVATION_TOLERANCE_M
                )

                forest_mask = (
                    (
                        lc_vals
                        >= FOREST_LANDCOVER_MIN
                    )
                    &
                    (
                        lc_vals
                        <= FOREST_LANDCOVER_MAX
                    )
                )

                valid_mask = (
                    elevation_mask
                    &
                    forest_mask
                    &
                    np.isfinite(elev_vals)
                    &
                    np.isfinite(lc_vals)
                )

                cell_pixel_count = int(
                    elev_vals.size
                )

                valid_pixel_count = int(
                    np.count_nonzero(
                        valid_mask
                    )
                )

                grad_vals = (
                    grad_vals_2d[
                        valid_mask
                    ]
                )

                aspect_vals = (
                    aspect_vals_2d[
                        valid_mask
                    ]
                )

                grad_vals = (
                    grad_vals[
                        np.isfinite(
                            grad_vals
                        )
                    ]
                )

                aspect_vals = (
                    aspect_vals[
                        np.isfinite(
                            aspect_vals
                        )
                    ]
                )

                if len(aspect_vals) > 0:

                    aspect_vals = (
                        aspect_vals
                        % 360
                    )

                    north_mask = (
                        (aspect_vals >= 270)
                        |
                        (aspect_vals < 90)
                    )

                    south_mask = (
                        (aspect_vals >= 90)
                        &
                        (aspect_vals < 270)
                    )

                    north_facing_fraction = float(
                        np.mean(
                            north_mask
                        )
                    )

                    south_facing_fraction = float(
                        np.mean(
                            south_mask
                        )
                    )

                    aspect_rad = np.deg2rad(
                        aspect_vals
                    )

                    mean_sin = np.mean(
                        np.sin(
                            aspect_rad
                        )
                    )

                    mean_cos = np.mean(
                        np.cos(
                            aspect_rad
                        )
                    )

                    resultant_length = np.sqrt(
                        mean_sin**2
                        +
                        mean_cos**2
                    )

                    aspect_variability = float(
                        1 - resultant_length
                    )

                else:

                    north_facing_fraction = np.nan
                    south_facing_fraction = np.nan
                    aspect_variability = np.nan

                if len(grad_vals) > 0:

                    gradient_p16 = float(
                        np.nanpercentile(
                            grad_vals,
                            16
                        )
                    )

                    gradient_p50 = float(
                        np.nanpercentile(
                            grad_vals,
                            50
                        )
                    )

                    gradient_p84 = float(
                        np.nanpercentile(
                            grad_vals,
                            84
                        )
                    )

                else:

                    gradient_p16 = np.nan
                    gradient_p50 = np.nan
                    gradient_p84 = np.nan

                df_grouped.at[
                    idx,
                    "north_facing_fraction"
                ] = north_facing_fraction

                df_grouped.at[
                    idx,
                    "south_facing_fraction"
                ] = south_facing_fraction

                df_grouped.at[
                    idx,
                    "aspect_variability"
                ] = aspect_variability

                df_grouped.at[
                    idx,
                    "gradient_p16"
                ] = gradient_p16

                df_grouped.at[
                    idx,
                    "gradient_p50"
                ] = gradient_p50

                df_grouped.at[
                    idx,
                    "gradient_p84"
                ] = gradient_p84

                df_grouped.at[
                    idx,
                    "topography_valid_pixel_count"
                ] = valid_pixel_count

                df_grouped.at[
                    idx,
                    "topography_cell_pixel_count"
                ] = cell_pixel_count

                processed += 1

            print(
                f"Finished topography for camera: "
                f"{cam}"
            )

            print(
                f"  processed: {processed}"
            )

            print(
                f"  skipped:   {skipped}"
            )

        print(
            "\nTopography summary:"
        )

        print(
            df_grouped[
                TOPOGRAPHY_COLS
            ].describe()
        )

        print(
            "\nRows with valid topography:"
        )

        print(
            (
                df_grouped[
                    "topography_valid_pixel_count"
                ]
                > 0
            )
            .value_counts(
                dropna=False
            )
        )

        return df_grouped


    # -----------------------------------------------------------------
    # COMPUTE TOPOGRAPHY IF EITHER TOPOGRAPHY ANALYSIS IS ACTIVE
    # -----------------------------------------------------------------
    if USE_TOPOGRAPHY:

        print(
            "\nTopography analysis enabled."
        )

        print(
            "Computing aspect/gradient summaries on grouped 1 km cells. "
            "Model fitting and filter search still use all grouped data. "
            "Topography filtering is only used for the extra OOB summaries."
        )

        df_grouped = (
            add_topography_to_grouped_df(
                df_grouped
            )
        )

        missing_after_compute = [
            col
            for col in [
                ASPECT_COL,
                ASPECT_VARIABILITY_COL,
            ]
            if col not in df_grouped.columns
        ]

        if missing_after_compute:

            raise KeyError(
                "Topography computation did not produce "
                f"required columns: "
                f"{missing_after_compute}"
            )

        df_grouped[
            ASPECT_GROUP_COL
        ] = pd.Series(
            pd.NA,
            index=df_grouped.index,
            dtype="object"
        )

        df_grouped.loc[
            df_grouped[
                ASPECT_COL
            ] > 0.5,
            ASPECT_GROUP_COL
        ] = "north_dominated"

        df_grouped.loc[
            df_grouped[
                ASPECT_COL
            ] < 0.5,
            ASPECT_GROUP_COL
        ] = "south_dominated"


        if USE_ASPECT_SPLIT:

            print(
                f"\nLow-variability aspect split rows will require "
                f"{ASPECT_VARIABILITY_COL} <= "
                f"{ASPECT_VARIABILITY_THRESHOLD}."
            )

            print(
                "\nAspect group counts before model filtering:"
            )

            print(
                df_grouped[
                    ASPECT_GROUP_COL
                ].value_counts(
                    dropna=False
                )
            )

            print(
                "\nLow aspect variability eligibility count "
                "before model filtering:"
            )

            print(
                (
                    np.isfinite(
                        df_grouped[
                            ASPECT_VARIABILITY_COL
                        ]
                    )
                    &
                    (
                        df_grouped[
                            ASPECT_VARIABILITY_COL
                        ]
                        <= ASPECT_VARIABILITY_THRESHOLD
                    )
                )
                .value_counts(
                    dropna=False
                )
            )


        if USE_HIGH_ASPECT_VARIABILITY:

            print(
                f"\nHigh-variability rows will require "
                f"{ASPECT_VARIABILITY_COL} > "
                f"{HIGH_ASPECT_VARIABILITY_THRESHOLD}."
            )

            print(
                "North/south aspect dominance will not be considered "
                "for this group."
            )

            print(
                "\nHigh aspect variability eligibility count "
                "before model filtering:"
            )

            print(
                (
                    np.isfinite(
                        df_grouped[
                            ASPECT_VARIABILITY_COL
                        ]
                    )
                    &
                    (
                        df_grouped[
                            ASPECT_VARIABILITY_COL
                        ]
                        > HIGH_ASPECT_VARIABILITY_THRESHOLD
                    )
                )
                .value_counts(
                    dropna=False
                )
            )


    # ===========================================================
    # PHASE 2: Bootstrap -> optimize filters -> train model
    # -> predict on OOB cameras
    # ===========================================================

    EG_COL = "Eg_strong"
    EV_COL = "Ev_strong"
    Y_BIN_COL = "JointSnowBinary"

    FRAC_W = 1.0
    N_BOOT = 1000
    N_SPLITS_CV = 5

    RATIO_GRID = np.round(
        np.arange(
            1.05,
            1.30 + 1e-9,
            0.01
        ),
        2
    )

    DQ_GRID = np.arange(
        18,
        37
    )

    TOL_NEAR = 0.003
    RNG = np.random.RandomState(42)


    def base_conditions_opt(df):

        return (
            (
                (df["FSC"] <= 0.005)
                |
                (df["FSC"] >= 0.995)
            )
            &
            (
                (df["TreeSnow"] == 0)
                |
                (df["TreeSnow"] == 1)
            )
            &
            (
                df["Eg_strong"] >= 0
            )
        )


    def base_conditions_boot(df):

        return (
            (df["FSC"] >= 0.0)
            &
            (df["FSC"] <= 1.0)
            &
            (df["Eg_strong"] >= 0)
        )


    def apply_filters_for_search(
        df,
        ratio_thresh,
        dq_thresh
    ):

        cond = (
            base_conditions_opt(df)
            &
            (
                (
                    df["Eg_strong"]
                    /
                    df["Eg_weak"]
                )
                >= ratio_thresh
            )
            &
            (
                df["data_quantity"]
                >= dq_thresh
            )
        )

        out = df.loc[cond].copy()

        out["JointSnow"] = (
            out["FSC"]
            +
            out["TreeSnow"]
        )

        out["JointSnowRounded"] = (
            np
            .round(
                out["JointSnow"]
            )
            .astype(int)
        )

        return out


    def apply_filters_for_boot(
        df,
        ratio_thresh,
        dq_thresh
    ):

        cond = (
            base_conditions_boot(df)
            &
            (
                (
                    df["Eg_strong"]
                    /
                    df["Eg_weak"]
                )
                >= ratio_thresh
            )
            &
            (
                df["data_quantity"]
                >= dq_thresh
            )
        )

        out = df.loc[cond].copy()

        out["JointSnow"] = (
            out["FSC"]
            +
            out["TreeSnow"]
        )

        out["JointSnowBinary"] = (
            out["JointSnow"]
            .apply(
                lambda x:
                    1
                    if x >= 1
                    else x
            )
            .astype(float)
        )

        return out


    def assign_folds_by_camera(
        df,
        n_splits=5
    ):

        counts = (
            df["camera"]
            .value_counts()
        )

        cams_sorted = (
            counts.index.tolist()
        )

        cam2fold = {
            cam: (
                i % n_splits
            )
            for i, cam
            in enumerate(
                cams_sorted
            )
        }

        return (
            df["camera"]
            .map(cam2fold)
            .to_numpy()
        )


    def cv_multinomial_metrics(
        df,
        df2,
        features=(
            "Eg_strong",
            "Ev_strong"
        ),
        n_splits=5
    ):

        if (
            df.shape[0] == 0
            or
            df["JointSnowRounded"].nunique() < 2
        ):

            return (
                np.nan,
                None,
                np.nan,
                np.nan,
                None,
                np.nan,
            )

        n_unique_cams = (
            df["camera"].nunique()
        )

        n_splits_eff = max(
            2,
            min(
                n_splits,
                n_unique_cams
            )
        )

        grp = assign_folds_by_camera(
            df,
            n_splits_eff
        )

        X = (
            df
            .loc[
                :,
                list(features)
            ]
            .to_numpy()
        )

        y = (
            df[
                "JointSnowRounded"
            ]
            .to_numpy()
        )

        X2 = (
            df2
            .loc[
                :,
                list(features)
            ]
            .to_numpy()
        )

        y2 = (
            df2[
                "JointSnowRounded"
            ]
            .to_numpy()
        )

        valid = (
            np.isfinite(X).all(axis=1)
            &
            np.isfinite(y)
        )

        if not np.any(valid):

            return (
                np.nan,
                None,
                np.nan,
                np.nan,
                None,
                np.nan,
            )

        X = X[valid]
        y = y[valid]
        grp = grp[valid]

        valid2 = (
            np.isfinite(X2).all(axis=1)
            &
            np.isfinite(y2)
        )

        X2 = X2[valid2]
        y2 = y2[valid2]

        if np.unique(y).size < 2:

            return (
                np.nan,
                None,
                np.nan,
                np.nan,
                None,
                np.nan,
            )

        all_true = []
        all_pred = []

        for f in range(
            n_splits_eff
        ):

            test_mask = (
                grp == f
            )

            train_mask = (
                ~test_mask
            )

            if (
                not np.any(test_mask)
                or
                not np.any(train_mask)
            ):
                continue

            Xtr = X[train_mask]
            ytr = y[train_mask]

            Xte = X[test_mask]
            yte = y[test_mask]

            if np.unique(ytr).size < 2:
                continue

            model = LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                random_state=0
            )

            model.fit(
                Xtr,
                ytr
            )

            yhat = model.predict(
                Xte
            )

            all_true.extend(
                yte.tolist()
            )

            all_pred.extend(
                yhat.tolist()
            )

        if len(all_true) == 0:

            return (
                np.nan,
                None,
                np.nan,
                np.nan,
                None,
                np.nan,
            )

        all_true = np.asarray(
            all_true
        )

        all_pred = np.asarray(
            all_pred
        )

        acc = accuracy_score(
            all_true,
            all_pred
        )

        cm = confusion_matrix(
            all_true,
            all_pred,
            labels=[0, 1, 2]
        )

        y_true_bin = (
            all_true >= 1
        ).astype(int)

        y_pred_bin = (
            all_pred >= 1
        ).astype(int)

        bin_acc = accuracy_score(
            y_true_bin,
            y_pred_bin
        )

        if X2.shape[0] == 0:

            return (
                acc,
                cm,
                bin_acc,
                np.nan,
                None,
                np.nan,
            )

        model = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=0
        )

        model.fit(
            X,
            y
        )

        yhat2 = model.predict(
            X2
        )

        oob_acc = accuracy_score(
            y2,
            yhat2
        )

        cm2 = confusion_matrix(
            y2,
            yhat2,
            labels=[0, 1, 2]
        )

        y_true_bin2 = (
            y2 >= 1
        ).astype(int)

        y_pred_bin2 = (
            yhat2 >= 1
        ).astype(int)

        oob_bin_acc = accuracy_score(
            y_true_bin2,
            y_pred_bin2
        )

        return (
            acc,
            cm,
            bin_acc,
            oob_acc,
            cm2,
            oob_bin_acc,
        )


    def grid_search_dedup(
        dedup_train,
        dedup_test,
        ratio_grid,
        dq_grid
    ):

        rows = []

        for r in ratio_grid:

            for dq in dq_grid:

                df_f = apply_filters_for_search(
                    dedup_train,
                    r,
                    dq
                )

                df_f2 = apply_filters_for_search(
                    dedup_test,
                    r,
                    dq
                )

                (
                    acc,
                    cm,
                    bin_acc,
                    oob_acc,
                    cm2,
                    oob_bin_acc,
                ) = cv_multinomial_metrics(
                    df_f,
                    df_f2
                )

                rows.append({
                    "ratio": r,
                    "dq": int(dq),
                    "accuracy": acc,
                    "bin_acc": bin_acc,
                    "oob_acc": oob_acc,
                    "oob_bin_acc": oob_bin_acc,
                    "n_rows": int(
                        len(df_f)
                    ),
                    "n_rows_test": int(
                        len(df_f2)
                    ),
                    "conf_mat": cm,
                    "conf_mat2": cm2,
                })

        res = (
            pd
            .DataFrame(rows)
            .dropna(
                subset=[
                    "accuracy"
                ]
            )
            .reset_index(
                drop=True
            )
        )

        return res


    def choose_best(
        res,
        tol=0.002
    ):

        if res.empty:

            return (
                None,
                pd.DataFrame(
                    columns=res.columns
                )
            )

        best = (
            res["accuracy"].max()
        )

        near = (
            res[
                res["accuracy"]
                >= best - tol
            ]
            .copy()
        )

        near = (
            near
            .sort_values(
                [
                    "n_rows",
                    "accuracy",
                    "ratio",
                    "dq",
                ],
                ascending=[
                    False,
                    False,
                    True,
                    True,
                ]
            )
            .reset_index(
                drop=True
            )
        )

        return (
            near.iloc[0].to_dict(),
            near
        )


    # -----------------------------------------------------------------
    # OVERALL NON-BOOTSTRAP ICESAT + TOPOGRAPHY COUNTS
    # -----------------------------------------------------------------
    if USE_TOPOGRAPHY:

        print("\n" + "=" * 80)

        print(
            "OVERALL NON-BOOTSTRAP ICESAT-2 + TOPOGRAPHY FILTER COUNTS"
        )

        print("=" * 80)

        overall_res = grid_search_dedup(
            df_grouped,
            df_grouped,
            RATIO_GRID,
            DQ_GRID
        )

        overall_chosen, overall_near = choose_best(
            overall_res,
            tol=TOL_NEAR
        )

        if overall_chosen is None:

            print(
                "No valid full-data ICESat-2 filter found."
            )

        else:

            overall_ratio = float(
                overall_chosen[
                    "ratio"
                ]
            )

            overall_dq = int(
                overall_chosen[
                    "dq"
                ]
            )

            print(
                f"Overall selected ICESat-2 filter: "
                f"Eg_strong/Eg_weak >= "
                f"{overall_ratio:.2f}, "
                f"data_quantity >= "
                f"{overall_dq}"
            )

            print(
                f"Full-data CV acc="
                f"{overall_chosen['accuracy']:.4f}, "
                f"CV bin acc="
                f"{overall_chosen['bin_acc']:.4f}"
            )

            overall_filtered = (
                apply_filters_for_boot(
                    df_grouped,
                    overall_ratio,
                    overall_dq
                )
            )

            overall_aspect_variability = (
                pd
                .to_numeric(
                    overall_filtered[
                        ASPECT_VARIABILITY_COL
                    ],
                    errors="coerce"
                )
                .to_numpy()
            )

            print(
                f"Total grouped cells: "
                f"{len(df_grouped)}"
            )

            print(
                f"Cells passing overall "
                f"ICESat-2 filters: "
                f"{len(overall_filtered)}"
            )


            # ---------------------------------------------------------
            # LOW-VARIABILITY NORTH/SOUTH COUNTS
            # ---------------------------------------------------------
            if USE_ASPECT_SPLIT:

                overall_aspect_groups = (
                    overall_filtered[
                        ASPECT_GROUP_COL
                    ]
                    .astype("string")
                    .fillna(
                        "unclassified"
                    )
                    .to_numpy()
                )

                overall_aspect_pass = (
                    np.isfinite(
                        overall_aspect_variability
                    )
                    &
                    (
                        overall_aspect_variability
                        <= ASPECT_VARIABILITY_THRESHOLD
                    )
                )

                overall_aspect_pass_classified = (
                    overall_aspect_pass
                    &
                    np.isin(
                        overall_aspect_groups,
                        ASPECT_GROUPS
                    )
                )

                print(
                    f"\nCells passing ICESat-2 filters + "
                    f"{ASPECT_VARIABILITY_COL} <= "
                    f"{ASPECT_VARIABILITY_THRESHOLD}: "
                    f"{int(np.count_nonzero(overall_aspect_pass))}"
                )

                print(
                    "Cells passing ICESat-2 filters + "
                    "low aspect variability filter + "
                    "valid north/south group: "
                    f"{int(np.count_nonzero(overall_aspect_pass_classified))}"
                )

                for aspect_group in ASPECT_GROUPS:

                    n_group = int(
                        np.count_nonzero(
                            overall_aspect_pass_classified
                            &
                            (
                                overall_aspect_groups
                                ==
                                aspect_group
                            )
                        )
                    )

                    print(
                        f"  {aspect_group}: "
                        f"{n_group}"
                    )

                print(
                    "\nBy observed JointSnowBinary class "
                    "after ICESat-2 + low-variability aspect filters:"
                )

                tmp_counts_df = (
                    overall_filtered
                    .loc[
                        overall_aspect_pass_classified
                    ]
                    .copy()
                )

                if len(tmp_counts_df) == 0:

                    print(
                        "  No cells passed both filters."
                    )

                else:

                    print(
                        tmp_counts_df[
                            "JointSnowBinary"
                        ]
                        .value_counts(
                            dropna=False
                        )
                        .sort_index()
                        .to_string()
                    )


            # ---------------------------------------------------------
            # HIGH-ASPECT-VARIABILITY COUNTS
            # ---------------------------------------------------------
            if USE_HIGH_ASPECT_VARIABILITY:

                overall_high_aspect_pass = (
                    np.isfinite(
                        overall_aspect_variability
                    )
                    &
                    (
                        overall_aspect_variability
                        >
                        HIGH_ASPECT_VARIABILITY_THRESHOLD
                    )
                )

                print(
                    f"\nCells passing ICESat-2 filters + "
                    f"{ASPECT_VARIABILITY_COL} > "
                    f"{HIGH_ASPECT_VARIABILITY_THRESHOLD}: "
                    f"{int(np.count_nonzero(overall_high_aspect_pass))}"
                )

                print(
                    "North/south aspect dominance is not considered "
                    "for this high-variability group."
                )

                print(
                    "\nBy observed JointSnowBinary class after "
                    "ICESat-2 + high aspect variability filter:"
                )

                tmp_high_counts_df = (
                    overall_filtered
                    .loc[
                        overall_high_aspect_pass
                    ]
                    .copy()
                )

                if len(tmp_high_counts_df) == 0:

                    print(
                        "  No cells passed the high-variability filter."
                    )

                else:

                    print(
                        tmp_high_counts_df[
                            "JointSnowBinary"
                        ]
                        .value_counts(
                            dropna=False
                        )
                        .sort_index()
                        .to_string()
                    )


    def mod2pi(x):

        return np.mod(
            x,
            2 * np.pi
        )


    def dccw(a, b):

        return mod2pi(
            b - a
        )


    def angular_sector_map(
        eg,
        ev,
        cx,
        cy,
        theta1,
        theta2,
        eps=1e-9
    ):

        eg = np.asarray(eg)
        ev = np.asarray(ev)

        theta = mod2pi(
            np.arctan2(
                ev - cy,
                eg - cx
            )
        )

        t1 = mod2pi(theta1)
        t2 = mod2pi(theta2)
        pi_m = mod2pi(np.pi)

        arc = dccw(
            t1,
            t2
        )

        arc = np.maximum(
            arc,
            eps
        )

        d1 = dccw(
            t1,
            theta
        )

        in_grad = (
            d1
            <= arc + eps
        )

        vals = np.empty_like(
            theta,
            dtype=float
        )

        vals[in_grad] = np.clip(
            d1[in_grad] / arc,
            0.0,
            1.0
        )

        d_from_t2 = dccw(
            t2,
            theta
        )

        d_t2_to_pi = dccw(
            t2,
            pi_m
        )

        in_high = (
            (~in_grad)
            &
            (
                d_from_t2
                <
                d_t2_to_pi - eps
            )
        )

        vals[in_high] = 1.0

        vals[
            ~(
                in_grad
                |
                in_high
            )
        ] = 0.0

        at_pi = np.isclose(
            mod2pi(theta),
            pi_m,
            atol=1e-12
        )

        vals[at_pi] = 0.0

        at_center = (
            np.isclose(
                eg,
                cx,
                atol=1e-12
            )
            &
            np.isclose(
                ev,
                cy,
                atol=1e-12
            )
        )

        vals[at_center] = 0.0

        return vals


    def tiny_arc_penalty(
        theta1,
        theta2,
        thresh=1e-3
    ):

        arc = dccw(
            mod2pi(theta1),
            mod2pi(theta2)
        )

        return (
            1e6
            *
            (
                thresh
                -
                arc
                +
                1e-9
            )
            if arc < thresh
            else 0.0
        )


    def weighted_rmse(
        y_true,
        y_pred,
        frac_weight=1.0,
        bin_weight=0.25
    ):

        y_true = np.asarray(
            y_true,
            dtype=float
        )

        y_pred = np.asarray(
            y_pred,
            dtype=float
        )

        w = np.where(
            (
                (y_true > 0)
                &
                (y_true < 1)
            ),
            frac_weight,
            bin_weight
        )

        return np.sqrt(
            np.sum(
                w
                *
                (
                    y_true
                    -
                    y_pred
                ) ** 2
            )
            /
            np.sum(w)
        )


    def fit_sector_model_with_group_binw(
        train_df
    ):

        data = train_df.dropna(
            subset=[
                EG_COL,
                EV_COL,
                Y_BIN_COL,
            ]
        ).copy()

        eg = data[
            EG_COL
        ].values

        ev = data[
            EV_COL
        ].values

        y = data[
            Y_BIN_COL
        ].astype(
            float
        ).values

        n_frac_total = int(
            (
                (y > 0)
                &
                (y < 1)
            ).sum()
        )

        n_bin_total = int(
            len(y)
            -
            n_frac_total
        )

        if BIN_W_PARAM == 0:

            BIN_W_GROUP = 1

        else:

            BIN_W_GROUP = (
                BIN_W_PARAM
                *
                (
                    n_frac_total
                    /
                    n_bin_total
                )
                if (
                    n_bin_total > 0
                    and
                    n_frac_total > 0
                )
                else 1.0
            )

        def init_params():

            return np.array(
                [
                    0.0,
                    1.8,
                    -np.pi / 4,
                    -np.pi / 8,
                ],
                dtype=float
            )

        bounds = [
            (-2, 0.0),
            (
                max(
                    1e-6,
                    0.0
                ),
                np.inf
            ),
            (
                -np.pi / 2,
                np.pi
            ),
            (
                -np.pi,
                0.0
            ),
        ]

        def objective(p):

            cx, cy, t1, t2 = p

            y_hat = angular_sector_map(
                eg,
                ev,
                cx,
                cy,
                t1,
                t2
            )

            return (
                weighted_rmse(
                    y,
                    y_hat,
                    frac_weight=FRAC_W,
                    bin_weight=BIN_W_GROUP
                )
                +
                tiny_arc_penalty(
                    t1,
                    t2
                )
            )

        res = minimize(
            objective,
            init_params(),
            method="L-BFGS-B",
            bounds=bounds
        )

        cx, cy, t1, t2 = res.x

        params = {
            "cx": cx,
            "cy": cy,
            "theta1": t1,
            "theta2": t2,
            "BIN_W_GROUP": BIN_W_GROUP,
        }

        return params


    def predict_sector(
        df,
        params
    ):

        eg = df[
            EG_COL
        ].values

        ev = df[
            EV_COL
        ].values

        p = angular_sector_map(
            eg,
            ev,
            params["cx"],
            params["cy"],
            params["theta1"],
            params["theta2"]
        )

        return np.clip(
            p,
            0.0,
            1.0
        )


    def empty_metrics():

        return dict(
            overall_rmse=np.nan,
            overall_bias=np.nan,
            overall_frac_rmse=np.nan,
            overall_frac_bias=np.nan,
            overall_none_rmse=np.nan,
            overall_none_bias=np.nan,
            overall_full_rmse=np.nan,
            overall_full_bias=np.nan,
        )


    def compute_metrics(
        y_true,
        y_pred
    ):

        y_true = np.asarray(
            y_true,
            dtype=float
        )

        y_pred = np.asarray(
            y_pred,
            dtype=float
        )

        if y_true.size == 0:
            return empty_metrics()

        overall_rmse = float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred
                )
            )
        )

        overall_bias = float(
            np.mean(
                y_pred - y_true
            )
        )

        frac_mask = (
            (y_true > 0)
            &
            (y_true < 1)
        )

        frac_rmse = (
            float(
                np.sqrt(
                    mean_squared_error(
                        y_true[frac_mask],
                        y_pred[frac_mask]
                    )
                )
            )
            if np.any(frac_mask)
            else np.nan
        )

        frac_bias = (
            float(
                np.mean(
                    y_pred[frac_mask]
                    -
                    y_true[frac_mask]
                )
            )
            if np.any(frac_mask)
            else np.nan
        )

        none_mask = (
            y_true == 0
        )

        none_rmse = (
            float(
                np.sqrt(
                    mean_squared_error(
                        y_true[none_mask],
                        y_pred[none_mask]
                    )
                )
            )
            if np.any(none_mask)
            else np.nan
        )

        none_bias = (
            float(
                np.mean(
                    y_pred[none_mask]
                    -
                    y_true[none_mask]
                )
            )
            if np.any(none_mask)
            else np.nan
        )

        full_mask = (
            y_true == 1
        )

        full_rmse = (
            float(
                np.sqrt(
                    mean_squared_error(
                        y_true[full_mask],
                        y_pred[full_mask]
                    )
                )
            )
            if np.any(full_mask)
            else np.nan
        )

        full_bias = (
            float(
                np.mean(
                    y_pred[full_mask]
                    -
                    y_true[full_mask]
                )
            )
            if np.any(full_mask)
            else np.nan
        )

        return dict(
            overall_rmse=overall_rmse,
            overall_bias=overall_bias,
            overall_frac_rmse=frac_rmse,
            overall_frac_bias=frac_bias,
            overall_none_rmse=none_rmse,
            overall_none_bias=none_bias,
            overall_full_rmse=full_rmse,
            overall_full_bias=full_bias,
        )


    def safe_metric_text(
        m,
        key
    ):

        val = m[key]

        return (
            f"{val:.4f}"
            if np.isfinite(val)
            else "nan"
        )


    # ------------------------------ bootstrap loop ------------------------------
    all_cameras = sorted(
        df_grouped[
            "camera"
        ].unique()
    )

    assert len(all_cameras) == num_cameras, (
        f"Expected {num_cameras} unique cameras, "
        f"found {len(all_cameras)}."
    )

    phase2_rows = []
    phase2_aspect_rows = []
    phase2_high_aspect_rows = []

    all_oob_y_true = []
    all_oob_y_pred = []
    all_oob_cams = []

    all_oob_aspect_groups = []
    all_oob_aspect_variability = []

    cumulative_oob_conf_mat = np.zeros(
        (3, 3),
        dtype=int
    )

    sample_oob_df = None

    test_counts = []
    test_counts_0 = []
    test_counts_1 = []
    test_counts_p = []

    unique_oob_cells = {}


    for b in range(N_BOOT):

        start = time.time()

        sampled_cams = RNG.choice(
            all_cameras,
            size=len(all_cameras),
            replace=True
        )

        sampled_unique = sorted(
            set(sampled_cams)
        )

        oob_cams = sorted(
            set(all_cameras)
            -
            set(sampled_unique)
        )

        boot_concat = pd.concat(
            [
                df_grouped[
                    df_grouped["camera"]
                    ==
                    cam
                ]
                for cam in sampled_cams
            ],
            ignore_index=True
        )

        dedup_train = (
            boot_concat.copy()
        )

        dedup_test = (
            df_grouped[
                df_grouped[
                    "camera"
                ].isin(
                    oob_cams
                )
            ]
            .copy()
        )

        res = grid_search_dedup(
            dedup_train,
            dedup_test,
            RATIO_GRID,
            DQ_GRID
        )

        chosen, near = choose_best(
            res,
            tol=TOL_NEAR
        )

        print(
            f"\n=== Bootstrap "
            f"{b + 1}/{N_BOOT} ==="
        )

        if chosen is None:

            print(
                "No valid filter produced a trainable "
                "dataset for search."
            )

            continue

        print(
            f"\nChosen filter -> "
            f"Eg_strong/Eg_weak >= "
            f"{chosen['ratio']:.2f}, "
            f"data_quantity >= "
            f"{int(chosen['dq'])} "
            f"| CV acc="
            f"{chosen['accuracy']:.4f}, "
            f"CV bin acc="
            f"{chosen['bin_acc']:.4f}, "
            f"n_rows(dedup)="
            f"{int(chosen['n_rows'])} "
            f"| OOB acc="
            f"{chosen['oob_acc']:.4f}, "
            f"OOB bin acc="
            f"{chosen['oob_bin_acc']:.4f}, "
            f"n_rows_test(dedup)="
            f"{int(chosen['n_rows_test'])} |"
        )

        if isinstance(
            chosen.get(
                "conf_mat2",
                None
            ),
            np.ndarray
        ):

            cumulative_oob_conf_mat += (
                chosen[
                    "conf_mat2"
                ]
                .astype(int)
            )

        boot_train = apply_filters_for_boot(
            boot_concat,
            chosen["ratio"],
            int(
                chosen["dq"]
            )
        )

        if len(boot_train) == 0:

            print(
                "Bootstrapped training set empty after "
                "filters; skipping."
            )

            continue

        params = (
            fit_sector_model_with_group_binw(
                boot_train
            )
        )

        oob_df = (
            df_grouped[
                df_grouped[
                    "camera"
                ].isin(
                    oob_cams
                )
            ]
            .copy()
        )

        oob_df = apply_filters_for_boot(
            oob_df,
            chosen["ratio"],
            int(
                chosen["dq"]
            )
        )

        if not oob_df.empty:

            vals = (
                oob_df[
                    Y_BIN_COL
                ]
                .values
            )

            ids = (
                oob_df[
                    "cell_id"
                ]
                .values
            )

            cls = np.where(
                vals == 0,
                0,
                np.where(
                    vals == 1,
                    1,
                    2
                )
            )

            for cid, c in zip(
                ids,
                cls
            ):

                if cid not in unique_oob_cells:

                    unique_oob_cells[
                        cid
                    ] = int(c)

        test_counts.append(
            len(oob_df)
        )

        test_counts_0.append(
            len(
                oob_df[
                    oob_df[
                        Y_BIN_COL
                    ] == 0
                ]
            )
        )

        test_counts_1.append(
            len(
                oob_df[
                    oob_df[
                        Y_BIN_COL
                    ] == 1
                ]
            )
        )

        test_counts_p.append(
            len(
                oob_df[
                    (
                        oob_df[
                            Y_BIN_COL
                        ] > 0
                    )
                    &
                    (
                        oob_df[
                            Y_BIN_COL
                        ] < 1
                    )
                ]
            )
        )

        if len(oob_df) > 0:

            y_pred = predict_sector(
                oob_df,
                params
            )

            y_true = (
                oob_df[
                    Y_BIN_COL
                ]
                .astype(float)
                .values
            )

            cams = (
                oob_df[
                    "camera"
                ]
                .values
            )

            # ---------------------------------------------------------
            # TOPOGRAPHY VARIABLES
            # ---------------------------------------------------------
            if USE_TOPOGRAPHY:

                aspect_variability = (
                    pd
                    .to_numeric(
                        oob_df[
                            ASPECT_VARIABILITY_COL
                        ],
                        errors="coerce"
                    )
                    .to_numpy()
                )

            else:

                aspect_variability = None


            # ---------------------------------------------------------
            # LOW-VARIABILITY NORTH/SOUTH ELIGIBILITY
            # ---------------------------------------------------------
            if USE_ASPECT_SPLIT:

                aspect_groups = (
                    oob_df[
                        ASPECT_GROUP_COL
                    ]
                    .astype("string")
                    .fillna(
                        "unclassified"
                    )
                    .to_numpy()
                )

                aspect_eligible = (
                    np.isfinite(
                        aspect_variability
                    )
                    &
                    (
                        aspect_variability
                        <=
                        ASPECT_VARIABILITY_THRESHOLD
                    )
                    &
                    np.isin(
                        aspect_groups,
                        ASPECT_GROUPS
                    )
                )

            else:

                aspect_groups = None
                aspect_eligible = None


            # ---------------------------------------------------------
            # HIGH-ASPECT-VARIABILITY ELIGIBILITY
            # No north/south aspect condition is used
            # ---------------------------------------------------------
            if USE_HIGH_ASPECT_VARIABILITY:

                high_aspect_eligible = (
                    np.isfinite(
                        aspect_variability
                    )
                    &
                    (
                        aspect_variability
                        >
                        HIGH_ASPECT_VARIABILITY_THRESHOLD
                    )
                )

            else:

                high_aspect_eligible = None


            all_oob_y_true.append(
                y_true.copy()
            )

            all_oob_y_pred.append(
                y_pred.copy()
            )

            all_oob_cams.append(
                cams.copy()
            )

            if USE_ASPECT_SPLIT:

                all_oob_aspect_groups.append(
                    aspect_groups.copy()
                )

            if USE_TOPOGRAPHY:

                all_oob_aspect_variability.append(
                    aspect_variability.copy()
                )

            if sample_oob_df is None:

                sample_oob_df = (
                    oob_df.copy()
                )

                sample_params = params

                sample_y_true = (
                    y_true.copy()
                )

                sample_y_pred = (
                    y_pred.copy()
                )

            m = compute_metrics(
                y_true,
                y_pred
            )

            print(
                f"OOB cameras: "
                f"{oob_cams if oob_cams else 'none (all cameras sampled)'}"
            )

            print(
                f"OOB n={len(oob_df)} | "
                f"RMSE="
                f"{safe_metric_text(m, 'overall_rmse')} | "
                f"Bias="
                f"{safe_metric_text(m, 'overall_bias')} | "
                f"FracRMSE="
                f"{safe_metric_text(m, 'overall_frac_rmse')} | "
                f"FracBias="
                f"{safe_metric_text(m, 'overall_frac_bias')} | "
                f"NoneBias="
                f"{safe_metric_text(m, 'overall_none_bias')} | "
                f"FullBias="
                f"{safe_metric_text(m, 'overall_full_bias')}"
            )


            # ---------------------------------------------------------
            # LOW-VARIABILITY NORTH/SOUTH METRICS
            # ---------------------------------------------------------
            if USE_ASPECT_SPLIT:

                n_eligible = int(
                    np.count_nonzero(
                        aspect_eligible
                    )
                )

                print(
                    f"Aspect split eligible OOB rows "
                    f"({ASPECT_VARIABILITY_COL} <= "
                    f"{ASPECT_VARIABILITY_THRESHOLD}): "
                    f"{n_eligible} / "
                    f"{len(oob_df)}"
                )

                for aspect_group in ASPECT_GROUPS:

                    group_mask = (
                        aspect_eligible
                        &
                        (
                            aspect_groups
                            ==
                            aspect_group
                        )
                    )

                    n_group = int(
                        np.count_nonzero(
                            group_mask
                        )
                    )

                    if n_group > 0:

                        m_aspect = compute_metrics(
                            y_true[group_mask],
                            y_pred[group_mask]
                        )

                    else:

                        m_aspect = empty_metrics()

                    phase2_aspect_rows.append({
                        "bootstrap": b + 1,
                        "aspect_group": aspect_group,
                        "aspect_variability_threshold":
                            ASPECT_VARIABILITY_THRESHOLD,
                        "n_rows_oob": n_group,
                        "oob_rmse":
                            m_aspect["overall_rmse"],
                        "oob_bias":
                            m_aspect["overall_bias"],
                        "oob_frac_rmse":
                            m_aspect["overall_frac_rmse"],
                        "oob_frac_bias":
                            m_aspect["overall_frac_bias"],
                        "oob_none_rmse":
                            m_aspect["overall_none_rmse"],
                        "oob_none_bias":
                            m_aspect["overall_none_bias"],
                        "oob_full_rmse":
                            m_aspect["overall_full_rmse"],
                        "oob_full_bias":
                            m_aspect["overall_full_bias"],
                    })

                    print(
                        f"{aspect_group} OOB "
                        f"n={n_group} | "
                        f"RMSE="
                        f"{safe_metric_text(m_aspect, 'overall_rmse')} | "
                        f"Bias="
                        f"{safe_metric_text(m_aspect, 'overall_bias')} | "
                        f"FracRMSE="
                        f"{safe_metric_text(m_aspect, 'overall_frac_rmse')} | "
                        f"FracBias="
                        f"{safe_metric_text(m_aspect, 'overall_frac_bias')} | "
                        f"NoneBias="
                        f"{safe_metric_text(m_aspect, 'overall_none_bias')} | "
                        f"FullBias="
                        f"{safe_metric_text(m_aspect, 'overall_full_bias')}"
                    )


            # ---------------------------------------------------------
            # HIGH-ASPECT-VARIABILITY METRICS
            # ---------------------------------------------------------
            if USE_HIGH_ASPECT_VARIABILITY:

                n_high = int(
                    np.count_nonzero(
                        high_aspect_eligible
                    )
                )

                print(
                    f"High aspect variability eligible OOB rows "
                    f"({ASPECT_VARIABILITY_COL} > "
                    f"{HIGH_ASPECT_VARIABILITY_THRESHOLD}): "
                    f"{n_high} / "
                    f"{len(oob_df)}"
                )

                if n_high > 0:

                    m_high = compute_metrics(
                        y_true[
                            high_aspect_eligible
                        ],
                        y_pred[
                            high_aspect_eligible
                        ]
                    )

                else:

                    m_high = empty_metrics()

                phase2_high_aspect_rows.append({
                    "bootstrap": b + 1,
                    "aspect_group":
                        HIGH_ASPECT_GROUP,
                    "aspect_variability_threshold":
                        HIGH_ASPECT_VARIABILITY_THRESHOLD,
                    "n_rows_oob":
                        n_high,
                    "oob_rmse":
                        m_high["overall_rmse"],
                    "oob_bias":
                        m_high["overall_bias"],
                    "oob_frac_rmse":
                        m_high["overall_frac_rmse"],
                    "oob_frac_bias":
                        m_high["overall_frac_bias"],
                    "oob_none_rmse":
                        m_high["overall_none_rmse"],
                    "oob_none_bias":
                        m_high["overall_none_bias"],
                    "oob_full_rmse":
                        m_high["overall_full_rmse"],
                    "oob_full_bias":
                        m_high["overall_full_bias"],
                })

                print(
                    f"{HIGH_ASPECT_GROUP} OOB "
                    f"n={n_high} | "
                    f"RMSE="
                    f"{safe_metric_text(m_high, 'overall_rmse')} | "
                    f"Bias="
                    f"{safe_metric_text(m_high, 'overall_bias')} | "
                    f"FracRMSE="
                    f"{safe_metric_text(m_high, 'overall_frac_rmse')} | "
                    f"FracBias="
                    f"{safe_metric_text(m_high, 'overall_frac_bias')} | "
                    f"NoneBias="
                    f"{safe_metric_text(m_high, 'overall_none_bias')} | "
                    f"FullBias="
                    f"{safe_metric_text(m_high, 'overall_full_bias')}"
                )


            phase2_rows.append({
                "bootstrap": b + 1,
                "ratio":
                    float(chosen["ratio"]),
                "dq":
                    int(chosen["dq"]),
                "n_rows_search":
                    int(chosen["n_rows"]),
                "n_rows_boot_train":
                    int(len(boot_train)),
                "n_rows_oob":
                    int(len(oob_df)),
                "oob_rmse":
                    m["overall_rmse"],
                "oob_bias":
                    m["overall_bias"],
                "oob_frac_rmse":
                    m["overall_frac_rmse"],
                "oob_frac_bias":
                    m["overall_frac_bias"],
                "oob_none_rmse":
                    m["overall_none_rmse"],
                "oob_none_bias":
                    m["overall_none_bias"],
                "oob_full_rmse":
                    m["overall_full_rmse"],
                "oob_full_bias":
                    m["overall_full_bias"],
                "n_oob_cameras":
                    len(oob_cams),
                "bin_w_group":
                    params.get(
                        "BIN_W_GROUP",
                        np.nan
                    ),
                "cv_acc":
                    float(
                        chosen["accuracy"]
                    ),
                "cv_bin_acc":
                    float(
                        chosen["bin_acc"]
                    ),
                "oob_acc":
                    float(
                        chosen["oob_acc"]
                    ),
                "oob_bin_acc":
                    float(
                        chosen["oob_bin_acc"]
                    ),
            })

        end = time.time()

        print(
            f"{round(end - start, 2)}s"
        )


    # ------------------------------ summary ------------------------------
    phase2_df = pd.DataFrame(
        phase2_rows
    )

    phase2_aspect_df = pd.DataFrame(
        phase2_aspect_rows
    )

    phase2_high_aspect_df = pd.DataFrame(
        phase2_high_aspect_rows
    )


    # -----------------------------------------------------------------
    # Per-site / per-camera OOB metrics across all bootstraps
    # -----------------------------------------------------------------
    y_true_all = (
        np.concatenate(
            all_oob_y_true
        )
        if len(all_oob_y_true)
        else np.array([])
    )

    y_pred_all = (
        np.concatenate(
            all_oob_y_pred
        )
        if len(all_oob_y_pred)
        else np.array([])
    )

    cam_all = (
        np.concatenate(
            all_oob_cams
        )
        if len(all_oob_cams)
        else np.array([])
    )

    if (
        y_true_all.size
        and
        y_pred_all.size
        and
        cam_all.size
    ):

        metrics_order = [
            "RMSE",
            "Bias",
            "Fractional RMSE",
            "Fractional Bias",
            "0%SC Error",
            "100%SC Error",
        ]

        key_order = [
            "overall_rmse",
            "overall_bias",
            "overall_frac_rmse",
            "overall_frac_bias",
            "overall_none_bias",
            "overall_full_bias",
        ]

        per_site_records = []

        for cam in sorted(
            np.unique(
                cam_all
            )
        ):

            mask = (
                cam_all == cam
            )

            if not np.any(mask):
                continue

            m = compute_metrics(
                y_true_all[mask],
                y_pred_all[mask]
            )

            rec = {
                "camera": cam,
                "n_oob_predictions":
                    int(
                        np.count_nonzero(
                            mask
                        )
                    ),
                "n_oob_0":
                    int(
                        np.count_nonzero(
                            y_true_all[mask]
                            == 0
                        )
                    ),
                "n_oob_1":
                    int(
                        np.count_nonzero(
                            y_true_all[mask]
                            == 1
                        )
                    ),
                "n_oob_partial":
                    int(
                        np.count_nonzero(
                            (
                                y_true_all[mask]
                                > 0
                            )
                            &
                            (
                                y_true_all[mask]
                                < 1
                            )
                        )
                    ),
            }

            for label, key in zip(
                metrics_order,
                key_order
            ):

                val = m.get(
                    key,
                    np.nan
                )

                rec[label] = (
                    val * 100.0
                    if np.isfinite(val)
                    else np.nan
                )

            per_site_records.append(
                rec
            )

        per_site_df = (
            pd.DataFrame.from_records(
                per_site_records
            )
        )

        print(
            "\nPer-site OOB metrics across all bootstraps, "
            "values in %:"
        )

        print(
            per_site_df.to_string(
                index=False
            )
        )

    else:

        print(
            "\nNo OOB predictions available "
            "for per-site metrics."
        )


    print(
        "\n===================="
    )

    print(
        "\nFilter choice frequency across bootstraps:"
    )

    freq = (
        phase2_df
        .dropna(
            subset=[
                "ratio",
                "dq",
            ]
        )
        .value_counts(
            subset=[
                "ratio",
                "dq",
            ]
        )
        .reset_index(
            name="count"
        )
        .sort_values(
            "count",
            ascending=False
        )
    )

    print(
        freq.to_string(
            index=False
        )
    )


    print(
        "\nOOB metrics mean +/- std across bootstraps, "
        "ignoring NaNs:"
    )


    def mean_std(series):

        return (
            f"{np.nanmean(series):.4f} "
            f"+/- "
            f"{np.nanstd(series):.4f}"
        )


    print(
        "RMSE:      ",
        mean_std(
            phase2_df[
                "oob_rmse"
            ]
        )
    )

    print(
        "Bias:      ",
        mean_std(
            phase2_df[
                "oob_bias"
            ]
        )
    )

    print(
        "Frac RMSE: ",
        mean_std(
            phase2_df[
                "oob_frac_rmse"
            ]
        )
    )

    print(
        "Frac Bias: ",
        mean_std(
            phase2_df[
                "oob_frac_bias"
            ]
        )
    )

    print(
        "0%SC Bias: ",
        mean_std(
            phase2_df[
                "oob_none_bias"
            ]
        )
    )

    print(
        "100%SC Bias: ",
        mean_std(
            phase2_df[
                "oob_full_bias"
            ]
        )
    )


    # -----------------------------------------------------------------
    # LOW-VARIABILITY NORTH/SOUTH SUMMARY
    # -----------------------------------------------------------------
    if (
        USE_ASPECT_SPLIT
        and
        not phase2_aspect_df.empty
    ):

        print(
            "\nOOB metrics by aspect group, "
            "mean +/- std across bootstraps "
            f"using rows where "
            f"{ASPECT_VARIABILITY_COL} <= "
            f"{ASPECT_VARIABILITY_THRESHOLD}:"
        )

        for aspect_group in ASPECT_GROUPS:

            sub = phase2_aspect_df[
                phase2_aspect_df[
                    "aspect_group"
                ]
                ==
                aspect_group
            ]

            if sub.empty:

                print(
                    f"\n{aspect_group}: "
                    f"no rows"
                )

                continue

            print(
                f"\n{aspect_group}:"
            )

            print(
                "Rows:        ",
                int(
                    np.nansum(
                        sub[
                            "n_rows_oob"
                        ]
                    )
                )
            )

            print(
                "RMSE:        ",
                mean_std(
                    sub[
                        "oob_rmse"
                    ]
                )
            )

            print(
                "Bias:        ",
                mean_std(
                    sub[
                        "oob_bias"
                    ]
                )
            )

            print(
                "Frac RMSE:   ",
                mean_std(
                    sub[
                        "oob_frac_rmse"
                    ]
                )
            )

            print(
                "Frac Bias:   ",
                mean_std(
                    sub[
                        "oob_frac_bias"
                    ]
                )
            )

            print(
                "0%SC Bias:   ",
                mean_std(
                    sub[
                        "oob_none_bias"
                    ]
                )
            )

            print(
                "100%SC Bias: ",
                mean_std(
                    sub[
                        "oob_full_bias"
                    ]
                )
            )


    # -----------------------------------------------------------------
    # HIGH-ASPECT-VARIABILITY SUMMARY
    # -----------------------------------------------------------------
    if (
        USE_HIGH_ASPECT_VARIABILITY
        and
        not phase2_high_aspect_df.empty
    ):

        print(
            "\nOOB metrics for high aspect variability, "
            "mean +/- std across bootstraps "
            f"using rows where "
            f"{ASPECT_VARIABILITY_COL} > "
            f"{HIGH_ASPECT_VARIABILITY_THRESHOLD}."
        )

        print(
            "North/south aspect dominance is not considered."
        )

        sub = phase2_high_aspect_df.copy()

        print(
            f"\n{HIGH_ASPECT_GROUP}:"
        )

        print(
            "Rows:        ",
            int(
                np.nansum(
                    sub[
                        "n_rows_oob"
                    ]
                )
            )
        )

        print(
            "RMSE:        ",
            mean_std(
                sub[
                    "oob_rmse"
                ]
            )
        )

        print(
            "Bias:        ",
            mean_std(
                sub[
                    "oob_bias"
                ]
            )
        )

        print(
            "Frac RMSE:   ",
            mean_std(
                sub[
                    "oob_frac_rmse"
                ]
            )
        )

        print(
            "Frac Bias:   ",
            mean_std(
                sub[
                    "oob_frac_bias"
                ]
            )
        )

        print(
            "0%SC Bias:   ",
            mean_std(
                sub[
                    "oob_none_bias"
                ]
            )
        )

        print(
            "100%SC Bias: ",
            mean_std(
                sub[
                    "oob_full_bias"
                ]
            )
        )


    print(
        "\nCV metrics mean +/- std across "
        "chosen filters per bootstrap:"
    )

    print(
        "Multiclass accuracy: ",
        mean_std(
            phase2_df[
                "oob_acc"
            ]
        )
    )

    print(
        "Binary accuracy 0 vs {1,2}: ",
        mean_std(
            phase2_df[
                "oob_bin_acc"
            ]
        )
    )

    print(
        f"Total Cells: "
        f"{np.sum(test_counts)}"
    )

    print(
        f"Total Non-Snow Cells: "
        f"{np.sum(test_counts_0)}"
    )

    print(
        f"Total Snow Cells: "
        f"{np.sum(test_counts_1)}"
    )

    print(
        f"Total Partial Snow Cells: "
        f"{np.sum(test_counts_p)}"
    )


finally:

    try:

        sys.stdout = _orig_stdout
        sys.stderr = _orig_stderr

    except Exception:

        pass

    try:

        log_fh.write(
            f"\nRun finished: "
            f"{datetime.now().isoformat(timespec='seconds')}\n"
        )

        log_fh.flush()
        log_fh.close()

    except Exception:

        pass