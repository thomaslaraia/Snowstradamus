from scripts.parallel_blocks import *

import os
import re
import numpy as np
import pandas as pd
import rioxarray


# ------------------------------------------------------------------
# WORLDCOVER SETTINGS
# ------------------------------------------------------------------
WC_FOLDER = "../scratch/data/WC/"
WC_FOREST_VALUE = 10


def find_worldcover_file(camera_name, wc_folder=WC_FOLDER):
    """
    Find the WorldCover raster for a camera.

    Expected filename format:
        ../scratch/data/WC/{camera_name}.tif
    """
    filepath = os.path.join(wc_folder, f"{camera_name}.tif")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Could not find WorldCover file for {camera_name}: {filepath}"
        )

    return filepath


def plot_parallel(atl03s, coefs, colors, title_date, X, Y, xx, yy,
                  beam=None, canopy_frac=None, terrain_frac=None,
                  file_index=None, three=None, data_quality=0):
    """
    Plotting function of pvpg_parallel. Shows a regression line for each available
    groundtrack in a bigger plot, as well as groundtrack visualisations in smaller plots.
    """

    title_color = ['black', 'red']

    beam_names = [f"Beam {i}" for i in range(1, 7)]

    fig = plt.figure(figsize=(10, 12))
    if three is None:
        ax1 = fig.add_subplot(331)
        ax2 = fig.add_subplot(332)
        ax3 = fig.add_subplot(334)
        ax4 = fig.add_subplot(335)
        ax5 = fig.add_subplot(337)
        ax6 = fig.add_subplot(338)
        ax7 = fig.add_subplot(133)
    else:
        ax1 = fig.add_subplot(321)
        ax2 = fig.add_subplot(322)
        ax3 = fig.add_subplot(323)
        ax4 = fig.add_subplot(324)
        ax5 = fig.add_subplot(325)
        ax6 = fig.add_subplot(326)

    axes = [ax1, ax2, ax3, ax4, ax5, ax6]

    if file_index is not None:
        fig.suptitle(
            title_date + ' - N = ' + str(file_index),
            fontsize=16,
            color=title_color[data_quality]
        )
    else:
        fig.suptitle(title_date, fontsize=16, color=title_color[data_quality])

    for i, c, atl03 in zip(np.arange(len(colors)), colors, atl03s):

        if (canopy_frac is not None) & (terrain_frac is not None):
            axes[c].set_title(
                f"{beam_names[c]} - TF = {round(terrain_frac[c], 2)}, "
                f"CF = {round(canopy_frac[c], 2)}"
            )
            plot(atl03, axes[c])

        elif canopy_frac is not None:
            axes[c].set_title(f"{beam_names[c]} - CF = {round(canopy_frac[c], 2)}")
            plot(atl03, axes[c])

        elif terrain_frac is not None:
            axes[c].set_title(f"{beam_names[c]} - TF = {round(terrain_frac[c], 2)}")
            plot(atl03, axes[c])

        else:
            axes[c].set_title(f"{beam_names[c]}")
            plot(atl03, axes[c])

        if three is None:

            if beam is not None:
                if c + 1 in beam:
                    ax7.scatter(X[i], Y[i], s=5, color=cmap3(2 * c + 1), marker='o')
                    ax7.scatter(xx[c], yy[c], s=5, color=cmap3(2 * c), marker='o')
                    ax7.plot(
                        np.array([0, 12]),
                        model([coefs[0], coefs[1 + i]], np.array([0, 12])),
                        label=f"Beam {int(c + 1)}",
                        color=cmap3(2 * c),
                        linestyle='--',
                        zorder=3
                    )
            else:
                ax7.scatter(X[i], Y[i], s=5, color=cmap3(2 * c + 1), marker='o')
                ax7.scatter(xx[c], yy[c], s=5, color=cmap3(2 * c), marker='o')
                ax7.plot(
                    np.array([0, 12]),
                    model([coefs[0], coefs[1 + i]], np.array([0, 12])),
                    label=f"Beam {int(c + 1)}",
                    color=cmap3(2 * c),
                    linestyle='--',
                    zorder=3
                )

    if three is None:
        ax7.annotate(
            r'$\rho_v/\rho_g \approx {:.2f}$'.format(-coefs[0]),
            xy=(.35, .98),
            xycoords='axes fraction',
            ha='right',
            va='top',
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.3",
                edgecolor="black",
                facecolor="white"
            )
        )

        ax7.set_title("Ev/Eg Rates", fontsize=8)
        ax7.set_xlabel('Eg (returns/shot)')
        ax7.set_ylabel('Ev (returns/shot)')
        ax7.set_xlim(0, 8)
        ax7.set_ylim(0, 40)
        ax7.legend(loc='best')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    return


def parallel_odr(dataset, intercepts, maxes, init=-1, lb=-100, ub=-1/100,
                 model=parallel_model, res=parallel_residuals, loss='arctan',
                 f_scale=.1, outlier_removal=False, method='normal',
                 w=[1.0, 0.25]):
    """
    Performs the parallel orthogonal distance regression on the given dataset.
    """

    cats = dataset.shape[1] - 5

    a = [lb] + [0] * cats
    b = [ub] + maxes
    bounds = (a, b)

    initial_params = [init] + intercepts

    beam_columns = [col for col in dataset.columns if 'Beam' in col]

    filtered_data = []
    full_data = []

    data_quant = 0

    for beam in beam_columns:
        beam_data = dataset[dataset[beam] == True][
            ['Eg', 'Ev', 'layer_flag', 'msw_flag', 'cloud_flag_atm'] + beam_columns
        ].copy()

        if outlier_removal is False:
            beam_data['Outlier'] = 1
            full_data.append(
                beam_data[['Eg', 'Ev', 'layer_flag', 'msw_flag',
                           'cloud_flag_atm', 'Outlier'] + beam_columns]
            )
            continue

        if len(beam_data) >= 2:

            if outlier_removal < 1:
                envelope = EllipticEnvelope(
                    contamination=outlier_removal,
                    random_state=42
                )
                envelope.fit(beam_data[['Eg', 'Ev']])
                beam_data['Outlier'] = envelope.predict(beam_data[['Eg', 'Ev']])
                beam_filtered = beam_data[beam_data['Outlier'] == 1]

            elif outlier_removal >= 2:
                outlier_flags = np.zeros(len(beam_data), dtype=bool)

                n = outlier_removal
                n_ = int(max(1, min(n, len(beam_data) - 3)))
                lof = LocalOutlierFactor(n_neighbors=n_, contamination='auto')
                preds = lof.fit_predict(beam_data[['Eg', 'Ev']])
                outlier_flags |= (preds == -1)

                beam_data['Outlier'] = np.where(outlier_flags, -1, 1)
                beam_filtered = beam_data[beam_data['Outlier'] == 1]
        else:
            beam_filtered = beam_data

        filtered_data.append(
            beam_filtered[['Eg', 'Ev', 'layer_flag', 'msw_flag',
                           'cloud_flag_atm'] + beam_columns]
        )
        full_data.append(
            beam_data[['Eg', 'Ev', 'layer_flag', 'msw_flag',
                       'cloud_flag_atm', 'Outlier'] + beam_columns]
        )

        data_quant = max(data_quant, len(beam_data))

    full_dataset = pd.concat(full_data).reset_index(drop=True)

    if outlier_removal is not False:
        filtered_dataset = pd.concat(filtered_data).reset_index(drop=True)
        dataset = filtered_dataset.copy()

    X = dataset.drop(columns=['Ev', 'layer_flag', 'msw_flag', 'cloud_flag_atm'])
    Y = dataset[['Ev']]

    if method == 'bimodal':
        params = least_squares(
            parallel_residuals,
            x0=initial_params,
            args=(X, Y, model, False, w),
            loss=loss,
            bounds=bounds
        )
        params = least_squares(
            parallel_residuals,
            x0=params.x,
            args=(X, Y, model, True, w),
            loss=loss,
            bounds=bounds
        )

    elif loss == 'linear':
        params = least_squares(
            parallel_residuals,
            x0=initial_params,
            args=(X, Y, model, False, w),
            loss=loss,
            bounds=bounds
        )

    else:
        params = least_squares(
            parallel_residuals,
            x0=initial_params,
            args=(X, Y, model, False, w),
            loss=loss,
            f_scale=f_scale,
            bounds=bounds,
            ftol=1e-15,
            xtol=1e-15,
            gtol=1e-15
        )

    lf = dataset.layer_flag.mean()
    msw = dataset.msw_flag.mean()

    bn = [
        int(re.search(r'Beam (\d+)', col).group(1))
        for col in dataset.columns if re.search(r'Beam \d+', col)
    ]

    strong_pv_mean = 0
    weak_pv_mean = 0
    strong_pv_max = 0
    strong_pg_max = 0

    for i, num in enumerate(bn):
        if num % 2 == 1:
            strong_pv_mean += params.x[i + 1]
            strong_pv_max = max(strong_pv_max, params.x[i + 1])
            strong_pg_max = max(strong_pg_max, -params.x[i + 1] / params.x[0])
        else:
            weak_pv_mean += params.x[i + 1]

    if weak_pv_mean != 0:
        pv_ratio = strong_pv_mean / weak_pv_mean
    else:
        pv_ratio = 0

    if ((lf == 0) | (msw == 0)) & (strong_pv_max <= 16) & (strong_pg_max <= 16):
        data_quality = 0
    else:
        data_quality = 1

    return params.x, dataset, full_dataset, data_quality


def plot_graph(coefs, colors, title_date, X, Y, xx, yy, coords,
               beam=None, file_index=None, data_quality=0):
    """
    Plotting function for graph_detail = 1.
    """

    title_color = ['black', 'red']

    fig = plt.figure(figsize=(10, 6))

    if file_index is not None:
        fig.suptitle(
            title_date + ' - N = ' + str(file_index),
            fontsize=18,
            color=title_color[data_quality]
        )
    else:
        fig.suptitle(title_date, fontsize=18, color=title_color[data_quality])

    for i, c in enumerate(colors):
        if beam is not None:
            if c + 1 in beam:
                plt.scatter(X[i], Y[i], s=7, color=cmap3(2 * c + 1), marker='o')
                plt.scatter(xx[c], yy[c], s=7, color=cmap3(2 * c), marker='o')
                plt.plot(
                    np.array([0, 12]),
                    model([coefs[0], coefs[1 + i]], np.array([0, 12])),
                    label=f"Beam {int(c + 1)}",
                    color=cmap3(2 * c),
                    linestyle='--',
                    zorder=3,
                    linewidth=2
                )
        else:
            plt.scatter(X[i], Y[i], s=7, color=cmap3(2 * c + 1), marker='o')
            plt.scatter(xx[c], yy[c], s=7, color=cmap3(2 * c), marker='o')
            plt.plot(
                np.array([0, 12]),
                model([coefs[0], coefs[1 + i]], np.array([0, 12])),
                label=f"Beam {int(c + 1)}",
                color=cmap3(2 * c),
                linestyle='--',
                zorder=3,
                linewidth=2
            )

    plt.annotate(
        r'$\rho_v/\rho_g \approx {:.2f}$'.format(-coefs[0]),
        xy=(.14, .967),
        xycoords='axes fraction',
        ha='right',
        va='top',
        fontsize=14,
        bbox=dict(
            boxstyle="round,pad=0.3",
            edgecolor="black",
            facecolor="white"
        )
    )

    plt.xlabel('Eg (returns/shot)', fontsize=14)
    plt.ylabel('Ev (returns/shot)', fontsize=14)
    plt.xlim(0, 9)
    plt.ylim(0, 9)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='best', fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    return


def pvpg_parallel(dirpath, atl03path, atl08path, coords, width=4, height=4,
                  f_scale=.1, loss='linear', init=-.6, lb=-100, ub=-1/100,
                  file_index=None, model=parallel_model, res=parallel_residuals,
                  odr=parallel_odr, zeros=None, beam_focus=None, y_init=np.max,
                  graph_detail=0, keep_flagged=True, opsys='bad',
                  altitude=None, alt_thresh=80, threshold=1, small_box=1,
                  rebinned=0, res_field='alongtrack', outlier_removal=False,
                  method='normal', landcover='forest', trim_atmospheric=0,
                  w=[1.0, 0.25], sat_flag=0, show_me_the_good_ones=0, WC=0):
    """
    Parallel regression of all tracks on a given overpass.

    Use WC=1 to sample ESA WorldCover from:
        ../scratch/data/WC/{camera}.tif

    WorldCover class 10 is treated as forest.
    """

    polygon = make_box(coords, width, height)
    min_lon, min_lat, max_lon, max_lat = polygon.total_bounds

    km_per_degree_lat = 111
    km_per_degree_lon = 111 * np.cos(np.radians(coords[1]))

    small_box_lat = small_box / km_per_degree_lat
    small_box_lon = small_box / km_per_degree_lon

    lats = np.arange(
        min_lat + small_box_lat / 2,
        max_lat + small_box_lat / 2,
        small_box_lat
    )

    lons = np.arange(
        min_lon + small_box_lon / 2,
        max_lon + small_box_lon / 2,
        small_box_lon
    )

    LATS = []
    LONS = []

    ALL_LATS = []
    ALL_LONS = []

    foldername = dirpath.split('/')[-2]

    Eg = [[] for _ in range(3 * len(lats) * len(lons))]
    Ev = [[] for _ in range(3 * len(lats) * len(lons))]
    trad_cc = [[] for _ in range(3 * len(lats) * len(lons))]
    beam_str = [[] for _ in range(3 * len(lats) * len(lons))]
    beam = [[] for _ in range(3 * len(lats) * len(lons))]
    data_quantity = [[] for _ in range(3 * len(lats) * len(lons))]

    variable_names = [
        'msw_flag', 'night_flag', 'asr', 'canopy_openness',
        'snr', 'segment_cover', 'segment_landcover',
        'h_te_best_fit', 'h_te_std', 'terrain_slope',
        'longitude', 'latitude',
        'cloud_flag_atm', 'layer_flag'
    ]

    if WC != 0:
        variable_names.append('WC')

    var_dict = {}

    for var in variable_names:
        var_dict[var] = [[] for _ in range(3 * len(lats) * len(lons))]

    dataset = [[] for _ in range(3 * len(lats) * len(lons))]
    plotX = [[] for _ in range(3 * len(lats) * len(lons))]
    plotY = [[] for _ in range(3 * len(lats) * len(lons))]
    atl03s = [[] for _ in range(3 * len(lats) * len(lons))]

    slope_init = [[] for _ in range(3 * len(lats) * len(lons))]
    slope_weight = [[] for _ in range(3 * len(lats) * len(lons))]

    A = h5py.File(atl03path, 'r')

    if list(A['orbit_info']['sc_orient'])[0] == 1:
        strong = ['gt1r', 'gt2r', 'gt3r']
        weak = ['gt1l', 'gt2l', 'gt3l']
    elif list(A['orbit_info']['sc_orient'])[0] == 0:
        strong = ['gt3l', 'gt2l', 'gt1l']
        weak = ['gt3r', 'gt2r', 'gt1r']
    else:
        print('Satellite in transition orientation.')
        A.close()
        return 0

    tracks = [strong[0], weak[0], strong[1], weak[1], strong[2], weak[2]]
    beam_names = [f"Beam {i}" for i in range(1, 7)]

    A.close()

    colors = [[] for _ in range(3 * len(lats) * len(lons))]

    mid_date = parse_filename_datetime(atl03path)
    title_date = datetime_to_title(mid_date)
    table_date = datetime_to_date(mid_date)

    intercepts = [[] for _ in range(3 * len(lats) * len(lons))]
    maxes = [[] for _ in range(3 * len(lats) * len(lons))]

    K = 0

    # Load WorldCover once per file if requested.
    wc_da = None
    if WC != 0:
        wc_filepath = find_worldcover_file(foldername)
        wc_da = rioxarray.open_rasterio(
            wc_filepath,
            masked=True
        ).rio.reproject("EPSG:4326")

    for i, gt in enumerate(tracks):

        try:
            atl03 = get_atl03_struct(atl03path, gt, atl08path)
        except (KeyError, ValueError, OSError, IndexError) as e:
            print(f"Failed to open ATL03 file for {foldername} file {file_index}'s beam {i + 1}.")
            continue

        try:
            atl08 = get_atl08_struct(atl08path, gt, atl03)
        except (KeyError, ValueError, OSError) as e:
            print(f"Failed to open ATL08 file for {foldername} file {file_index}'s beam {i + 1}.")
            continue

        atl03.df = atl03.df[
            (atl03.df['lon_ph'] >= min_lon) &
            (atl03.df['lon_ph'] <= max_lon) &
            (atl03.df['lat_ph'] >= min_lat) &
            (atl03.df['lat_ph'] <= max_lat)
        ]

        atl08.df = atl08.df[
            (atl08.df['longitude'] >= min_lon) &
            (atl08.df['longitude'] <= max_lon) &
            (atl08.df['latitude'] >= min_lat) &
            (atl08.df['latitude'] <= max_lat)
        ]

        if rebinned != 0:
            if atl08.df.shape[0] == 0:
                print(f"Nothing in rebinned section for {foldername} file {file_index}'s beam {i + 1}.")
                continue

            atl08.df = rebin_atl08(atl03, atl08, gt, rebinned, res_field)

        atl08.df = atl08.df[
            (atl08.df.photon_rate_can_nr < 16) &
            (atl08.df.photon_rate_te < 16)
        ]

        # ------------------------------------------------------------------
        # WORLDCOVER LAND-COVER LOOKUP
        # ------------------------------------------------------------------
        if WC != 0:
            if atl08.df.shape[0] == 0:
                atl08.df['WC'] = np.array([], dtype='float32')
            else:
                atl08.df['WC'] = wc_da.sel(band=1).interp(
                    y=("points", atl08.df.latitude.values),
                    x=("points", atl08.df.longitude.values),
                    method="nearest"
                ).values

        # ------------------------------------------------------------------
        # LAND-COVER MASKING
        # ------------------------------------------------------------------
        if landcover == 'forest':
            if WC != 0:
                # ESA WorldCover: class 10 = tree cover
                atl08.df = atl08.df[atl08.df['WC'] == WC_FOREST_VALUE]
            else:
                # Original ATL08/CORINE-style forest mask
                atl08.df = atl08.df[atl08.df['segment_landcover'].isin(
                    [111, 112, 113, 114, 115, 116,
                     121, 122, 123, 124, 125, 126]
                )]

        elif landcover == 'all':
            if WC != 0:
                # Keep valid WorldCover classes, excluding no-data/unknown if present.
                atl08.df = atl08.df[~atl08.df['WC'].isin([0])]
            else:
                atl08.df = atl08.df[~atl08.df['segment_landcover'].isin(
                    [60, 40, 100, 50, 70, 80, 200, 0]
                )]

        if altitude is not None:
            atl08.df = atl08.df[
                abs(atl08.df['h_te_best_fit'] - altitude) <= alt_thresh
            ]

        if trim_atmospheric != 0:
            atl08.df = atl08.df[
                (atl08.df['layer_flag'] == 0) |
                (atl08.df['msw_flag'] == 0)
            ]

        if sat_flag != 0:
            atl08.df = atl08.df[atl08.df['sat_flag'] == 0]

        k = K

        if i % 2 == 0:
            LATS = []
            LONS = []

            lats = np.arange(
                min_lat + small_box_lat / 2,
                max_lat + small_box_lat / 2,
                small_box_lat
            )

            if len(lats) <= 1:
                lats = [(min_lat + max_lat) / 2]

        if i % 2 == 1:
            if len(LONS) == 0:
                continue

            lats, lons = LATS, LONS

        for n, lat in enumerate(lats):

            if i % 2 == 0:
                polygon = make_box((coords[1], lat), width, small_box / 2)
                sub_min_lon, sub_min_lat, sub_max_lon, sub_max_lat = polygon.total_bounds

                atl03_temp = atl03.df[
                    (atl03.df['lat_ph'] >= sub_min_lat) &
                    (atl03.df['lat_ph'] <= sub_max_lat)
                ].copy()

                atl08_temp = atl08.df[
                    (atl08.df['latitude'] >= sub_min_lat) &
                    (atl08.df['latitude'] <= sub_max_lat)
                ].copy()

                if len(atl08_temp) != 0:
                    lon = atl08_temp.longitude.mean()
                else:
                    print(f'Beam {i + 1}, box {n} in {foldername} file {file_index} has no data.')
                    continue

            if i % 2 == 1:
                lon = lons[n]

            polygon = make_box((lon, lat), small_box / 2, small_box / 2)
            sub_min_lon, sub_min_lat, sub_max_lon, sub_max_lat = polygon.total_bounds

            atl03_temp = atl03.df[
                (atl03.df['lon_ph'] >= sub_min_lon) &
                (atl03.df['lon_ph'] <= sub_max_lon) &
                (atl03.df['lat_ph'] >= sub_min_lat) &
                (atl03.df['lat_ph'] <= sub_max_lat)
            ].copy()

            atl08_temp = atl08.df[
                (atl08.df['longitude'] >= sub_min_lon) &
                (atl08.df['longitude'] <= sub_max_lon) &
                (atl08.df['latitude'] >= sub_min_lat) &
                (atl08.df['latitude'] <= sub_max_lat)
            ].copy()

            if atl08_temp.shape[0] < threshold:
                print(f'Beam {i + 1}, box {n} in {foldername} file {file_index} has insufficient data.')

                if i % 2 == 1:
                    k += 1

                continue

            X = atl08_temp.photon_rate_te
            Y = atl08_temp.photon_rate_can_nr

            if i + 1 == 3:
                X /= 0.85
                Y /= 0.85

            layer_flag = atl08_temp.layer_flag
            msw_flag = atl08_temp.msw_flag
            cloud_flag_atm = atl08_temp.cloud_flag_atm

            plotX[k].append(X)
            plotY[k].append(Y)

            if i % 2 == 0:
                LATS.append(lat)
                LONS.append(lon)

            atl03s[k].append(atl03_temp)
            colors[k].append(i)

            Eg[k].append(X)
            Ev[k].append(Y)
            data_quantity[k].append([len(X) for x in range(len(X))])

            trad_cc[k].append(
                (atl08_temp['n_ca_photons'] + atl08_temp['n_toc_photons']) /
                (
                    atl08_temp['n_ca_photons'] +
                    atl08_temp['n_toc_photons'] +
                    atl08_temp['n_te_photons']
                )
            )

            for var in variable_names:
                var_dict[var][k].append(atl08_temp[var])

            if i % 2 == 0:
                beam_str[k].append(
                    ['strong' for _ in range(len(atl08_temp['n_ca_photons']))]
                )
            else:
                beam_str[k].append(
                    ['weak' for _ in range(len(atl08_temp['n_ca_photons']))]
                )

            beam[k].append([i + 1 for _ in range(len(atl08_temp['n_ca_photons']))])

            for x, y, lf, mf, cfa in zip(X, Y, layer_flag, msw_flag, cloud_flag_atm):
                dataset[k].append([x, y, beam_names[i], lf, mf, cfa])

            intercept, slope = starting_intercept(X, Y)

            slope_init[k].append(min(max(slope, -100 + 1e-3), -1/100 - 1e-3))
            slope_weight[k].append(len(Y))
            intercepts[k].append(min(intercept, 16))
            maxes[k].append(16)

            k += 1

            continue

        if i % 2 == 0:
            ALL_LATS.extend(LATS)
            ALL_LONS.extend(LONS)

        if i % 2 == 1:
            LATS = []
            LONS = []
            K = k

    rows = []

    k = 0

    for lat, lon in zip(ALL_LATS, ALL_LONS):
        if len(dataset[k]) == 0:
            k += 1
            continue

        slope_weight[k] /= np.sum([slope_weight[k]])
        slope_init[k] = np.dot(slope_init[k], slope_weight[k])

        df = pd.DataFrame(
            dataset[k],
            columns=['Eg', 'Ev', 'gt', 'layer_flag', 'msw_flag', 'cloud_flag_atm']
        )

        df_encoded = pd.get_dummies(df, columns=['gt'], prefix='', prefix_sep='')

        coefs, xy, full_xy, data_quality = odr(
            df_encoded,
            intercepts=intercepts[k],
            maxes=maxes[k],
            init=slope_init[k],
            lb=lb,
            ub=ub,
            model=model,
            res=res,
            loss=loss,
            f_scale=f_scale,
            outlier_removal=outlier_removal,
            method=method,
            w=w
        )

        xx = [[] for _ in range(6)]
        yy = [[] for _ in range(6)]

        beams_in_play = []

        for i in range(1, 7):
            if f'Beam {i}' in xy.columns:
                xx[i - 1] = xy[xy[f'Beam {i}'] == True]['Eg']
                yy[i - 1] = xy[xy[f'Beam {i}'] == True]['Ev']
                beams_in_play.append(i)

        if show_me_the_good_ones == 0 or data_quality == 0:

            if len(colors) == 0:
                graph_detail = 0

            if graph_detail == 3:
                plot_parallel(
                    atl03s=atl03s[k],
                    coefs=coefs,
                    colors=colors[k],
                    title_date=title_date,
                    X=plotX[k],
                    Y=plotY[k],
                    xx=xx,
                    yy=yy,
                    beam=beam_focus,
                    file_index=file_index,
                    three=True,
                    data_quality=0
                )

            elif graph_detail == 2:
                plot_parallel(
                    atl03s=atl03s[k],
                    coefs=coefs,
                    colors=colors[k],
                    title_date=title_date,
                    X=plotX[k],
                    Y=plotY[k],
                    xx=xx,
                    yy=yy,
                    beam=beam_focus,
                    file_index=file_index,
                    data_quality=data_quality
                )

            elif graph_detail == 1:
                plot_graph(
                    coefs=coefs,
                    colors=colors[k],
                    title_date=title_date,
                    X=plotX[k],
                    Y=plotY[k],
                    xx=xx,
                    yy=yy,
                    coords=(lat, lon),
                    beam=beam_focus,
                    file_index=file_index,
                    data_quality=data_quality
                )

        indices_to_insert = [i for i in range(1, 7) if i not in beams_in_play]

        for index in indices_to_insert:
            coefs = np.insert(coefs, index, None)

        if np.all(np.isnan([coefs[1], coefs[3], coefs[5]])):
            y_strong = np.nan
        else:
            y_strong = np.nanmean([coefs[1], coefs[3], coefs[5]])
            y_strong_max = np.nanmax([coefs[1], coefs[3], coefs[5]])

        if np.all(np.isnan([coefs[2], coefs[4], coefs[6]])):
            y_weak = np.nan
        else:
            y_weak = np.nanmean([coefs[2], coefs[4], coefs[6]])
            y_weak_max = np.nanmax([coefs[2], coefs[4], coefs[6]])

        if np.any(np.isnan([y_strong, y_weak])):
            pv_ratio_mean = np.nan
            pv_ratio_max = np.nan
        else:
            pv_ratio_mean = y_strong / y_weak
            pv_ratio_max = y_strong_max / y_weak_max

        y_intercept_dict = {
            1: coefs[1],
            2: coefs[2],
            3: coefs[3],
            4: coefs[4],
            5: coefs[5],
            6: coefs[6]
        }

        x_intercept_dict = {
            1: -coefs[1] / coefs[0],
            2: -coefs[2] / coefs[0],
            3: -coefs[3] / coefs[0],
            4: -coefs[4] / coefs[0],
            5: -coefs[5] / coefs[0],
            6: -coefs[6] / coefs[0]
        }

        for j in range(len(non_negative_subset(Eg[k]))):
            current_beam = non_negative_subset(beam[k])[j]

            row_data = [
                foldername,
                table_date,
                lon,
                lat,
                -coefs[0],
                y_intercept_dict[current_beam],
                x_intercept_dict[current_beam],
                non_negative_subset(Eg[k])[j],
                non_negative_subset(Ev[k])[j],
                non_negative_subset(data_quantity[k])[j],
                data_quality,
                altitude,
                pv_ratio_mean,
                pv_ratio_max,
                non_negative_subset(trad_cc[k])[j],
                current_beam,
                non_negative_subset(beam_str[k])[j]
            ]

            row_data.append(full_xy['Outlier'].iloc[j])

            for var in variable_names:
                row_data.append(non_negative_subset(var_dict[var][k])[j])

            rows.append(row_data)

        k += 1

    columns_list = [
        'camera', 'date', 'lon', 'lat', 'pvpg', 'pv', 'pg',
        'Eg', 'Ev', 'data_quantity', 'data_quality', 'altitude',
        'pv_ratio_mean', 'pv_ratio_max', 'trad_cc', 'beam', 'beam_str',
        'outlier'
    ]

    for var in variable_names:
        columns_list.append(var)

    BIG_DF = pd.DataFrame(rows, columns=[columns_list])
    BIG_DF.columns = BIG_DF.columns.get_level_values(0)

    return BIG_DF