from scripts.imports import os, glob, pdb, np, h5py, pd, xr, gpd, Proj, Transformer, CRS, \
                        plt, cmap, Model, Data, ODR, datetime, rasterio, show, \
                        ccrs, cfeature

import sys

sys.path.insert(1,'/home/s1803229/src/PhoREAL')

from phoreal.reader import get_atl03_struct, get_atl08_struct

def plot(df, ax):
    class_dict = {-1: {'color':cmap(0.98),
                       'name':'Unclassified'},
                   0: {'color':cmap(0.2),
                       'name':'Noise'},
                   1: {'color':cmap(0.8),
                       'name':'Ground'},
                   2: {'color':cmap(0.4),
                       'name':'Canopy'},
                   3: {'color':cmap(0.6),
                       'name':'Top of canopy'}}

    if 'classification' in df.columns:
        for c in np.unique(df.classification):
            mask = df.classification==c
            ax.scatter(df[mask].lat_ph,
                       df[mask].h_ph,
                       color=class_dict[c]['color'],
                       label=class_dict[c]['name'],
                       s = 3)

            ax.legend(loc='best')
    else:
            ax.scatter(df.lat_ph,
                      df.h_ph,
                      s = 3)
    ax.set_xlabel('Latitude (°)')
    ax.set_ylabel('Elevation (m)')
    return

def parse_filename_datetime(filename):
    # Extracting only the filename from the full path
    filename_only = filename.split('/')[-1]
    date_str = filename_only.split('_')[2][:14]  # Extracting yyyymmddhhmmss part
    datetime_obj = datetime.strptime(date_str, '%Y%m%d%H%M%S')
    return datetime_obj.strftime('%B %d, %Y, %H:%M:%S')

def pvpg(atl03path, atl08path, j=None):

    i = 0

    tracks = ['gt1r', 'gt1l', 'gt2r', 'gt2l', 'gt3r', 'gt3l']
    A = h5py.File(atl03path, 'r')

    fig, axes = plt.subplots(6, 2, figsize=(8, 30))
    ax = axes.flatten()

    # Extracting date and time from the filename
    title_date = parse_filename_datetime(atl03path)

	# Set the figure title
    if j != None:
        fig.suptitle(title_date + ' - N = ' + str(j), fontsize=16)
    else:
        fig.suptitle(title_date, fontsize=16)

    for gt in tracks:

        if not 'heights' in A[gt].keys():
            i += 2
            continue

        try:
            atl03 = get_atl03_struct(atl03path, gt, atl08path)
        except (KeyError, ValueError, OSError) as e:
            i += 2
            continue
        try:
        	atl08 = get_atl08_struct(atl08path, gt)
        except (KeyError, ValueError, OSError) as e:
            i += 2
            continue

        # Set the ground track title
        ax[i].set_title(gt)
        plot(atl03.df, ax[i])

        def linear_model(params, x):
            return params[0] * x + params[1]

        atl08_ = atl08.df[(atl08.df.photon_rate_can_nr < 100) & (atl08.df.photon_rate_te < 100)]

        linear = Model(linear_model)
        data = Data(atl08_.photon_rate_te, atl08_.photon_rate_can_nr)
        odr = ODR(data, linear, beta0=[1.0, 1.0])
        result = odr.run()
        slope, intercept = result.beta

        # Set the title for the 100m Photon Rates plot
        ax[i + 1].set_title(f"{gt} 100m Photon Rates")

        ax[i + 1].scatter(atl08_.photon_rate_te, atl08_.photon_rate_can_nr, s=10)
        ax[i + 1].plot([0, -intercept/slope], [intercept, 0])
        ax[i + 1].set_xlabel('Eg (returns/shot)')
        ax[i + 1].set_ylabel('Ev (returns/shot)')
        ax[i + 1].set_xlim(0,8)
        ax[i + 1].set_ylim(0,8)
        ax[i + 1].annotate(r'$\rho_v/\rho_g \approx {:.2f}$'.format(-slope),
                           xy=(.95, .95),
                           xycoords='axes fraction',
                           ha='right',
                           va='top',
                           bbox=dict(boxstyle="round,pad=0.3",
                                     edgecolor="black",
                                     facecolor="white"))
        i += 2

    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust the layout to make room for the suptitle
    plt.show()
    return
