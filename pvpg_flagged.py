from imports import os, glob, pdb, np, h5py, pd, xr, gpd, Proj, Transformer, CRS, \
                        plt, cmap, Model, Data, ODR, datetime, rasterio, show, \
                        ccrs, cfeature
from classes_fixed import *

def parse_filename_datetime(filename):
    # Extracting only the filename from the full path
    filename_only = filename.split('/')[-1]
    date_str = filename_only.split('_')[2][:14]  # Extracting yyyymmddhhmmss part
    datetime_obj = datetime.strptime(date_str, '%Y%m%d%H%M%S')
    return datetime_obj.strftime('%B %d, %Y, %H:%M:%S')

def pvpg_flagged(atl03path, atl08path, j = None):

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

#         if not 'heights' in A[gt].keys():
#             i += 2
#             continue

#         atl03 = ATL03(atl03path, atl08path, gt)
        try:
            atl03 = ATL03(atl03path, atl08path, gt)
        except (KeyError, ValueError, OSError) as e:
            i += 2
            continue
        atl08 = ATL08(atl08path, gt)

        atl03.plot(ax[i])
#         if gt == 'gt3l':
#             print(atl08.df)

        def linear_model(params, x):
            return params[0]*x + params[1]

        linear = Model(linear_model)
        data = Data(atl08.df.Eg,atl08.df.Ev)
        odr = ODR(data, linear, beta0 = [1.0,1.0])
        result = odr.run()
        slope, intercept = result.beta

        if 0 in list(A[gt]['geolocation']['ph_index_beg']):
            ax[i+1].set_title(f"{gt} 100m Photon Rates - Flagged")
        else:
            ax[i+1].set_title(f"{gt} 100m Photon Rates - Fine")
        ax[i+1].scatter(atl08.df.Eg, atl08.df.Ev, s=10)
        ax[i+1].plot([0,-intercept/slope],[intercept,0])
        ax[i+1].set_xlabel('Eg (returns/shot)')
        ax[i+1].set_ylabel('Ev (returns/shot)')
        ax[i+1].set_xlim(0,8)
        ax[i+1].set_ylim(0,8)
        ax[i+1].annotate(r'$\rho_v/\rho_g \approx {:.2f}$'.format(-slope),
                       xy=(.95,.95),
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
