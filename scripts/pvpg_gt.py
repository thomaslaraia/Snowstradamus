from scripts.imports import os, glob, pdb, np, h5py, pd, xr, gpd, Proj, Transformer, CRS, \
                        plt, cmap, Model, Data, ODR, datetime, rasterio, show, \
                        ccrs, cfeature
from scripts.classes_fixed import *

def parse_filename_datetime(filename):
    # Extracting only the filename from the full path
    filename_only = filename.split('/')[-1]
    date_str = filename_only.split('_')[2][:14]  # Extracting yyyymmddhhmmss part
    datetime_obj = datetime.strptime(date_str, '%Y%m%d%H%M%S')
    return datetime_obj.strftime('%B %d, %Y, %H:%M:%S')

def pvpg_single(atl03path, atl08path, gt):

    i = 0

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax = axes.flatten()

    # Extracting date and time from the filename
    title_date = parse_filename_datetime(atl03path)

    # Set the figure title
    fig.suptitle(title_date, fontsize=16)

    try:
        atl03 = ATL03(atl03path, atl08path, gt)
    except (KeyError, ValueError, OSError) as e:
        print('Sucks to suck.')
    atl08 = ATL08(atl08path, gt)

    atl03.plot(ax[i])

    def linear_model(params, x):
        return params[0]*x + params[1]

    linear = Model(linear_model)
    data = Data(atl08.df.Eg,atl08.df.Ev)
    odr = ODR(data, linear, beta0 = [1.0,1.0])
    result = odr.run()
    slope, intercept = result.beta

    ax[i+1].set_title(f"{gt} 100m Photon Rates")
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

    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust the layout to make room for the suptitle
    plt.show()
    return
