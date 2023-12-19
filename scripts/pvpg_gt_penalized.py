from scripts.imports import os, glob, pdb, np, h5py, pd, xr, gpd, Proj, Transformer, CRS, \
                        plt, cmap, Model, Data, ODR, datetime, rasterio, show, \
                        ccrs, cfeature
from scripts.classes_fixed import *
from sklearn.linear_model import HuberRegressor
from scipy.optimize import least_squares

def parse_filename_datetime(filename):
    # Extracting only the filename from the full path
    filename_only = filename.split('/')[-1]
    date_str = filename_only.split('_')[2][:14]  # Extracting yyyymmddhhmmss part
    datetime_obj = datetime.strptime(date_str, '%Y%m%d%H%M%S')
    return datetime_obj.strftime('%B %d, %Y, %H:%M:%S')

def pvpg_gt_penalized(atl03path, atl08path, gt):

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

    def model(params, x):
        return params[0]*x + params[1]
        
    def residuals(params, x, y):
        return y - model(params, x)
        
    initial_guess = [1.0,1.0]
    
    X = atl08.df.Eg
    Y = atl08.df.Ev
    
    # Use least_squares to perform orthogonal distance regression with HuberRegressor
    result = least_squares(residuals, initial_guess, loss='huber', f_scale=1.0, args=(X, Y))
    
    # Extract the optimized parameters
    a_opt, b_opt = result.x
    
    # Create a HuberRegressor for comparison
    huber_regressor = HuberRegressor()
    huber_regressor.fit(X.values.reshape(-1,1), Y.values)
    a_huber, b_huber = huber_regressor.coef_, huber_regressor.intercept_

    ax[i+1].set_title(f"{gt} 100m Photon Rates")
    ax[i+1].scatter(X, Y, s=10)
    ax[i+1].plot(X, model([a_opt, b_opt], X), label='Orthogonal Distance Regression')
    ax[i+1].plot(X, model([a_huber, b_huber], X), label='Huber Regressor')
    ax[i+1].set_xlabel('Eg (returns/shot)')
    ax[i+1].set_ylabel('Ev (returns/shot)')
    ax[i+1].set_xlim(0,12)
    ax[i+1].set_ylim(0,12)
    ax[i+1].annotate(r'$\rho_v/\rho_g \approx {:.2f}$'.format(-huber_regressor.coef_[0]),
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
