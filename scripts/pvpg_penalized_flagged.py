# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 15:34:49 2023

@author: s1803229
"""

from scripts.imports import os, glob, pdb, np, h5py, pd, xr, gpd, Proj, Transformer, CRS, \
                        plt, cmap, Model, Data, ODR, datetime, rasterio, show, \
                        ccrs, cfeature
from scripts.classes_fixed import *
from scipy.optimize import least_squares

from sklearn.metrics import r2_score, mean_squared_error

def parse_filename_datetime(filename):
    # Extracting only the filename from the full path
    filename_only = filename.split('/')[-1]
    date_str = filename_only.split('_')[2][:14]  # Extracting yyyymmddhhmmss part
    datetime_obj = datetime.strptime(date_str, '%Y%m%d%H%M%S')
    return datetime_obj.strftime('%B %d, %Y, %H:%M:%S')

def pvpg_penalized_flagged(atl03path, atl08path, f_scale = .1, file_index = None):
    
    i = 0
    
    tracks = ['gt1r', 'gt1l', 'gt2r', 'gt2l', 'gt3r', 'gt3l']
    A = h5py.File(atl03path, 'r')
        
    for gt in tracks:
        try:
            if 0 in A[gt]['geolocation']['ph_index_beg']:
                print('File ' + str(file_index) + ' has been skipped.')
                A.close()
                return
                # This block will be executed if 0 is found in the list
        except (KeyError, FileNotFoundError):
            # Handle the exception (e.g., print a message or log the error)
            continue
            
            
    fig, axes = plt.subplots(6, 3, figsize=(8, 16))
    ax = axes.flatten()
    
    # Extracting date and time from the filename
    title_date = parse_filename_datetime(atl03path)
    
    # Set the figure title
    if file_index != None:
        fig.suptitle(title_date + ' - N = ' + str(file_index), fontsize=16)
    else:
        fig.suptitle(title_date, fontsize=16)

    for gt in tracks:
        
        try:
            atl03 = ATL03(atl03path, atl08path, gt)
        except (KeyError, ValueError, OSError) as e:
            i += 3
            continue
        atl08 = ATL08(atl08path, gt)

        atl03.plot(ax[i])

        def model(params, x):
            return params[0]*x + params[1]

        X = atl08.df.Eg
        Y = atl08.df.Ev
        
        def residuals(params, x, y):
            # weights = (1+y**2)/np.sqrt(1+x**2)
            #guess = np.max(y)/np.max(x)
            # regularization_term = 0.01*(params[0]**2 + 1/params[0]**2)
            return np.abs(model(params, x) - y)/np.sqrt(1 + params[0]**2)
        
        initial_guess = [-1,np.max(Y)]
        
        result = least_squares(residuals, initial_guess, loss='arctan', args=(X, Y), f_scale=f_scale, bounds=([-100, 0], [-1/100, 16]))
            
        a_guess, b_guess = result.x
        
        y_pred = model((a_guess, b_guess), X)
        
        r_squared = r2_score(Y, y_pred)
        rmse = np.sqrt(mean_squared_error(Y, y_pred))
        
        ax[i+1].set_title(f"{gt} Photon Rates", fontsize=8)
        ax[i+1].scatter(X, Y, s=4)
        ax[i+1].plot(np.array([-10,20]), model([a_guess, b_guess], np.array([-10,20])), label='Orthogonal Distance Regression', color='red', linestyle='--')
        ax[i+1].set_xlabel('Eg (returns/shot)')
        ax[i+1].set_ylabel('Ev (returns/shot)')
        ax[i+1].set_xlim(0,12)
        ax[i+1].set_ylim(0,12)
        ax[i+1].annotate(r'$\rho_v/\rho_g \approx {:.2f}$'
                         '\n'
                         r'$R^2: {:.2f}$'
                         '\n'
                         r'$RMSE: {:.2f}$'.format(-a_guess, r_squared, rmse),
                       xy=(.95,.95),
                       xycoords='axes fraction',
                       ha='right',
                       va='top',
                       bbox=dict(boxstyle="round,pad=0.3",
                                 edgecolor="black",
                                 facecolor="white"))
        
        
        def residuals(params, x, y):
            weights = np.sqrt(1+y**2)/np.sqrt(1+x**2)
            #guess = np.max(y)/np.max(x)
            regularization_term = 0.01*(params[0]**2 - 1/params[0]**2)
            return weights*np.abs(model(params, x) - y)/np.sqrt(1 + params[0]**2) + regularization_term
        
        initial_guess = [-1,np.max(Y)]
        
        result = least_squares(residuals, initial_guess, loss = 'huber', args=(X, Y), bounds=([-100, 0], [-1/100, 16]))
        
        a_opt, b_opt = result.x
        
        y_pred = model((a_opt, b_opt), X)
        
        r_squared = r2_score(Y, y_pred)
        rmse = np.sqrt(mean_squared_error(Y, y_pred))
        
        ax[i+2].set_title(f"{gt} Photon Rates", fontsize=8)
        ax[i+2].scatter(X, Y, s=4)
        ax[i+2].plot(np.array([-10,20]), model([a_opt, b_opt], np.array([-10,20])), label='Orthogonal Distance Regression', color='red', linestyle='--')
        ax[i+2].set_xlabel('Eg (returns/shot)')
        ax[i+2].set_ylabel('Ev (returns/shot)')
        ax[i+2].set_xlim(0,12)
        ax[i+2].set_ylim(0,12)
        ax[i+2].annotate(r'$\rho_v/\rho_g \approx {:.2f}$'
                         '\n'
                         r'$R^2: {:.2f}$'
                         '\n'
                         r'$RMSE: {:.2f}$'.format(-a_opt, r_squared, rmse),
                       xy=(.95,.95),
                       xycoords='axes fraction',
                       ha='right',
                       va='top',
                       bbox=dict(boxstyle="round,pad=0.3",
                                 edgecolor="black",
                                 facecolor="white"))
        i += 3
        
    A.close()
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust the layout to make room for the suptitle
    plt.show()
    
    plt.close(fig)
    return