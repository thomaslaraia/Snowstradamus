from scripts.imports import os, glob, pdb, np, h5py, pd, xr, gpd, Proj, Transformer, CRS, \
                        plt, cmap, cmap2, Model, Data, ODR, datetime, rasterio, show, \
                        ccrs, cfeature
from scripts.classes_fixed import *
from scipy.optimize import least_squares
from sklearn.metrics import r2_score, mean_squared_error
from scripts.ransac import run_ransac, plot_ransac
from scripts.odr import odr#, parallel_odr, parallel_residuals

def parse_filename_datetime(filename):
    # Extracting only the filename from the full path
    filename_only = filename.split('/')[-1]
    date_str = filename_only.split('_')[2][:14]  # Extracting yyyymmddhhmmss part
    datetime_obj = datetime.strptime(date_str, '%Y%m%d%H%M%S')
    return datetime_obj.strftime('%B %d, %Y, %H:%M:%S')
    
def model(params, x):
    return params[0]*x + params[1]

# Orthogonal Distance Regression Function
def residuals(params, x, y):
    return (y - model(params, x))/np.sqrt(1 + params[0]**2)

def pvpg(atl03path, atl08path, j = None):
    """
    Computes pv/pg for each groundtrack and plots each groundtrack alongside its pv/pg plot.
    Most basic form of this function.
    
    atl03path - Path/to/ATL03/file
    atl08path - Path/to/matching/ATL08/file
    j - Index of filepath names in array if cycling through several filepath pairs.
    """
    i = 0
    
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
        return
        
    tracks = [strong[0], weak[0], strong[1], weak[1], strong[2], weak[2]]
    
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
            # Simple representation of linear model as a function
            return params[0]*x + params[1]
        
        #Orthogonal Distance Regression with simple initial guess
        linear = Model(linear_model)
        data = Data(atl08.df.Eg,atl08.df.Ev)
        odr = ODR(data, linear, beta0 = [-1.0,np.max(atl08.df.Ev)])
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
        i += 2
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust the layout to make room for the suptitle
    plt.show()
    return
    
def pvpg_flagged(atl03path, atl08path, j = None):
    """
    Adjusted pvpg function that checks the ATL03 file for 100m segments with zero photon
    returns and marks such tracks as flagged.
    
    atl03path - Path/to/ATL03/file
    atl08path - Path/to/matching/ATL08/file
    j - Index of filepath names in array if cycling through several filepath pairs.
    """

    i = 0
    
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
        return
        
    tracks = [strong[0], weak[0], strong[1], weak[1], strong[2], weak[2]]

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
   
    
def pvpg_penalized_flagged(atl03path, atl08path,f_scale = .1, loss = 'linear', bounds = ([-100, 0], [-1/100, 16]), file_index = None, res = residuals, model = model, rt = None, zeros=False):
    """
    Adjustment of pvpg_penalized where flagged files are simply skipped.

    atl03path - Path/to/ATL03/file
    atl08path - Path/to/matching/ATL08/file
    f_scale - Parameter in least_squares() function when loss is nonlinear, indiciating the value of the soft margin between inlier and outlier residuals.
    loss - string for loss parameter in least_squares().
    bounds - bounds for slope of line and y-intercept in shape ([--,--],[--,--]), allowing us to restrict the regression to realistic values.
    file_index - Index of file if cycling through an array of filenames, displayed in figure titles for a given file. Allows us to easily pick out strange cases for investigation.
    res - Default holds the ODR residuals function to be used in least_squares(). Can hold adjusted residual functions as well.
    x_guess - Initial guess for regression slope for each regression, default set to -1.
    y_guess - Function used to produce initial guess for y_intercept, default set as highest Ev return in track.
    rt - this will trigger RANSAC regression, and is also equal to the residual threshold of the regression.
    """
    
    i = 0
    
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
        return
        
    tracks = [strong[0], weak[0], strong[1], weak[1], strong[2], weak[2]]
        
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
    
    fig, axes = plt.subplots(6, 2, figsize=(8, 20))
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
            i += 2
            continue
            
        if zeros == False:
            atl08 = ATL08(atl08path, gt)
        else:
            atl08 = ATL08_with_zeros(atl08path, gt)

        atl03.plot(ax[i])

        X = atl08.df.Eg
        Y = atl08.df.Ev
        
        init = [-1, np.max(Y)]
        
        if rt != None:
            a_guess, b_guess, ransac_model, inlier_mask = run_ransac(X, Y, loss=loss, rt=rt)
            ax[i+1] = plot_ransac(X, Y, ransac_model, inlier_mask, ax[i+1])
        
        else:
            a_guess, b_guess = odr(X, Y, res = res, init=init, loss=loss, bounds=bounds, f_scale=f_scale)
            ax[i+1].scatter(X, Y, s=10)
            ax[i+1].plot(np.array([-10,20]), model([a_guess, b_guess], np.array([-10,20])), label='Orthogonal Distance Regression', color='red', linestyle='--')
        
        ax[i+1].set_title(f"{gt} Photon Rates, Beam {int(i/2 + 1)}", fontsize=8)
        ax[i+1].set_xlabel('Eg (returns/shot)')
        ax[i+1].set_ylabel('Ev (returns/shot)')
        ax[i+1].set_xlim(0,12)
        ax[i+1].set_ylim(0,12)
        ax[i+1].annotate(r'$\rho_v/\rho_g \approx {:.2f}$'.format(-a_guess),
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
