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
    
def model(params, x):
    return params[0]*x + params[1]

# Orthogonal Distance Regression Function
def residuals(params, x, y):
    return np.abs(model(params, x) - y)/np.sqrt(1 + params[0]**2)

def pvpg(atl03path, atl08path, j = None):
    """
    Computes pv/pg for each groundtrack and plots each groundtrack alongside its pv/pg plot.
    Most basic form of this function.
    
    atl03path - Path/to/ATL03/file
    atl08path - Path/to/matching/ATL08/file
    j - Index of filepath names in array if cycling through several filepath pairs.
    """
    i = 0
    
    tracks = ['gt1r', 'gt1l', 'gt2r', 'gt2l', 'gt3r', 'gt3l']
    #A = h5py.File(atl03path, 'r')
    
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
   
    
def pvpg_penalized_flagged(atl03path, atl08path,f_scale = [.1,.1], loss = ['arctan','arctan'], bounds0 = ([-100, 0], [-1/100, 16]), bounds1 = ([-100, 0], [-1/100, 16]), file_index = None, res0 = residuals, res1 = residuals, model = model, x_guess = [-1,-1], y_guess = [np.max,np.max]):
    """
    Adjustment of pvpg_penalized where flagged files are simply skipped.

    atl03path - Path/to/ATL03/file
    atl08path - Path/to/matching/ATL08/file
    f_scale - Parameter in least_squares() function when loss is nonlinear, indiciating the value of the soft margin between inlier and outlier residuals. This is an array of two values, one for each regression.
    loss - string for loss parameter in least_squares(). Array of two values, one for each regression.
    bounds0, bounds1 - bounds for slope of line and y-intercept in shape ([--,--],[--,--]), allowing us to restrict the regression to realistic values. One for each regression.
    file_index - Index of file if cycling through an array of filenames, displayed in figure titles for a given file. Allows us to easily pick out strange cases for investigation.
    res - Default holds the ODR residuals function to be used in least_squares(). Can hold adjusted residual functions as well.
    x_guess - Initial guess for regression slope for each regression, default set to -1.
    y_guess - Function used to produce initial guess for y_intercept, default set as highest Ev return in track.
    """
    
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
    
    fig, axes = plt.subplots(6, 3, figsize=(8, 30))
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

        X = atl08.df.Eg
        Y = atl08.df.Ev
        
        initial_guess = [x_guess[0],y_guess[0](Y)]
        
        result = least_squares(res0, initial_guess, loss=loss[0], f_scale=f_scale[0], args=(X, Y), bounds = bounds0)
            
        a_guess, b_guess = result.x
        
        ax[i+1].set_title(f"{gt} Photon Rates", fontsize=8)
        ax[i+1].scatter(X, Y, s=10)
        ax[i+1].plot(np.array([-10,20]), model([a_guess, b_guess], np.array([-10,20])), label='Orthogonal Distance Regression', color='red', linestyle='--')
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
        
        initial_guess = [x_guess[1],y_guess[1](Y)]
        
        result = least_squares(res1, initial_guess, loss=loss[1], f_scale=f_scale[1], args=(X, Y), bounds = bounds1)
        
        a_opt, b_opt = result.x
        
        ax[i+2].set_title(f"{gt} Photon Rates", fontsize=8)
        ax[i+2].scatter(X, Y, s=10)
        ax[i+2].plot(np.array([-10,20]), model([a_opt, b_opt], np.array([-10,20])), label='Orthogonal Distance Regression', color='red', linestyle='--')
        ax[i+2].set_xlabel('Eg (returns/shot)')
        ax[i+2].set_ylabel('Ev (returns/shot)')
        ax[i+2].set_xlim(0,12)
        ax[i+2].set_ylim(0,12)
        ax[i+2].annotate(r'$\rho_v/\rho_g \approx {:.2f}$'.format(-a_opt),
                       xy=(.95,.95),
                       xycoords='axes fraction',
                       ha='right',
                       va='top',
                       bbox=dict(boxstyle="round,pad=0.3",
                                 edgecolor="black",
                                 facecolor="white"))
        i += 3
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust the layout to make room for the suptitle
    plt.show()
    return
    
def pvpg_strongweak(atl03path, atl08path, xlim, ylim, j = None):

    """
    Same as pvpg, but designed purely to show a problem with ground return densities.
    
    atl03path - path/to/ATL03/file/
    atl08path - path/to/ATL08/file/
    xlim - plot limits for x
    ylim - plot limits for y
    j - index of file if cycling through array of filepaths
    """
    
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
        ax[i].set_xlim(xlim)
        ax[i].set_ylim(ylim)
#         if gt == 'gt3l':
#             print(atl08.df)

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
        i += 2
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust the layout to make room for the suptitle
    plt.show()
    return
