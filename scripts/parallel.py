from scripts.imports import os, glob, pdb, np, h5py, pd, xr, gpd, Proj, Transformer, CRS, \
                        plt, cmap, cmap2, Model, Data, ODR, datetime, rasterio, show, \
                        ccrs, cfeature
from scripts.classes_fixed import *
from scripts.pvpg_fixed import *
from scripts.track_pairs import *
from scripts.show_tracks import *

def pvpg_parallel_method1(atl03path, atl08path,f_scale = .1, loss = 'arctan', init = -1, lb = -100, ub = -1/100, file_index = None, model = model, rt = None, zeros=False, beam = None, y_init = np.max):
    """
    Parallel regression of all tracks on a given overpass.

    atl03path - Path/to/ATL03/file
    atl08path - Path/to/matching/ATL08/file
    f_scale - Parameter in least_squares() function when loss is nonlinear, indiciating the value of the soft margin between inlier and outlier residuals.
    loss - string for loss parameter in least_squares().
    lb - Lower bound of allowed value for the slope of the regression, default -100
    ub - Upper bound of allowed value for the slope of the regression, default -1/100
    file_index - Index of file if cycling through an array of filenames, displayed in figure titles for a given file. Allows us to easily pick out strange cases for investigation.
    res - Default holds the ODR residuals function to be used in least_squares(). Can hold adjusted residual functions as well.
    model - model function to be used, e.g. params[0]*x + params[1]
    rt - this will trigger RANSAC regression, and is also equal to the residual threshold of the regression.
    """
    
    def parallel_model(params, x):
        common_slope, *intercepts = params
        return [common_slope * x[i] + intercept for i, intercept in enumerate(intercepts)]

    def parallel_residuals(params, x, *datasets):
        model_outputs = parallel_model(params, x)
    
        all_residuals=[]
        for model_output, dataset in zip(model_outputs, datasets):
            residuals = (dataset - model_output)/np.sqrt(1 + params[0]**2)
            non_nan_residuals = residuals[~np.isnan(residuals)]
            all_residuals.extend(non_nan_residuals)
        return np.array(all_residuals)
    
    def parallel_odr(datasets, init = -1, lb = -100, ub = -1/100, res = parallel_residuals, loss='linear', f_scale=.1, y_init=np.max):
        xs, ys = zip(*datasets)
    
        a = [lb] + [0]*len(ys)
        b = [ub] + [16]*len(ys)
    
        bounds = (a,b)
    
        initial_params = [init] + [y_init(y) for y in ys]

        params = least_squares(res, x0=initial_params, args=(xs, *ys), loss = loss, f_scale=f_scale, bounds = bounds).x
    
        return params
    
    i = 0

    datasets = []
    
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
    beam_names = [f"Beam {i}" for i in range(1,7)]
        
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
    
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(331)
    ax2 = fig.add_subplot(332)
    ax3 = fig.add_subplot(334)
    ax4 = fig.add_subplot(335)
    ax5 = fig.add_subplot(337)
    ax6 = fig.add_subplot(338)
    axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    ax7 = fig.add_subplot(133)
    
    # Extracting date and time from the filename
    title_date = parse_filename_datetime(atl03path)

    #Keep indices of colors to plot regression lines later:
    colors = []

    # Set the figure title
    if file_index != None:
        fig.suptitle(title_date + ' - N = ' + str(file_index), fontsize=16)
    else:
        fig.suptitle(title_date, fontsize=16)

    for i, gt in enumerate(tracks):
        
        try:
            atl03 = ATL03(atl03path, atl08path, gt)
        except (KeyError, ValueError, OSError) as e:
            i += 1
            continue
        if zeros == False:
            atl08 = ATL08(atl08path, gt)
        else:
            atl08 = ATL08_with_zeros(atl08path, gt)

        atl03.plot_small(axes[i], beam_names[i])

        X = atl08.df.Eg.astype(dtype=np.float32)
        Y = atl08.df.Ev.astype(dtype=np.float32)

        datasets.append((X,Y))
        
        if beam != None:
            if i == beam - 1:
                ax7.scatter(X,Y, s=5, color=cmap2(i))
        else:
            ax7.scatter(X,Y, s=5, color=cmap2(i))

        colors.append(i)

        i += 1

    coefs = parallel_odr(datasets, init = init, lb=lb, ub=ub, res = parallel_residuals, loss=loss, f_scale=f_scale, y_init = y_init)
    print(coefs)
    for i, c in enumerate(colors):
        if beam != None:
            if i == beam - 1:
                ax7.plot(np.array([0,12]), model([coefs[0], coefs[1+i]], np.array([0,12])), label=f"Beam {int(i+1)}", color=cmap2(c), linestyle='--', zorder=3)
        else:
            ax7.plot(np.array([0,12]), model([coefs[0], coefs[1+i]], np.array([0,12])), label=f"Beam {int(i+1)}", color=cmap2(c), linestyle='--', zorder=3)
    ax7.annotate(r'$\rho_v/\rho_g \approx {:.2f}$'.format(-coefs[0]),
                   xy=(.35,.98),
                   xycoords='axes fraction',
                   ha='right',
                   va='top',
                   fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.3",
                             edgecolor="black",
                             facecolor="white"))
    
    ax7.set_title(f"Ev/Eg Rates", fontsize=8)
    ax7.set_xlabel('Eg (returns/shot)')
    ax7.set_ylabel('Ev (returns/shot)')
    ax7.set_xlim(0,8)
    ax7.set_ylim(0,20)
    ax7.legend(loc='best')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust the layout to make room for the suptitle
    plt.show()
    return
    
    
    
    
    


def pvpg_parallel_method2(atl03path, atl08path,f_scale = .1, loss = 'arctan', init = -1, lb = -100, ub = -1/100, file_index = None, model = model, rt = None, zeros=False, beam = None, y_init = np.max):
    """
    Parallel regression of all tracks on a given overpass.

    atl03path - Path/to/ATL03/file
    atl08path - Path/to/matching/ATL08/file
    f_scale - Parameter in least_squares() function when loss is nonlinear, indiciating the value of the soft margin between inlier and outlier residuals.
    loss - string for loss parameter in least_squares().
    lb - Lower bound of allowed value for the slope of the regression, default -100
    ub - Upper bound of allowed value for the slope of the regression, default -1/100
    file_index - Index of file if cycling through an array of filenames, displayed in figure titles for a given file. Allows us to easily pick out strange cases for investigation.
    res - Default holds the ODR residuals function to be used in least_squares(). Can hold adjusted residual functions as well.
    model - model function to be used, e.g. params[0]*x + params[1]
    rt - this will trigger RANSAC regression, and is also equal to the residual threshold of the regression.
    """
    def parallel_model(params, x):
        # print(x)
        common_slope, *parallel = params

        # Get all columns starting with 'Beam'
        beam_columns = [col for col in x.columns if col.startswith('Beam')]
        return common_slope*x['Eg'] + np.dot(x[beam_columns], parallel)

    def parallel_residuals(params, x, y):
        model_output = parallel_model(params, x)
        # print(y.T.values[0])
        return (y.T.values[0] - model_output)/np.sqrt(1 + params[0]**2)

    def parallel_odr(dataset, maxes, init = -1, lb = -100, ub = -1/100, res = parallel_residuals, loss='linear', f_scale=.1):
        cats = dataset.shape[1]-2
    
        a = [lb] + [0]*cats
        b = [ub] + [16]*cats
    
        bounds = (a,b)
    
        initial_params = [init] + maxes
        # print(initial_params)

        X = dataset.drop(columns=['Ev'])
        Y = dataset[['Ev']]

        params = least_squares(res, x0=initial_params, args=(X, Y), loss = loss, f_scale=f_scale, bounds = bounds, ftol = 1e-15, xtol=1e-15, gtol=1e-15).x
    
        return params
    
    i = 0

    dataset = []
    
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
    beam_names = [f"Beam {i}" for i in range(1,7)]
        
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
    
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(331)
    ax2 = fig.add_subplot(332)
    ax3 = fig.add_subplot(334)
    ax4 = fig.add_subplot(335)
    ax5 = fig.add_subplot(337)
    ax6 = fig.add_subplot(338)
    axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    ax7 = fig.add_subplot(133)
    
    # Extracting date and time from the filename
    title_date = parse_filename_datetime(atl03path)

    #Keep indices of colors to plot regression lines later:
    colors = []

    maxes = []

    # Set the figure title
    if file_index != None:
        fig.suptitle(title_date + ' - N = ' + str(file_index), fontsize=16)
    else:
        fig.suptitle(title_date, fontsize=16)

    for i, gt in enumerate(tracks):
        
        try:
            atl03 = ATL03(atl03path, atl08path, gt)
        except (KeyError, ValueError, OSError) as e:
            i += 1
            continue
        if zeros == False:
            atl08 = ATL08(atl08path, gt)
        else:
            atl08 = ATL08_with_zeros(atl08path, gt)

        atl03.plot_small(axes[i], gt)

        X = atl08.df.Eg
        Y = atl08.df.Ev
        for x, y in zip(X,Y):
            dataset.append([x, y, beam_names[i]])
            
        if beam != None:
            if i == beam - 1:
                ax7.scatter(X,Y, s=5, color=cmap2(i))
        else:
            ax7.scatter(X,Y, s=5, color=cmap2(i))

        colors.append(i)
        maxes.append(y_init(Y))

        i += 1

    # Create DataFrame
    df = pd.DataFrame(dataset, columns=['Eg', 'Ev', 'gt'])

    # Dummy encode the categorical variable
    df_encoded = pd.get_dummies(df, columns=['gt'], prefix='', prefix_sep='')
    
    coefs = parallel_odr(df_encoded, maxes = maxes, init = init, lb=lb, ub=ub, res = parallel_residuals, loss=loss, f_scale=f_scale)
    print(coefs)
    for i, c in enumerate(colors):
        if beam != None:
            if i == beam - 1:
                ax7.plot(np.array([0,12]), model([coefs[0], coefs[1+i]], np.array([0,12])), label=f"Beam {int(i+1)}", color=cmap2(c), linestyle='--', zorder=3)
        else:
            ax7.plot(np.array([0,12]), model([coefs[0], coefs[1+i]], np.array([0,12])), label=f"Beam {int(i+1)}", color=cmap2(c), linestyle='--', zorder=3)
    ax7.annotate(r'$\rho_v/\rho_g \approx {:.2f}$'.format(-coefs[0]),
                   xy=(.35,.98),
                   xycoords='axes fraction',
                   ha='right',
                   va='top',
                   fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.3",
                             edgecolor="black",
                             facecolor="white"))
    
    ax7.set_title(f"Ev/Eg Rates", fontsize=8)
    ax7.set_xlabel('Eg (returns/shot)')
    ax7.set_ylabel('Ev (returns/shot)')
    ax7.set_xlim(0,8)
    ax7.set_ylim(0,20)
    ax7.legend(loc='best')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust the layout to make room for the suptitle
    plt.show()
    return
