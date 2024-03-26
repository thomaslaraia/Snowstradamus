from scripts.imports import os, glob, pdb, np, h5py, pd, xr, gpd, Proj, Transformer, CRS, \
                        plt, cmap, cmap2, Model, Data, ODR, datetime, rasterio, show, \
                        ccrs, cfeature
from scripts.classes_fixed import *
from scripts.pvpg_concise import *
from scripts.track_pairs import *
from scripts.show_tracks import *
from scipy.optimize import least_squares
from sklearn.metrics import r2_score, mean_squared_error
from scripts.odr import odr

# This function is called if the graph_detail is set to 2!
# I know I used different coding structure for this one but
# all I can really say is whoops and move on.
def plot_parallel(atl03s, coefs, colors, title_date, X, Y, beam = None, canopy_frac = None, terrain_frac = None, file_index=None, three=None):
    """
    Plotting function of pvpg_parallel. Shows a regression line for each available groudntrack in a bigger plot, as well as groundtrack visualisations in a smaller plot.
    
    atl03s - This is an array of ATL03 objects, one for each groundtrack that was successfully turned into an object. If only Beams 5 and 6 exist, then this has two objects in it, one for each of those beams.
    coefs - Array of parameters that are optimized, starting with the slope in coefs[0] and another parameter for each beam to control the y-intercept.
    colors - This holds the integers minus one of the beams that have groundtracks in the file. This is to keep the coloring in the plots consistent for each beam across all files.
    title_date - This is just the data and time of the ICESat-2 overpass. The parse_filename_datetime() function will take care of this for you.
    X - Array of each Eg dataset, [[data1],[data2],...]. This always has six arrays in it, one for each groundtrack from Beam 1 to Beam 6. If nothing is read, you get an empty array [], e.g. [[data1],[],[data3],...]
    Y - Array of each Ev dataset, see X description.
    beam - An array of beams to focus on. For example, if you only want to see pv/pg information on the plot for Beams 3 and 4, then you would set beam = [3,4]. Default is None, and all beams are shown.
    file_index - Default set to None. If changed, this will show the index of the file in an array of all ATL03 file paths so that it is easy to find and focus on interesting cases. Works if you are in a loop of filepaths and you need to know which one is being funky.
    canopy_frac - Default is None. If changed, this will say in the title of the groundtrack what percentage of the data has canopy photon data. Low canopy fraction could indicate poor quality data. This is only displayed if Detail = 2.
    """

    # Simple array of all the beam names
    beam_names = [f"Beam {i}" for i in range(1,7)]
    
    # Six small figures for groundtracks and one for the pv/pg plot
    fig = plt.figure(figsize=(10, 12))
    if three == None:
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
    
    # Set the figure title
    if file_index != None:
        fig.suptitle(title_date + ' - N = ' + str(file_index), fontsize=16)
    else:
        fig.suptitle(title_date, fontsize=16)
    
    # we go through each color and atl03 object together.
    # In this loop, we plot all of the groundtracks where they belong
    # depending on which beam it is and plot the data in the scatterplot
    for i, c, atl03 in zip(np.arange(len(colors)),colors, atl03s):
        
        # If there's a canopy fraction wanted, we stick it in the title
        if (canopy_frac != None) & (terrain_frac != None):
            atl03.plot_small(axes[c], f"{beam_names[c]} - TF = {round(terrain_frac[c],2)}, CF = {round(canopy_frac[c],2)}")
        
        elif canopy_frac != None:
            atl03.plot_small(axes[c], f"{beam_names[c]} - CF = {round(canopy_frac[c],2)}")
        
        elif terrain_frac != None:
            atl03.plot_small(axes[c], f"{beam_names[c]} - TF = {round(terrain_frac[c],2)}")
        
        else:
            atl03.plot_small(axes[c], beam_names[c])
        
        # If there's a focus on certain beams, we run this if statement to
        # check if the current beam is in the list of beams the user wants.
        # Then we throw the data onto the scatterplot with the color of choice
        # along with a regression line of the same color
        if three == None:
        
            if beam != None:
                if c + 1 in beam:
                    ax7.scatter(X[c],Y[c], s=5, color=cmap2(c))
                    ax7.plot(np.array([0,12]), model([coefs[0], coefs[1+i]], np.array([0,12])), label=f"Beam {int(c+1)}", color=cmap2(c), linestyle='--', zorder=3)
            else:
                ax7.scatter(X[c],Y[c], s=5, color=cmap2(c))
                ax7.plot(np.array([0,12]), model([coefs[0], coefs[1+i]], np.array([0,12])), label=f"Beam {int(c+1)}", color=cmap2(c), linestyle='--', zorder=3)
    
    
    if three == None:        
        # Show the pv/pg estimate on the plot
        ax7.annotate(r'$\rho_v/\rho_g \approx {:.2f}$'.format(-coefs[0]),
                       xy=(.35,.98),
                       xycoords='axes fraction',
                       ha='right',
                       va='top',
                       fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.3",
                                 edgecolor="black",
                                 facecolor="white"))
    
        # Set all the boring plot details
        ax7.set_title(f"Ev/Eg Rates", fontsize=8)
        ax7.set_xlabel('Eg (returns/shot)')
        ax7.set_ylabel('Ev (returns/shot)')
        ax7.set_xlim(0,8)
        ax7.set_ylim(0,40)
        ax7.legend(loc='best')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust the layout to make room for the suptitle
    plt.show()
    return

# This corresponds to graph_detail = 1
def plot_graph(coefs, colors, title_date, X, Y, beam = None, file_index=None):
    """
    Plotting function of pvpg_parallel. Shows a regression line for each available groudntrack in a bigger plot, as well as groundtrack visualisations in a smaller plot.
    
    coefs - Array of parameters that are optimized, starting with the slope in coefs[0] and another parameter for each beam to control the y-intercept.
    colors - This holds the integers minus one of the beams that have groundtracks in the file. This is to keep the coloring in the plots consistent for each beam across all files.
    title_date - This is just the data and time of the ICESat-2 overpass. The parse_filename_datetime() function will take care of this for you.
    X - Array of each Eg dataset, [[data1],[data2],...]. This always has six arrays in it, one for each groundtrack from Beam 1 to Beam 6. If nothing is read, you get an empty array [], e.g. [[data1],[],[data3],...]
    Y - Array of each Ev dataset, see X description.
    beam - An array of beams to focus on. For example, if you only want to see pv/pg information on the plot for Beams 3 and 4, then you would set beam = [3,4]. Default is None, and all beams are shown.
    file_index - Default set to None. If changed, this will show the index of the file in an array of all ATL03 file paths so that it is easy to find and focus on interesting cases. Works if you are in a loop of filepaths and you need to know which one is being funky.
    """
    
    # Big plot that we want
    fig = plt.figure(figsize=(10, 6))
    
    # Set the figure title
    if file_index != None:
        fig.suptitle(title_date + ' - N = ' + str(file_index), fontsize=16)
    else:
        fig.suptitle(title_date, fontsize=16)
    
    # Plot the data and the regression lines. If the beam parameter is active,
    # then only for the beams of interest
    for i, c in enumerate(colors):
        if beam != None:
            if c + 1 in beam:
                # scatter
                plt.scatter(X[c],Y[c], s=5, color=cmap2(c))
                # regress
                plt.plot(np.array([0,12]), model([coefs[0], coefs[1+i]], np.array([0,12])), label=f"Beam {int(c+1)}", color=cmap2(c), linestyle='--', zorder=3)
        else:
            #scatter
            plt.scatter(X[c],Y[c], s=5, color=cmap2(c))
            #regress
            plt.plot(np.array([0,12]), model([coefs[0], coefs[1+i]], np.array([0,12])), label=f"Beam {int(c+1)}", color=cmap2(c), linestyle='--', zorder=3)
    # Display the pv/pg estimate
    plt.annotate(r'$\rho_v/\rho_g \approx {:.2f}$'.format(-coefs[0]),
                   xy=(.081,.98),
                   xycoords='axes fraction',
                   ha='right',
                   va='top',
                   fontsize=8,
                   bbox=dict(boxstyle="round,pad=0.3",
                             edgecolor="black",
                             facecolor="white"))
    
    # Do all the boring plot display stuff
    plt.title(f"Ev/Eg Rates", fontsize=8)
    plt.xlabel('Eg (returns/shot)')
    plt.ylabel('Ev (returns/shot)')
    plt.xlim(0,8)
    plt.ylim(0,8)
    plt.legend(loc='best')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust the layout to make room for the suptitle
    plt.show()

def parallel_model(params, x):
    # print(x)
    common_slope, *parallel = params

    # Get all columns starting with 'Beam'
    beam_columns = [col for col in x.columns if col.startswith('Beam')]
    return common_slope*x['Eg'] + np.dot(x[beam_columns], parallel)

def parallel_residuals(params, x, y, model = parallel_model):
    model_output = model(params, x)
    # print(y.T.values[0])
    return (y.T.values[0] - model_output)/np.sqrt(1 + params[0]**2)

def parallel_odr(dataset, maxes, init = -1, lb = -100, ub = -1/100, model = parallel_model, res = parallel_residuals, loss='arctan', f_scale=.1):
    """
    Performs the parallel orthogonal distance regression on the given dataset.
    
    dataset - Pandas Dataframe with columns Eg, Ev, and Beam _ for each beam with data.
    maxes - Array that holds the initial y_intercept guess for each beam. If only Beams 5 and 6 made it, then there are only two values in this array.
    init - Initial slope guess
    lb - Lower bound constraint for slope
    ub - Upper bound constraint for slope
    model - Model to estimate Ev and Eg.
    res - Residuals to put into least_squares function
    loss - Loss function in regression
    f_scale - f_scale parameter for least_squares, affects how much it cares about outliers.
    """
   
    # cats is the number of groundtracks that have data that we could read
    cats = dataset.shape[1]-2
    
    # a is the lower bound of the parameters, [slope, intercept_for_first_dataset, etc.]
    # b is the upper bound, same setup.
    # We then put it together into a bounds variable that we can use in least_squares()
    a = [lb] + [0]*cats
    b = [ub] + [16]*cats
    bounds = (a,b)
    
    # Initial guess [slope, y_intercept_first_dataset, y_intercept_second_dataset, etc.]
    initial_params = [init] + maxes
    # print(initial_params)
    
    # Just like in machine learning, we drop Y from the data to be our dependent variable
    # and we keep everything else, our features, in X.
    X = dataset.drop(columns=['Ev'])
    Y = dataset[['Ev']]
    
    # We call least_squares to do the heavy lifting for us.
    params = least_squares(res, x0=initial_params, args=(X, Y, model), loss = loss, f_scale=f_scale, bounds = bounds, ftol = 1e-15, xtol=1e-15, gtol=1e-15).x
    
    # Return the resulting coefficients
    return params

def pvpg_parallel(atl03path, atl08path,f_scale = .1, loss = 'arctan', init = -1, lb = -100, ub = -1/100, file_index = None, model = parallel_model, res = parallel_residuals, odr = parallel_odr, zeros=None, beam = None, y_init = np.max, graph_detail = 0, canopy_frac = None, terrain_frac = None, keep_flagged=None):
    """
    Parallel regression of all tracks on a given overpass.

    atl03path - Path/to/ATL03/file
    atl08path - Path/to/matching/ATL08/file
    f_scale - Parameter in least_squares() function when loss is nonlinear, indiciating the value of the soft margin between inlier and outlier residuals.
    loss - string for loss parameter in least_squares().
    init - initial slope guess for the parallel slope parameter
    lb - Lower bound of allowed value for the slope of the regression, default -100
    ub - Upper bound of allowed value for the slope of the regression, default -1/100
    file_index - Index of file if cycling through an array of filenames, displayed in figure titles for a given file. Allows us to easily pick out strange cases for investigation.
    model - model function to be used in least squares. Default is the parallel model function
    res - Default holds the ODR residuals function to be used in least_squares(). Can hold adjusted residual functions as well.
    odr - function that performs the orthogonal regression. Replace with great care if you do.
    zeros - Default is None. If changed, this will keep all the canopy height = 0 and Ev = 0 outliers in the data.
    beam - Default is None. Put in input in the form of an array of integers. For example, if you only want to display pv/pg on the plot for Beams 3 and 4, the input is [3,4]
    y_init - This is the function used to initialize the guess for the y intercept. Default is simply the maximum value, as this is expected to correspond with the data point closest to the y-intercept.
    graph_detail - Default is 0. If set to 1, will show a single pv/pg plot for all chosen, available beams. If set to 2, will also show each available groundtrack.
    canopy_frac - Default is None. If changed, this will say in the title of the groundtrack what percentage of the data has canopy photon data. Low canopy fraction could indicate poor quality data. This is only displayed if Detail = 2.
    keep_flagged - Default is None. If changed, we keep the tracks that were thrown out for having segments with zero photon returns.
    """
    
    # This will hold all of the data in one place:
    # [[Eg, Ev, Beam 1],...[Eg,Ev,Beam 1],[Eg,Ev,Beam 2],...,[Eg,Ev,Beam6],[Eg,Ev,Beam 6]]
    # This will be made into a dataframe later.
    meanEgstrong = []
    meanEgweak = []
    meanEvstrong = []
    meanEvweak = []

    msw_flag = []
    night_flag = []
    asr = []
    
    dataset = []
    
    # Holds all of the X data to plot later.
    plotX = []
    
    # Holds all of the Y data to plot later.
    plotY = []
    
    # Holds all of the ATL03 objects to plot groundtracks later
    atl03s = []
    
    # Holds the indices of the beams that successfully read
    I = []
    
    # Check the satellite orientation so we know which beams are strong and weak.
    # Listed from Beam 1 to Beam 6 in the tracks array
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
        return 0, 0, 0, 0
    tracks = [strong[0], weak[0], strong[1], weak[1], strong[2], weak[2]]
    
    # The only purpose of this is to keep the data organised later.
    beam_names = [f"Beam {i}" for i in range(1,7)]
        
    # Very quick quality check; if any of the segments have zero return photons at all,
    # the file is just skipped on assumptions that the data quality isn't good
    if keep_flagged == None:
        for gt in tracks:
            try:
                if 0 in A[gt]['geolocation']['ph_index_beg']:
                    print('File ' + str(file_index) + ' has been skipped because some segments contain zero photon returns.')
                    A.close()
                    return 0, 0, 0, 0
                # This block will be executed if 0 is found in the list
            except (KeyError, FileNotFoundError):
            # Handle the exception (e.g., print a message or log the error)
                continue

    A.close()

    #Keep indices of colors to plot regression lines later:
    colors = []
    
    # Extracting date and time from the filename
    title_date = parse_filename_datetime(atl03path)
    
    # Holds the maximum of the successfully read Ev values to use as y-intercept
    # guesses in the regression
    maxes = []

    B = h5py.File(atl08path, 'r')
    
    # If the user wants to know the fraction of segments that have canopy photons,
    # then we need an array to save it
    if (canopy_frac != None) & (terrain_frac != None):
        canopy_frac = []
        terrain_frac = []
    elif canopy_frac != None:
        canopy_frac = []
    elif terrain_frac != None:
        terrain_frac = []
    
    # Now that we have assurances that the data is good quality,
    # we loop through the ground tracks
    for i, gt in enumerate(tracks):
        
        # If the object fails to be created, we put worthless information into
        # plotX, plotY, and canopy_frac to save us looping effort later
        try:
            atl03 = ATL03(atl03path, atl08path, gt)
        except (KeyError, ValueError, OSError) as e:
            plotX.append([])
            plotY.append([])
            if canopy_frac != None:
                canopy_frac.append(-1)
            if terrain_frac != None:
                terrain_frac.append(-1)
            continue
            
        # The user specifies whether or not they want outliers to be present
        # in the data, generally data points with zero canopy height or canopy photon returns
        if zeros == None:
            atl08 = ATL08(atl08path, gt)
        else:
            atl08 = ATL08_with_zeros(atl08path, gt)
            
        # Retrieve the canopy fraction (fraction of segments that contain any
        # canopy photons) if the user wants it.
        if canopy_frac != None:
            canopy_frac.append(np.array(list(B[gt]['land_segments']['canopy']['subset_can_flag'])).flatten().mean())
        if terrain_frac != None:
            terrain_frac.append(np.array(list(B[gt]['land_segments']['terrain']['subset_te_flag'])).flatten().mean())

        msw_flag = np.concatenate((msw_flag,B[gt]['land_segments']['msw_flag']))
        night_flag = np.concatenate((night_flag,B[gt]['land_segments']['night_flag']))
        asr = np.concatenate((asr,B[gt]['land_segments']['asr']))
        
        
        # X and Y are data for the regression
        X = atl08.df.Eg
        Y = atl08.df.Ev
        
        if i % 2 == 0:
            meanEgstrong.append(np.mean(X))
            meanEvstrong.append(np.mean(Y))
        else:
            meanEgweak.append(np.mean(X))
            meanEvweak.append(np.mean(Y))
        
        # Save it for plotting after the loop goes through all the groundtracks
        plotX.append(X)
        plotY.append(Y)
        
        # Save the ATL03 object
        atl03s.append(atl03)
        
        # Save each individual data point from the ground track along with the Beam it belongs to.
        for x, y in zip(X,Y):
            dataset.append([x, y, beam_names[i]])
            
        if len(Y) == 0:
            print(f'Beam {i + 1} in file {file_index} has been skipped because of no data.')
            continue
        
        # We append the colour we need for the plotting later.
        # Useful when the function is run many times to have many plots
        # and we want the colours to be consistent
        colors.append(i)
        
        # Save the initial y_intercept guess
        maxes.append(y_init(Y))

    # Create DataFrame
    df = pd.DataFrame(dataset, columns=['Eg', 'Ev', 'gt'])

    # Dummy encode the categorical variable
    df_encoded = pd.get_dummies(df, columns=['gt'], prefix='', prefix_sep='')

    if df_encoded.shape[0] == 0:
        print(f'No beams have data in file {file_index}, cannot regress.')
        return 0, 0
    # Retrieve optimal coefficients [slope, y_intercept_dataset_1, y_intercept_dataset_2, etc.]
    coefs = odr(df_encoded, maxes = maxes, init = init, lb=lb, ub=ub, model = model, res = res, loss=loss, f_scale=f_scale)
    
    if len(colors) == 0:
        graph_detail = 0
        
    if graph_detail == 3:
        plot_parallel(atl03s = atl03s,
                      coefs = coefs,
                      colors = colors,
                      title_date = title_date,
                      X = plotX,
                      Y = plotY,
                      beam = beam,
                      canopy_frac = canopy_frac,
                      terrain_frac = terrain_frac,
                      file_index = file_index,
                      three = True)

    # Activate this if you want the whole shebang
    elif graph_detail == 2:
        plot_parallel(atl03s = atl03s,
                      coefs = coefs,
                      colors = colors,
                      title_date = title_date,
                      X = plotX,
                      Y = plotY,
                      beam = beam,
                      canopy_frac = canopy_frac,
                      terrain_frac = terrain_frac,
                      file_index = file_index)
    
    # Activate this if you don't want the groundtracks, just the plot
    elif graph_detail == 1:
        plot_graph(coefs = coefs,
                   colors = colors,
                   title_date = title_date,
                   X = plotX,
                   Y = plotY,
                   beam = beam,
                   file_index = file_index)
    # Don't activate either of them if you don't want a plot
    
    means = [meanEgstrong, meanEgweak, meanEvstrong, meanEvweak]
    
    #Return the coefficients
    return coefs, means, np.mean(msw_flag), np.mean(night_flag), np.mean(asr)
