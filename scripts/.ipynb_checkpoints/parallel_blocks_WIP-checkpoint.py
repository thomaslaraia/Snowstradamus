from scripts.parallel_blocks import *

def plot_parallel(atl03s, coefs, colors, title_date, X, Y, xx, yy, beam = None, canopy_frac = None, terrain_frac = None, file_index=None, three=None):
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
            axes[c].set_title(f"{beam_names[c]} - TF = {round(terrain_frac[c],2)}, CF = {round(canopy_frac[c],2)}")
            plot(atl03.df, axes[c])
        
        elif canopy_frac != None:
            axes[c].set_title(f"{beam_names[c]} - CF = {round(canopy_frac[c],2)}")
            plot(atl03.df, axes[c])
        
        elif terrain_frac != None:
            axes[c].set_title(f"{beam_names[c]} - TF = {round(terrain_frac[c],2)}")
            plot(atl03.df, axes[c])
        
        else:
            axes[c].set_title(f"{beam_names[c]}")
            plot(atl03.df, axes[c])
        
        # If there's a focus on certain beams, we run this if statement to
        # check if the current beam is in the list of beams the user wants.
        # Then we throw the data onto the scatterplot with the color of choice
        # along with a regression line of the same color
        if three == None:
        
            if beam != None:
                if c + 1 in beam:
                    # scatter
                    ax7.scatter(X[c],Y[c], s=5, color=cmap3(2*c+1), marker='o')
                    ax7.scatter(xx[c], yy[c], s=5, color=cmap3(2*c), marker='o')
                    # regress
                    ax7.plot(np.array([0,12]), model([coefs[0], coefs[1+i]], np.array([0,12])), label=f"Beam {int(c+1)}", color=cmap3(2*c), linestyle='--', zorder=3)
            else:
                #scatter
                ax7.scatter(X[c],Y[c], s=5, color=cmap3(2*c+1), marker='o')
                ax7.scatter(xx[c], yy[c], s=5, color=cmap3(2*c), marker='o')
                #regress
                ax7.plot(np.array([0,12]), model([coefs[0], coefs[1+i]], np.array([0,12])), label=f"Beam {int(c+1)}", color=cmap3(2*c), linestyle='--', zorder=3)
    
    
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
    #plt.savefig('./images/groundtracks.svg')
    plt.show()
    return

def parallel_odr(dataset, intercepts, maxes, init = -1, lb = -100, ub = -1/100, model = parallel_model, res = parallel_residuals, loss='arctan', f_scale=.1, outlier_removal = False, method='normal', w=[1.0,0.25]):
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
    cats = dataset.shape[1]-5
    
    # a is the lower bound of the parameters, [slope, intercept_for_first_dataset, etc.]
    # b is the upper bound, same setup.
    # We then put it together into a bounds variable that we can use in least_squares()
    a = [lb] + [0]*cats
    b = [ub] + maxes
    bounds = (a,b)
    
    # Initial guess [slope, y_intercept_first_dataset, y_intercept_second_dataset, etc.]
    initial_params = [init] + intercepts
    # print(initial_params)

    #################################


    # if outlier_removal != False:

    beam_columns = [col for col in dataset.columns if 'Beam' in col]

    filtered_data = []
    full_data = []

    data_quant = 0

    for beam in beam_columns:
        # Select rows where the current beam is True
        beam_data = dataset[dataset[beam] == True][['Eg', 'Ev', 'layer_flag', 'msw_flag', 'cloud_flag_atm'] + beam_columns].copy()
        #print(len(beam_data))

        if outlier_removal == False:
            beam_data['Outlier'] = 1
            full_data.append(beam_data[['Eg', 'Ev', 'layer_flag', 'msw_flag', 'cloud_flag_atm', 'Outlier'] + beam_columns])
            continue
        
        # # Detect outliers based on Z-score for 'Eg' and 'Ev'
        # beam_data['Eg_z'] = zscore(beam_data['Eg'])
        # beam_data['Ev_z'] = zscore(beam_data['Ev'])
        # if len(beam_data) >= 4:
        #     # Filter out rows with Z-scores above a threshold (outliers)
        #     beam_filtered = beam_data[(beam_data['Eg_z'].abs() <= outlier_removal) & (beam_data['Ev_z'].abs() <= outlier_removal)]
        # else:
        #     beam_filtered = beam_data
        
        # mean = beam_data[['Eg', 'Ev']].median().values
        # cov = np.cov(beam_data[['Eg', 'Ev']].values, rowvar=False)
        # inv_cov = np.linalg.inv(cov)
        # # Compute Mahalanobis distance
        # diff = beam_data[['Eg', 'Ev']].values - mean
        # mahalanobis_dist = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
        # # Set a threshold based on Chi-squared distribution
        # threshold = chi2.ppf(0.6, df=2)  # 99.7% confidence for 2 dimensions
        # beam_data['Mahalanobis_dist'] = mahalanobis_dist
        # if len(beam_data) >= 4:
        #     beam_filtered = beam_data[beam_data['Mahalanobis_dist'] <= np.sqrt(threshold)]
        # else:
        #     beam_filtered = beam_data

        if len(beam_data) >= 2:

            if outlier_removal < 1:
                # Fit an EllipticEnvelope model
                envelope = EllipticEnvelope(contamination=outlier_removal, random_state=42)  # Adjust contamination as needed
                envelope.fit(beam_data[['Eg', 'Ev']])
                # Predict inliers (1) and outliers (-1)
                beam_data['Outlier'] = envelope.predict(beam_data[['Eg', 'Ev']])
                beam_filtered = beam_data[beam_data['Outlier'] == 1]

            elif outlier_removal >= 2:
            #     # Fit an Local Outlier Factor model
            #     lof = LocalOutlierFactor(n_neighbors=round(outlier_removal), contamination='auto')
            #     beam_data['Outlier'] = lof.fit_predict(beam_data[['Eg', 'Ev']])
            #     beam_filtered = beam_data[beam_data['Outlier'] == 1]

            # else:

                # Initialize an array to track if a point is ever flagged as an outlier
                outlier_flags = np.zeros(len(beam_data), dtype=bool)
                
                for n in range(10, 16):
                    lof = LocalOutlierFactor(n_neighbors=n, contamination='auto')
                    preds = lof.fit_predict(beam_data[['Eg', 'Ev']])
                    outlier_flags |= (preds == -1)  # Mark as outlier if flagged at this n_neighbors
                
                beam_data['Outlier'] = np.where(outlier_flags, -1, 1)
                beam_filtered = beam_data[beam_data['Outlier'] == 1]
        else:
            beam_filtered = beam_data

        # # Fit DBSCAN
        # dbscan = DBSCAN(eps=0.5, min_samples=4)  # Tune eps and min_samples
        # beam_data['Cluster'] = dbscan.fit_predict(beam_data[['Eg', 'Ev']])
        # # Remove outliers (Cluster -1)
        # beam_filtered = beam_data[beam_data['Cluster'] != -1]

        filtered_data.append(beam_filtered[['Eg', 'Ev', 'layer_flag', 'msw_flag', 'cloud_flag_atm'] + beam_columns])  # Keep only Eg, Ev, and beam columns
        full_data.append(beam_data[['Eg', 'Ev', 'layer_flag', 'msw_flag', 'cloud_flag_atm', 'Outlier'] + beam_columns])
        # print(full_data)
        data_quant = max(data_quant, len(beam_data))


    full_dataset = pd.concat(full_data).reset_index(drop=True)

    if outlier_removal != False:
        # Combine filtered data for all beams, maintaining the original beam columns with True/False values
        filtered_dataset = pd.concat(filtered_data).reset_index(drop=True)
        dataset = filtered_dataset.copy()

    #################################

    # Just like in machine learning, we drop Y from the data to be our dependent variable
    # and we keep everything else, our features, in X.
    X = dataset.drop(columns=['Ev', 'layer_flag', 'msw_flag', 'cloud_flag_atm'])
    Y = dataset[['Ev']]

    # print(initial_params)
    # print(X)
    # print(Y)

    if method == 'bimodal':
    
        #params = minimize(res, x0=initial_params, args=(X, Y, model))
        #np.set_printoptions(threshold=np.inf)
        #print(X)
        #print(Y)
        #if np.any(np.isnan(X)) or np.any(np.isnan(Y)):
        #     print(f"NaNs detected in iteration {k}")
        #if np.any(np.isinf(X)) or np.any(np.isinf(Y)):
        #    print(f"Infs detected in iteration {k}")
        #np.set_printoptions(threshold=1000)
        
        params = least_squares(parallel_residuals, x0=initial_params, args=(X, Y, model, False, w), loss = loss, bounds = bounds)#, verbose=2)
        params = least_squares(parallel_residuals, x0=params.x, args=(X, Y, model, True, w), loss = loss, bounds = bounds)
        
        # params = differential_evolution(parallel_residuals, bounds = list(zip(bounds[0],bounds[1])), args=(X, Y, model, True))
    
    elif loss == 'linear':
        params = least_squares(parallel_residuals, x0=initial_params, args=(X, Y, model, False, w), loss = loss, bounds = bounds)

    
    # We call least_squares to do the heavy lifting for us.
    else:
        params = least_squares(parallel_residuals, x0=initial_params, args=(X, Y, model, False, w), loss = loss, f_scale=f_scale, bounds = bounds, ftol=1e-15, xtol=1e-15, gtol=1e-15)

    # data quality
    lf = dataset.layer_flag.mean()
    msw = dataset.msw_flag.mean()
    # print(params.x[1:])
    bn = [
        int(re.search(r'Beam (\d+)', col).group(1)) 
        for col in dataset.columns if re.search(r'Beam \d+', col)
    ]
    strong_pv_mean = 0
    weak_pv_mean = 0
    strong_pv_max = 0
    strong_pg_max = 0
    for i, num in enumerate(bn):
        if num%2 == 1:
            strong_pv_mean += params.x[i+1]
            strong_pv_max = max(strong_pv_max, params.x[i+1])
            strong_pg_max = max(strong_pg_max, -params.x[i+1]/params.x[0])
        else:
            weak_pv_mean += params.x[i+1]
    if weak_pv_mean != 0:
        pv_ratio = strong_pv_mean/weak_pv_mean
    else:
        pv_ratio = 0
    # print(pv_ratio)
    # print(dataset.columns)
    #print(data_quant)

    # PLACEHOLDER
    if ((lf <= 0.7)&(msw < 0.2))&(pv_ratio>=1.3)&(params.x[0]<=7.5)&(strong_pv_max <= 16)&(strong_pg_max <= 16):
        data_quality = 0
    else:
        data_quality = 1
    
    # Return the resulting coefficients
    return params.x, dataset, full_dataset, data_quality

# This corresponds to graph_detail = 1
def plot_graph(coefs, colors, title_date, X, Y, xx, yy, coords, beam = None, file_index=None, data_quality = 0):
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
    title_color = ['black', 'red']
    
    # Big plot that we want
    fig = plt.figure(figsize=(10, 6))
    
    # Set the figure title
    if file_index != None:
        fig.suptitle(title_date + ' - N = ' + str(file_index) + ' - ' + str(coords), fontsize=16, color = title_color[data_quality])
    else:
        fig.suptitle(title_date + ' - ' + str(coords), fontsize=16, color = title_color[data_quality])
    
    # Plot the data and the regression lines. If the beam parameter is active,
    # then only for the beams of interest
    for i, c in enumerate(colors):
        if beam != None:
            if c + 1 in beam:
                # scatter
                plt.scatter(X[i],Y[i], s=5, color=cmap3(2*c+1), marker='o')
                plt.scatter(xx[i], yy[i], s=5, color=cmap3(2*c), marker='o')
                # regress
                plt.plot(np.array([0,12]), model([coefs[0], coefs[1+i]], np.array([0,12])), label=f"Beam {int(c+1)}", color=cmap3(2*c), linestyle='--', zorder=3)
        else:
            #scatter
            plt.scatter(X[i],Y[i], s=5, color=cmap3(2*c+1), marker='o')
            plt.scatter(xx[c], yy[c], s=5, color=cmap3(2*c), marker='o')
            #regress
            plt.plot(np.array([0,12]), model([coefs[0], coefs[1+i]], np.array([0,12])), label=f"Beam {int(c+1)}", color=cmap3(2*c), linestyle='--', zorder=3)
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
    return

def pvpg_parallel(dirpath, atl03path, atl08path, coords, width=4, height=4, f_scale = .1, loss = 'linear', init = -.6,\
                  lb = -100, ub = -1/100,file_index = None, model = parallel_model, res = parallel_residuals,\
                  odr = parallel_odr, zeros=None,beam_focus = None, y_init = np.max, graph_detail = 0, keep_flagged=True,\
                  opsys='bad', altitude=None,alt_thresh=80, threshold = 1, small_box = 1, rebinned = 0, res_field='alongtrack',
                  outlier_removal=False, method='normal', landcover = 'forest', trim_atmospheric=0, w=[1.0,0.25], sat_flag = 1,
                  show_me_the_good_ones = 0, DW=0):
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
    keep_flagged - Default is True. If None, we throw out tracks that have segments with zero photon returns.
    """
    
    # print(dirpath, file_index)

    polygon = make_box(coords, width,height)
    min_lon, min_lat, max_lon, max_lat = polygon.total_bounds
    # print(min_lon, max_lon)
    # print(min_lat, max_lat)

    # Convert small_box from kilometers to degrees
    km_per_degree_lat = 111  # Kilometers per degree of latitude
    km_per_degree_lon = 111 * np.cos(np.radians(coords[1]))  # Kilometers per degree of longitude at the given latitude

    # Calculate the increment in degrees for the small box size
    small_box_lat = small_box / km_per_degree_lat
    small_box_lon = small_box / km_per_degree_lon

    # Generate the latitude and longitude ranges using the converted small box sizes
    lats = np.arange(min_lat + small_box_lat / 2,
                     max_lat + small_box_lat / 2,
                     small_box_lat)
    # if follow_beams == 0:
    lons = np.arange(min_lon + small_box_lon / 2,
                     max_lon + small_box_lon / 2,
                     small_box_lon)

    LATS = []
    LONS = []

    ALL_LATS = []
    ALL_LONS = []
    
    foldername = dirpath.split('/')[-2]
    # print(lats, lons)
    
    # This will hold all of the data in one place:
    # [[Eg, Ev, Beam 1],...[Eg,Ev,Beam 1],[Eg,Ev,Beam 2],...,[Eg,Ev,Beam6],[Eg,Ev,Beam 6]]
    # This will be made into a dataframe later.
    Eg = [[] for _ in range(len(lats)*len(lons))]
    Ev = [[] for _ in range(len(lats)*len(lons))]
    #EvEg = [[] for _ in range(len(lats)*len(lons))]
    trad_cc = [[] for _ in range(len(lats)*len(lons))]
    beam_str = [[] for _ in range(len(lats)*len(lons))]
    beam = [[] for _ in range(len(lats)*len(lons))]
    data_quantity = [[] for _ in range(len(lats)*len(lons))]

    # Define base variable names
    variable_names = [
        'msw_flag', 'night_flag', 'asr', 'canopy_openness', 
        'snr', 'segment_cover', 'segment_landcover', 
        'h_te_interp', 'h_te_std', 'terrain_slope', 'longitude', 'latitude',
        'cloud_flag_atm', 'layer_flag'
    ]
    if DW != 0:
        variable_names.append('DW')
    # removed 'dem_h', 'h_te_best_fit'

    # Define dictionaries to store the arrays for both strong and weak pairs
    var_dict = {}

    # Initialize empty lists in both strong and weak dictionaries
    for var in variable_names:
        var_dict[var] = [[] for _ in range(len(lats)*len(lons))]

    #EvEg = [-1 for _ in range(len(lats)*len(lons))]
    
    dataset = [[] for _ in range(len(lats)*len(lons))]
    
    # Holds all of the X data to plot later.
    plotX = [[] for _ in range(len(lats)*len(lons))]
    
    # Holds all of the Y data to plot later.
    plotY = [[] for _ in range(len(lats)*len(lons))]
    
    # Holds all of the ATL03 objects to plot groundtracks later
    atl03s = [[] for _ in range(len(lats)*len(lons))]

    # To find the starting slope guess
    slope_init = [[] for _ in range(len(lats)*len(lons))]
    slope_weight = [[] for _ in range(len(lats)*len(lons))]
    
#     for i in range(len(lats)*len(lons)):
#         dataset.append([])
#         plotX.append([])
#         plotY.append([])
#         msw_flag.append([])
#         night_flag.append([])
#         asr.append([])
#         atl03s.append([])
    
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
        return 0
    tracks = [strong[0], weak[0], strong[1], weak[1], strong[2], weak[2]]
    #print(tracks)
    
    # The only purpose of this is to keep the data organised later.
    beam_names = [f"Beam {i}" for i in range(1,7)]
    
    # Very quick quality check; if any of the segments have zero return photons at all,
    # the file is just skipped on assumptions that the data quality isn't good
#     if keep_flagged == None:
#         for gt in tracks:
#             try:
#                 if 0 in A[gt]['geolocation']['ph_index_beg']:
#                     print('File ' + str(file_index) + ' has been skipped because some segments contain zero photon returns.')
#                     A.close()
#                     return 0, 0, 0, 0, 0, 0
#                 # This block will be executed if 0 is found in the list
#             except (KeyError, FileNotFoundError):
#             # Handle the exception (e.g., print a message or log the error)
#                 continue

    A.close()

    #Keep indices of colors to plot regression lines later:
    colors = [[] for _ in range(len(lats)*len(lons))]
    
    # Extracting date and time from the filename
    mid_date = parse_filename_datetime(atl03path)
    title_date = datetime_to_title(mid_date)
    table_date = datetime_to_date(mid_date)
    
    # Holds the maximum of the successfully read Ev values to use as y-intercept
    # guesses in the regression
    intercepts = [[] for _ in range(len(lats)*len(lons))]
    maxes = [[] for _ in range(len(lats)*len(lons))]

    K = 0
    
    # Now that we have assurances that the data is good quality,
    # we loop through the ground tracks
    for i, gt in enumerate(tracks):
        
        # If the object fails to be created, we put worthless information into
        # plotX, plotY, and canopy_frac to save us looping effort later
        try:
            # print(atl03path, gt, atl08path)
            atl03 = get_atl03_struct(atl03path, gt, atl08path)
        except (KeyError, ValueError, OSError, IndexError) as e:
            # for k in range(len(lats)*len(lons)):
            #     plotX[k].append([])
            #     plotY[k].append([])
            # # msw_flag = np.concatenate((msw_flag,-1))
            # # night_flag = np.concatenate((night_flag,-1))
            # # asr = np.concatenate((asr,-1))
            #     Eg[k].append([-1])
            #     Ev[k].append([-1])
            #     data_quantity[k].append([-1])
            #     #EvEg[k].append([-1])
            #     trad_cc[k].append([-1])
            #     for var in variable_names:
            #         var_dict[var][k].append([-1])
            #     beam_str[k].append([-1])
            #     beam[k].append([-1])
            print(f"Failed to open ATL03 file for {foldername} file {file_index}'s beam {i+1}.")
            continue
            
        try:
            atl08 = get_atl08_struct(atl08path, gt, atl03)
        except (KeyError, ValueError, OSError) as e:
            # for k in range(len(lats)*len(lons)):
            #     plotX[k].append([])
            #     plotY[k].append([])
                
            #     Eg[k].append([-1])
            #     Ev[k].append([-1])
            #     data_quantity[k].append([-1])
            #     #EvEg[k].append([-1])
            #     trad_cc[k].append([-1])
            #     for var in variable_names:
            #         var_dict[var][k].append([-1])
            #     beam_str[k].append([-1])
            #     beam[k].append([-1])
            print(f"Failed to open ATL08 file for {foldername} file {file_index}'s beam {i+1}.")
            continue
        
        atl03.df = atl03.df[(atl03.df['lon_ph'] >= min_lon) & (atl03.df['lon_ph'] <= max_lon) &\
                                (atl03.df['lat_ph'] >= min_lat) & (atl03.df['lat_ph'] <= max_lat)]

        atl08.df = atl08.df[(atl08.df['longitude'] >= min_lon) & (atl08.df['longitude'] <= max_lon) &\
                                (atl08.df['latitude'] >= min_lat) & (atl08.df['latitude'] <= max_lat)]

        # print(str(list(atl08.df.columns)))
        
        if rebinned != 0:
            if atl08.df.shape[0] == 0:
                # for k in range(len(lats)*len(lons)):
                #     plotX[k].append([])
                #     plotY[k].append([])
                    
                #     Eg[k].append([-1])
                #     Ev[k].append([-1])
                #     data_quantity[k].append([-1])
                #     #EvEg[k].append([-1])
                #     trad_cc[k].append([-1])
                #     for var in variable_names:
                #         var_dict[var][k].append([-1])
                #     beam_str[k].append([-1])
                #     beam[k].append([-1])
                print(f"Nothing in rebinned section for {foldername} file {file_index}'s beam {i+1}.")
                continue
            atl08.df = rebin_atl08(atl03, atl08, gt, rebinned, res_field)

        # print(str(list(atl08.df.columns)))
        
        atl08.df = atl08.df[(atl08.df.photon_rate_can_nr < 10) & (atl08.df.photon_rate_te < 10)]# & (atl08.df.h_canopy < 100)]
        
        # print(len(atl08.df))
        # NEW BIT FOR LAND COVER CLASSIFICATION ##############################################################################
        # print(atl08.df['landcover'])
        if DW != 0:
            filepath = find_dynamicworld_file(foldername)
            da = rioxarray.open_rasterio(filepath, masked=True).rio.reproject("EPSG:4326")
            atl08.df['DW'] = da.sel(band=1).interp(
                y=("points", atl08.df.latitude.values),
                x=("points", atl08.df.longitude.values),
                method="nearest"
            ).values
            atl08.df = atl08.df[~atl08.df['DW'].isin([0])]
        
        if landcover == 'forest':
            atl08.df = atl08.df[atl08.df['segment_landcover'].isin([111,112,113,114,115,116,121,122,123,124,125,126])]
        elif landcover == 'all':
            atl08.df = atl08.df[~atl08.df['segment_landcover'].isin([60,40,100,50,70,80,200,0])]
        # print(atl08.df)
        # print(atl08.df.h_te_interp)
        if altitude != None:
            atl08.df = atl08.df[abs(atl08.df['h_te_interp'] - altitude) <= alt_thresh]
        
        if trim_atmospheric != 0:
            atl08.df = atl08.df[(atl08.df['layer_flag'] < 1)|(atl08.df['msw_flag']<1)]
        # print(len(atl08.df))
        if sat_flag != 0:
            atl08.df = atl08.df[atl08.df['sat_flag'] == 0]

        k = K
        if i % 2 == 0:
            LATS = []
            LONS = []
            lats = np.arange(min_lat + small_box_lat / 2,
                 max_lat + small_box_lat / 2,
                 small_box_lat)
        if i % 2 == 1:
            if len(LONS) == 0:
                continue
            lats, lons = LATS, LONS
            
        for n, lat in enumerate(lats):

            if i % 2 == 0:
                polygon = make_box((coords[1],lat), small_box/2, small_box/2)
                sub_min_lon, sub_min_lat, sub_max_lon, sub_max_lat = polygon.total_bounds
                
                atl03_temp = atl03.df[(atl03.df['lat_ph'] >= sub_min_lat) & (atl03.df['lat_ph'] <= sub_max_lat)].copy()
                atl08_temp = atl08.df[(atl08.df['latitude'] >= sub_min_lat) & (atl08.df['latitude'] <= sub_max_lat)].copy()
    
                if len(atl08_temp) != 0:
                    lon = atl08_temp.longitude.mean()
                else:
                    continue

            if i % 2 == 1:
                lon = lons[n]

            # print(lons,lon)
            polygon = make_box((lon,lat), small_box/2,small_box/2)
            sub_min_lon, sub_min_lat, sub_max_lon, sub_max_lat = polygon.total_bounds
            # print(atl08.df)
            atl03_temp = atl03.df[(atl03.df['lon_ph'] >= sub_min_lon) & (atl03.df['lon_ph'] <= sub_max_lon) &\
                                    (atl03.df['lat_ph'] >= sub_min_lat) & (atl03.df['lat_ph'] <= sub_max_lat)].copy()
            atl08_temp = atl08.df[(atl08.df['longitude'] >= sub_min_lon) & (atl08.df['longitude'] <= sub_max_lon) &\
                                    (atl08.df['latitude'] >= sub_min_lat) & (atl08.df['latitude'] <= sub_max_lat)].copy()
            
            if atl08_temp.shape[0] < threshold:
                # print(f'Beam {i + 1}, box {k} in {foldername} file {file_index} has no data.')
                # plotX[k].append([])
                # plotY[k].append([])
                
                # Eg[k].append([-1])
                # Ev[k].append([-1])
                # data_quantity[k].append([-1])
                # #EvEg[k].append([-1])
                # trad_cc[k].append([-1])
                # for var in variable_names:
                #     var_dict[var][k].append([-1])
                # beam_str[k].append([-1])
                # beam[k].append([-1])
                # k += 1
                if i % 2 == 1:
                    k += 1
                continue
            # Retrieve the canopy fraction (fraction of segments that contain any
            # canopy photons) if the user wants it.
    
            # X and Y are data for the regression
            X = atl08_temp.photon_rate_te
            Y = atl08_temp.photon_rate_can_nr

            if i + 1 == 3:
                X /= 0.85
                Y /= 0.85

            layer_flag = atl08_temp.layer_flag
            msw_flag = atl08_temp.msw_flag
            cloud_flag_atm = atl08_temp.cloud_flag_atm
    
            # Save it for plotting after the loop goes through all the groundtracks
            plotX[k].append(X)
            plotY[k].append(Y)

            if i % 2 == 0:
                LATS.append(lat)
                LONS.append(lon)
            
            atl03s[k].append(atl03)
            colors[k].append(i)

            Eg[k].append(X)
            Ev[k].append(Y)
            data_quantity[k].append([len(X) for x in range(len(X))])
            #EvEg[k].append(Y/X)
            trad_cc[k].append((atl08_temp['n_ca_photons']+atl08_temp['n_toc_photons'])/\
                                     (atl08_temp['n_ca_photons']+atl08_temp['n_toc_photons']+atl08_temp['n_te_photons']))
            for var in variable_names:
                # print(var, atl08_temp[var])
                var_dict[var][k].append(atl08_temp[var])
            
            if i % 2 == 0:
                beam_str[k].append(['strong' for _ in range(len(atl08_temp['n_ca_photons']))])
                    # print(strong_dict[f"{var}_strong"])
            else:
                beam_str[k].append(['weak' for _ in range(len(atl08_temp['n_ca_photons']))])
            beam[k].append([i+1 for _ in range(len(atl08_temp['n_ca_photons']))])
                
            # print(X)
            # print(Y)
            # Save each individual data point from the ground track along with the Beam it belongs to.
            for x, y, lf, mf, cfa in zip(X,Y, layer_flag, msw_flag, cloud_flag_atm):
                dataset[k].append([x, y, beam_names[i], lf, mf, cfa])

            # tweaking starting parameters
            ############################################################

            intercept, slope = starting_intercept(X,Y)
                    
            slope_init[k].append(slope)
            # slope_init[k].append(-.3)
            slope_weight[k].append(len(Y))
            # Save the initial y_intercept guess
            intercepts[k].append(min(intercept,16))
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

    del atl03
    del atl08
    gc.collect()
    
    k = 0
    for lat, lon in zip(ALL_LATS, ALL_LONS):
        if len(dataset[k]) == 0:
            k+=1
            continue
        
        slope_weight[k] /= np.sum([slope_weight[k]])
        slope_init[k] = np.dot(slope_init[k],slope_weight[k])
        
        # Create DataFrame
        df = pd.DataFrame(dataset[k], columns=['Eg', 'Ev', 'gt', 'layer_flag', 'msw_flag', 'cloud_flag_atm'])
        # Dummy encode the categorical variable
        df_encoded = pd.get_dummies(df, columns=['gt'], prefix='', prefix_sep='')
        
        coefs, xy, full_xy, data_quality = odr(df_encoded, intercepts = intercepts[k], maxes = maxes[k], init = slope_init[k],\
                    lb=lb, ub=ub, model = model, res = res, loss=loss, f_scale=f_scale,
                          outlier_removal=outlier_removal, method=method, w=w)

        # Create the array of empty lists
        xx = [[] for _ in range(6)]
        yy = [[] for _ in range(6)]

        beams_in_play = []
        
        # Iterate over each beam column and append the Eg values where the condition is True
        for i in range(1, 7):  # Beam 1 to Beam 6
            if f'Beam {i}' in xy.columns:
                xx[i-1] = xy[xy[f'Beam {i}'] == True]['Eg']
                yy[i-1] = xy[xy[f'Beam {i}'] == True]['Ev']
                beams_in_play.append(i)

        # print(plotX[k])
        # print(xx)

        if show_me_the_good_ones == 0 or data_quality == 0:
        
            if len(colors) == 0:
                graph_detail = 0
                
            if graph_detail == 3:
                plot_parallel(atl03s = atl03s[k],
                              coefs = coefs,
                              colors = colors[k],
                              title_date = title_date,
                              X = plotX[k],
                              Y = plotY[k],
                              xx = xx,
                              yy = yy,
                              beam = beam_focus,
                              file_index = file_index,
                              three = True)
                
            elif graph_detail == 2:
                plot_parallel(atl03s = atl03s[k],
                              coefs = coefs,
                              colors = colors[k],
                              title_date = title_date,
                              X = plotX[k],
                              Y = plotY[k],
                              xx = xx,
                              yy = yy,
                              beam = beam_focus,
                              file_index = file_index)

            # Activate this if you don't want the groundtracks, just the plot
            elif graph_detail == 1:
                plot_graph(coefs = coefs,
                           colors = colors[k],
                           title_date = title_date,
                           X = plotX[k],
                           Y = plotY[k],
                           xx = xx,
                           yy = yy,
                           coords=(lat,lon),
                           beam = beam_focus,
                           file_index = file_index,
                           data_quality = data_quality)

        # print(asr[k])
        # print(meanEgstrong[k])

        # print(non_negative_subset(asr[k]))

        # if len(meanEgstrong) > 0:
        #     EvEg[k] = safe_mean(non_negative_subset(meanEvstrong[k]))/safe_mean(non_negative_subset(meanEgstrong[k]))

        # for entry in Eg[k]:
        #     print(type(
        # print(Eg[k])
        indices_to_insert = [i for i in range(1,7) if i not in beams_in_play]
        for index in indices_to_insert:
            coefs = np.insert(coefs, index, None)
        
        if np.all(np.isnan([coefs[1],coefs[3],coefs[5]])):
            y_strong = np.nan
        else:
            y_strong = np.nanmean([coefs[1],coefs[3],coefs[5]])
            y_strong_max = np.nanmax([coefs[1],coefs[3],coefs[5]])
            
        if np.all(np.isnan([coefs[2],coefs[4],coefs[6]])):
            y_weak = np.nan
        else:
            y_weak = np.nanmean([coefs[2],coefs[4],coefs[6]])
            y_weak_max = np.nanmax([coefs[2],coefs[4],coefs[6]])
            
        if np.any(np.isnan([y_strong, y_weak])):
            pv_ratio_mean = np.nan
            pv_ratio_max = np.nan
        else:
            pv_ratio_mean = y_strong/y_weak
            pv_ratio_max = y_strong_max/y_weak_max
        
        y_intercept_dict = {1: coefs[1], 2: coefs[2], 3: coefs[3], 4: coefs[4], 5: coefs[5], 6: coefs[6]}
        x_intercept_dict = {1: -coefs[1]/coefs[0], 2: -coefs[2]/coefs[0], 3: -coefs[3]/coefs[0], 4: -coefs[4]/coefs[0],
                           5: -coefs[5]/coefs[0], 6: -coefs[6]/coefs[0]}

        # print(x_intercept_dict['strong'])
        # print(non_negative_subset(msw_flag[k]),msw_flag[k])

        # Append the row dynamically
        for j in range(len(non_negative_subset(Eg[k]))):
            row_data = [foldername, table_date, lon, lat, -coefs[0],
                        y_intercept_dict[non_negative_subset(beam[k])[j]], x_intercept_dict[non_negative_subset(beam[k])[j]],
                        non_negative_subset(Eg[k])[j], non_negative_subset(Ev[k])[j],
                        non_negative_subset(data_quantity[k])[j], data_quality, altitude, pv_ratio_mean, pv_ratio_max,
                        non_negative_subset(trad_cc[k])[j], non_negative_subset(beam[k])[j], non_negative_subset(beam_str[k])[j]]
            row_data.append(full_xy['Outlier'].iloc[j])

            # Add the rest of the strong-weak pairs dynamically
            for var in variable_names:  # Start from msw, as meanEg and meanEv are already included
                # var = f"{var}"
                row_data.append(non_negative_subset(var_dict[var][k])[j])
            # Append the row to the rows list
            rows.append(row_data)
        k+=1

    columns_list = ['camera', 'date', 'lon', 'lat', 'pvpg', 'pv', 'pg', 'Eg', 'Ev',
                    'data_quantity', 'data_quality', 'altitude', 'pv_ratio_mean', 'pv_ratio_max', 'trad_cc','beam', 'beam_str',
                    'outlier']
    for var in variable_names:  # Start from msw, as meanEg and meanEv are already included
        columns_list.append(var)
    
    BIG_DF = pd.DataFrame(rows,columns=[columns_list])
    
    BIG_DF.columns = BIG_DF.columns.get_level_values(0)
            
    return BIG_DF
