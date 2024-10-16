from scripts.parallel_phoreal import *

def non_negative_subset(asr_list):
    cleaned_data = []
    
    for item in asr_list:
        # Check if the item is a pandas Series (from your dataframe)
        if isinstance(item, pd.Series):
            # Append the non-negative values from the pandas Series
            cleaned_data.extend(item.values)
        # Check if it's a list with a single value [-1]
        elif isinstance(item, list) and item == [-1]:
            continue  # Skip the [-1] list, as it represents missing data
        # If it's a regular list, append non-negative values
        elif isinstance(item, list) and ('strong' in item or 'weak' in item):
            cleaned_data.extend([x for x in item])
        # elif isinstance(item, list):
        #     cleaned_data.extend([x for x in item])
    
    return np.array(cleaned_data)  # Return as a numpy array

def flatten_structure(structure):
    flat_list = []
    if isinstance(structure, (list, tuple, np.ndarray)):
        for item in structure:
            flat_list.extend(flatten_structure(item))
    else:
        flat_list.append(structure)
    return flat_list

def datetime_to_date(datetime_obj):
    return datetime_obj.strftime('%d/%m/%Y')

def safe_mean(arr):
    if arr.size == 0:  # Check if the array is empty
        return np.nan
    else:
        return np.mean(arr)

def pvpg_parallel(dirpath, atl03path, atl08path, coords, width=5, height=5, f_scale = .1, loss = 'arctan', init = -.6,\
                  lb = -np.inf, ub = 0,file_index = None, model = parallel_model, res = parallel_residuals,\
                  odr = parallel_odr, zeros=None,beam = None, y_init = np.max, graph_detail = 0, keep_flagged=True,\
                  opsys='bad', altitude=None,alt_thresh=80, threshold = 1, small_box = 1):
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

    polygon = make_box(coords, width,height)
    min_lon, min_lat, max_lon, max_lat = polygon.total_bounds

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
    lons = np.arange(min_lon + small_box_lon / 2,
                     max_lon + small_box_lon / 2,
                     small_box_lon)
    
    foldername = dirpath.split('/')[-2]
    # print(lats, lons)
    
    # This will hold all of the data in one place:
    # [[Eg, Ev, Beam 1],...[Eg,Ev,Beam 1],[Eg,Ev,Beam 2],...,[Eg,Ev,Beam6],[Eg,Ev,Beam 6]]
    # This will be made into a dataframe later.
    Eg = [[] for _ in range(len(lats)*len(lons))]
    Ev = [[] for _ in range(len(lats)*len(lons))]
    EvEg = [[] for _ in range(len(lats)*len(lons))]
    trad_cc = [[] for _ in range(len(lats)*len(lons))]
    beam = [[] for _ in range(len(lats)*len(lons))]

    # Define base variable names
    variable_names = [
        'msw_flag', 'night_flag', 'asr', 'canopy_openness', 
        'snr', 'segment_cover', 'segment_landcover', 
        'h_te_interp', 'h_te_std', 'terrain_slope', 'longitude', 'latitude'
    ]
    # removed 'dem_h', 'h_te_best_fit'

    # Define dictionaries to store the arrays for both strong and weak pairs
    var_dict = {}

    # Initialize empty lists in both strong and weak dictionaries
    for var in variable_names:
        var_dict[f"{var}"] = [[] for _ in range(len(lats)*len(lons))]

    #EvEg = [-1 for _ in range(len(lats)*len(lons))]
    
    dataset = [[] for _ in range(len(lats)*len(lons))]
    
    # Holds all of the X data to plot later.
    plotX = [[] for _ in range(len(lats)*len(lons))]
    
    # Holds all of the Y data to plot later.
    plotY = [[] for _ in range(len(lats)*len(lons))]
    
    # Holds all of the ATL03 objects to plot groundtracks later
    atl03s = [[] for _ in range(len(lats)*len(lons))]

    # To find the starting slope guess
    # slope_init = [[] for _ in range(len(lats)*len(lons))]
    # slope_weight = [[] for _ in range(len(lats)*len(lons))]
    
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
        return 0, 0, 0, 0, 0, 0
    tracks = [strong[0], weak[0], strong[1], weak[1], strong[2], weak[2]]
    
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
    # intercepts = [[] for _ in range(len(lats)*len(lons))]
    # maxes = [[] for _ in range(len(lats)*len(lons))]
    
    # Now that we have assurances that the data is good quality,
    # we loop through the ground tracks
    for i, gt in enumerate(tracks):
        
        # If the object fails to be created, we put worthless information into
        # plotX, plotY, and canopy_frac to save us looping effort later
        try:
            # print(atl03path, gt, atl08path)
            atl03 = get_atl03_struct(atl03path, gt, atl08path)
        except (KeyError, ValueError, OSError, IndexError) as e:
            for k in range(len(lats)*len(lons)):
                plotX[k].append([])
                plotY[k].append([])
            # msw_flag = np.concatenate((msw_flag,-1))
            # night_flag = np.concatenate((night_flag,-1))
            # asr = np.concatenate((asr,-1))
                Eg[k].append([-1])
                Ev[k].append([-1])
                EvEg[k].append([-1])
                trad_cc[k].append([-1])
                for var in variable_names:
                    var_dict[f"{var}"][k].append([-1])
                beam[k].append([-1])
            print(f"Failed to open ATL03 file for {foldername} file {file_index}'s beam {i+1}.")
            continue
            
        try:
            atl08 = get_atl08_struct(atl08path, gt, atl03)
        except (KeyError, ValueError, OSError) as e:
            for k in range(len(lats)*len(lons)):
                plotX[k].append([])
                plotY[k].append([])
                
                Eg[k].append([-1])
                Ev[k].append([-1])
                EvEg[k].append([-1])
                trad_cc[k].append([-1])
                for var in variable_names:
                    var_dict[f"{var}"][k].append([-1])
                beam[k].append([-1])
            print(f"Failed to open ATL08 file for {foldername} file {file_index}'s beam {i+1}.")
            continue
        
        atl03.df = atl03.df[(atl03.df['lon_ph'] >= min_lon) & (atl03.df['lon_ph'] <= max_lon) &\
                                (atl03.df['lat_ph'] >= min_lat) & (atl03.df['lat_ph'] <= max_lat)]
        atl08.df = atl08.df[(atl08.df['longitude'] >= min_lon) & (atl08.df['longitude'] <= max_lon) &\
                                (atl08.df['latitude'] >= min_lat) & (atl08.df['latitude'] <= max_lat)]
        
        atl08.df = atl08.df[(atl08.df.photon_rate_can_nr < 100) & (atl08.df.photon_rate_te < 100)]# & (atl08.df.h_canopy < 100)]
        

        # NEW BIT FOR LAND COVER CLASSIFICATION ##############################################################################
        # print(atl08.df['landcover'])
        atl08.df = atl08.df[~atl08.df['segment_landcover'].isin([50, 80, 100])]
        if altitude != None:
            atl08.df = atl08.df[abs(atl08.df['h_te_interp'] - altitude) <= alt_thresh]
        # print(atl08.df['landcover'])
        
        k = 0
        for lat in lats:
            for lon in lons:
                polygon = make_box((lon,lat), small_box/2,small_box/2)
                sub_min_lon, sub_min_lat, sub_max_lon, sub_max_lat = polygon.total_bounds
                atl03_temp = atl03.df[(atl03.df['lon_ph'] >= sub_min_lon) & (atl03.df['lon_ph'] <= sub_max_lon) &\
                                        (atl03.df['lat_ph'] >= sub_min_lat) & (atl03.df['lat_ph'] <= sub_max_lat)].copy()
                atl08_temp = atl08.df[(atl08.df['longitude'] >= sub_min_lon) & (atl08.df['longitude'] <= sub_max_lon) &\
                                        (atl08.df['latitude'] >= sub_min_lat) & (atl08.df['latitude'] <= sub_max_lat)].copy()
                
                
                if atl08_temp.shape[0] == 0:
                    plotX[k].append([])
                    plotY[k].append([])
                    
                    Eg[k].append([-1])
                    Ev[k].append([-1])
                    EvEg[k].append([-1])
                    trad_cc[k].append([-1])
                    for var in variable_names:
                        var_dict[f"{var}"][k].append([-1])
                    beam[k].append([-1])
                    k += 1
                    continue
                # Retrieve the canopy fraction (fraction of segments that contain any
                # canopy photons) if the user wants it.
        
                # X and Y are data for the regression
                X = atl08_temp.photon_rate_te
                Y = atl08_temp.photon_rate_can_nr
        
                # Save it for plotting after the loop goes through all the groundtracks
                plotX[k].append(X)
                plotY[k].append(Y)
        
#         if atl03.df.size != 0:
#             # Save the ATL03 object
#             atl03s.append(atl03)
#             colors.append(i)
            
        
                if len(Y) < threshold:
                    print(f'Beam {i + 1}, box {k} in {foldername} file {file_index} has insufficient data.')
                    Eg[k].append([-1])
                    Ev[k].append([-1])
                    EvEg[k].append([-1])
                    trad_cc[k].append([-1])
                    for var in variable_names:
                        var_dict[f"{var}"][k].append([-1])
                    beam[k].append([-1])
                    k += 1
                    continue
                else:
                    atl03s[k].append(atl03)
                    colors[k].append(i)

                    Eg[k].append(X)
                    Ev[k].append(Y)
                    EvEg[k].append(Y/X)
                    trad_cc[k].append((atl08_temp['n_ca_photons']+atl08_temp['n_toc_photons'])/\
                                             (atl08_temp['n_ca_photons']+atl08_temp['n_toc_photons']+atl08_temp['n_te_photons']))
                    for var in variable_names:
                        # print(var, atl08_temp[var])
                        var_dict[f"{var}"][k].append(atl08_temp[var])
                    
                    if i % 2 == 0:
                        beam[k].append(['strong' for _ in range(len(atl08_temp['n_ca_photons']))])
                            # print(strong_dict[f"{var}_strong"])
                    else:
                        beam[k].append(['weak' for _ in range(len(atl08_temp['n_ca_photons']))])
                    
            
                # Save each individual data point from the ground track along with the Beam it belongs to.
                for x, y in zip(X,Y):
                    dataset[k].append([x, y, beam_names[i]])

                # tweaking starting parameters
                ############################################################
                # if len(Y) == 1:
                #     slope = -1
                #     intercept = 1
                # else:
                #     lower_X, lower_Y, upper_X, upper_Y = divide_arrays_2(X, Y)

                #     y1 = np.mean(lower_Y)
                #     y2 = np.mean(upper_Y)

                #     x1 = np.mean(lower_X)
                #     x2 = np.mean(upper_X)

                #     slope, intercept = find_slope_and_intercept(x1, y1, x2, y2)
                #     # print(slope)
                #     if slope > -0.1:
                #         slope = -0.1
                #         intercept = intercept_from_slope_and_point(slope, (np.mean([x1,x2]),np.mean([y1,y2])))
                #     elif slope < -1.5:
                #         slope = -1.5
                #         intercept = intercept_from_slope_and_point(slope, (np.mean([x1,x2]),np.mean([y1,y2])))
                
                # slope_init[k].append(slope)
                # slope_weight[k].append(len(Y))
                # # Save the initial y_intercept guess
                # intercepts[k].append(intercept)
                # maxes[k].append(16)
                
                k += 1
        #############################################################
                continue
            
    rows = []
    
    k = 0
    for lat in lats:
        for lon in lons:
            if len(dataset[k]) == 0:
                k+=1
                continue
            
            # slope_weight[k] /= np.sum([slope_weight[k]])
            # slope_init[k] = np.dot(slope_init[k],slope_weight[k])
            
            # Create DataFrame
            # df = pd.DataFrame(dataset[k], columns=['Eg', 'Ev', 'gt'])
            # # Dummy encode the categorical variable
            # df_encoded = pd.get_dummies(df, columns=['gt'], prefix='', prefix_sep='')
            
            # coefs = odr(df_encoded, intercepts = intercepts[k], maxes = maxes[k], init = slope_init[k],\
            #             lb=lb, ub=ub, model = model, res = res, loss=loss, f_scale=f_scale)
            
            
            # if len(colors) == 0:
            #     graph_detail = 0
                
            # if graph_detail == 3:
            #     plot_parallel(atl03s = atl03s[k],
            #                   coefs = coefs,
            #                   colors = colors[k],
            #                   title_date = title_date,
            #                   X = plotX[k],
            #                   Y = plotY[k],
            #                   beam = beam,
            #                   file_index = file_index,
            #                   three = True)
                
            # elif graph_detail == 2:
            #     plot_parallel(atl03s = atl03s[k],
            #                   coefs = coefs,
            #                   colors = colors[k],
            #                   title_date = title_date,
            #                   X = plotX[k],
            #                   Y = plotY[k],
            #                   beam = beam,
            #                   file_index = file_index)

            # # Activate this if you don't want the groundtracks, just the plot
            # elif graph_detail == 1:
            #     plot_graph(coefs = coefs,
            #                colors = colors[k],
            #                title_date = title_date,
            #                X = plotX[k],
            #                Y = plotY[k],
            #                beam = beam,
            #                file_index = file_index)

            # print(asr[k])
            # print(meanEgstrong[k])

            # print(non_negative_subset(asr[k]))

            # if len(meanEgstrong) > 0:
            #     EvEg[k] = safe_mean(non_negative_subset(meanEvstrong[k]))/safe_mean(non_negative_subset(meanEgstrong[k]))
            
            # indices_to_insert = [i + 1 for i, entry in enumerate(asr[k]) if entry == -1]
            # for index in indices_to_insert:
            #     coefs = np.insert(coefs, index, -1)

            # y_strong = np.mean(non_negative_subset([coefs[1],coefs[3],coefs[5]]))
            # y_weak = np.mean(non_negative_subset([coefs[2],coefs[4],coefs[6]]))
            # print(non_negative_subset(msw_flag[k]),msw_flag[k])

            # Append the row dynamically
            for j in range(len(non_negative_subset(Eg[k]))):
                row_data = [foldername, table_date, lon, lat,
                            non_negative_subset(Eg[k])[j], non_negative_subset(Ev[k])[j],
                            non_negative_subset(EvEg[k])[j],
                            non_negative_subset(trad_cc[k])[j], non_negative_subset(beam[k])[j]]

                # Add the rest of the strong-weak pairs dynamically
                for var in variable_names:  # Start from msw, as meanEg and meanEv are already included
                    var = f"{var}"
                    row_data.append(non_negative_subset(var_dict[var][k])[j])
                # Append the row to the rows list
                rows.append(row_data)
            k+=1

    columns_list = ['camera', 'date', 'lon', 'lat', 'Eg', 'Ev', 'EvEg', 'trad_cc','beam']
    for var in variable_names:  # Start from msw, as meanEg and meanEv are already included
        columns_list.append(f"{var}")
    
    BIG_DF = pd.DataFrame(rows,columns=[columns_list])
            
    return BIG_DF
