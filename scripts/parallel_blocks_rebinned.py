from scripts.parallel_phoreal import *

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

def pvpg_parallel(dirpath, atl03path, atl08path, coords, width=.1, height=.1, f_scale = .1, loss = 'arctan', init = -.6,\
                  lb = -np.inf, ub = 0,file_index = None, model = parallel_model, res = parallel_residuals,\
                  odr = parallel_odr, zeros=None,beam = None, y_init = np.max, graph_detail = 0, keep_flagged=True,\
                  opsys='bad', altitude=None,alt_thresh=200, threshold = 2, small_box = 0.01, res_field='alongtrack',res=30):
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

    lats = np.arange(min_lat+small_box/2, max_lat+small_box/2, small_box)
    lons = np.arange(min_lon+small_box/(2*np.cos(np.radians(coords[1]))),\
                     max_lon+small_box/(2*np.cos(np.radians(coords[1]))),\
                     small_box/np.cos(np.radians(coords[1])))
    
    foldername = dirpath.split('/')[-2]
    # print(lats, lons)
    
    # This will hold all of the data in one place:
    # [[Eg, Ev, Beam 1],...[Eg,Ev,Beam 1],[Eg,Ev,Beam 2],...,[Eg,Ev,Beam6],[Eg,Ev,Beam 6]]
    # This will be made into a dataframe later.
    meanEgstrong = [[] for _ in range(len(lats)*len(lons))]
    meanEgweak = [[] for _ in range(len(lats)*len(lons))]
    meanEvstrong = [[] for _ in range(len(lats)*len(lons))]
    meanEvweak = [[] for _ in range(len(lats)*len(lons))]

    msw_flag = [[] for _ in range(len(lats)*len(lons))]
    night_flag = [[] for _ in range(len(lats)*len(lons))]
    asr = [[] for _ in range(len(lats)*len(lons))]
    n_photons = [[] for _ in range(len(lats)*len(lons))]
    
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

    data_amount = np.zeros(len(lats)*len(lons))
    
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
    intercepts = [[] for _ in range(len(lats)*len(lons))]
    maxes = [[] for _ in range(len(lats)*len(lons))]
    
    # Now that we have assurances that the data is good quality,
    # we loop through the ground tracks
    for i, gt in enumerate(tracks):
        
        # If the object fails to be created, we put worthless information into
        # plotX, plotY, and canopy_frac to save us looping effort later
        try:
#             print(atl03path, gt, atl08path)
            atl03 = get_atl03_struct(atl03path, gt, atl08path)
        except (KeyError, ValueError, OSError) as e:
            for k in range(len(lats)*len(lons)):
                plotX[k].append([])
                plotY[k].append([])
            # msw_flag = np.concatenate((msw_flag,-1))
            # night_flag = np.concatenate((night_flag,-1))
            # asr = np.concatenate((asr,-1))
                msw_flag[k].append(-1)
                night_flag[k].append(-1)
                asr[k].append(-1)
                n_photons[k].append(-1)
                if i % 2 == 0:
                    meanEgstrong[k].append(-1)
                    meanEvstrong[k].append(-1)
                else:
                    meanEgweak[k].append(-1)
                    meanEvweak[k].append(-1)
            print(f"Failed to open ATL03 file for {foldername} file {file_index}'s beam {i+1}.")
            continue
            
        try:
            atl08 = get_atl08_struct(atl08path, gt, atl03)
        except (KeyError, ValueError, OSError) as e:
            for k in range(len(lats)*len(lons)):
                plotX[k].append([])
                plotY[k].append([])
                msw_flag[k].append(-1)
                night_flag[k].append(-1)
                asr[k].append(-1)
                n_photons[k].append(-1)
                if i % 2 == 0:
                    meanEgstrong[k].append(-1)
                    meanEvstrong[k].append(-1)
                else:
                    meanEgweak[k].append(-1)
                    meanEvweak[k].append(-1)
            print(f"Failed to open ATL08 file for {foldername} file {file_index}'s beam {i+1}.")
            continue

        
        atl03.df = atl03.df[(atl03.df['lon_ph'] >= min_lon) & (atl03.df['lon_ph'] <= max_lon) &\
                                (atl03.df['lat_ph'] >= min_lat) & (atl03.df['lat_ph'] <= max_lat)]
        atl08.df = atl08.df[(atl08.df['longitude'] >= min_lon) & (atl08.df['longitude'] <= max_lon) &\
                                (atl08.df['latitude'] >= min_lat) & (atl08.df['latitude'] <= max_lat)]

        atl08.df = rebin_atl08(atl03, atl08, gt, res, res_field)
        
        atl08.df = atl08.df[(atl08.df.photon_rate_can_nr < 100) & (atl08.df.photon_rate_te < 100) & (atl08.df.h_canopy < 100)]
        

        # NEW BIT FOR LAND COVER CLASSIFICATION ##############################################################################
        # print(atl08.df['landcover'])
        atl08.df = atl08.df[atl08.df['segment_landcover'].isin([111, 112, 113, 114, 115, 116, 121, 122, 123, 124, 125, 126])]
        if altitude != None:
            atl08.df = atl08.df[abs(atl08.df['h_te_best_fit'] - altitude) <= alt_thresh]
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
                    msw_flag[k].append(-1)
                    night_flag[k].append(-1)
                    asr[k].append(-1)
                    n_photons[k].append(-1)
                    plotX[k].append([])
                    plotY[k].append([])
                    if i % 2 == 0:
                        meanEgstrong[k].append(-1)
                        meanEvstrong[k].append(-1)
                    else:
                        meanEgweak[k].append(-1)
                        meanEvweak[k].append(-1)
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
                    print(f'Beam {i + 1}, box {k} in file {file_index} has insufficient data.')
                    msw_flag[k].append(-1)
                    night_flag[k].append(-1)
                    asr[k].append(-1)
                    n_photons[k].append(-1)
                    if i % 2 == 0:
                        meanEgstrong[k].append(-1)
                        meanEvstrong[k].append(-1)
                    else:
                        meanEgweak[k].append(-1)
                        meanEvweak[k].append(-1)
                    k += 1
                    continue
                else:
                    data_amount[k] += len(Y)
                    atl03s[k].append(atl03)
                    colors[k].append(i)

                    if i % 2 == 0:
                        meanEgstrong[k].append(np.mean(X))
                        meanEvstrong[k].append(np.mean(Y))
                    else:
                        meanEgweak[k].append(np.mean(X))
                        meanEvweak[k].append(np.mean(Y))

                    msw_flag[k].append(atl08_temp['msw_flag'].mean())
                    night_flag[k].append(atl08_temp['night_flag'].mean())
                    asr[k].append(atl08_temp['asr'].mean())
                    n_photons[k].append(atl08_temp['n_seg_ph'].mean())
            
                # Save each individual data point from the ground track along with the Beam it belongs to.
                for x, y in zip(X,Y):
                    dataset[k].append([x, y, beam_names[i]])

                # tweaking starting parameters
                ############################################################
                lower_X, lower_Y, upper_X, upper_Y = divide_arrays_2(X, Y)

                y1 = np.mean(lower_Y)
                y2 = np.mean(upper_Y)

                x1 = np.mean(lower_X)
                x2 = np.mean(upper_X)

                slope, intercept = find_slope_and_intercept(x1, y1, x2, y2)
                # print(slope)
                if slope > -0.1:
                    slope = -0.1
                    intercept = intercept_from_slope_and_point(slope, (np.mean([x1,x2]),np.mean([y1,y2])))
                elif slope < -1.5:
                    slope = -1.5
                    intercept = intercept_from_slope_and_point(slope, (np.mean([x1,x2]),np.mean([y1,y2])))
                
                slope_init[k].append(slope)
                slope_weight[k].append(len(Y))
                # Save the initial y_intercept guess
                intercepts[k].append(intercept)
                maxes[k].append(16)
                
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
            
            slope_weight[k] /= np.sum([slope_weight[k]])
            slope_init[k] = np.dot(slope_init[k],slope_weight[k])
            
            # Create DataFrame
            df = pd.DataFrame(dataset[k], columns=['Eg', 'Ev', 'gt'])
            # Dummy encode the categorical variable
            df_encoded = pd.get_dummies(df, columns=['gt'], prefix='', prefix_sep='')
            
            coefs = odr(df_encoded, intercepts = intercepts[k], maxes = maxes[k], init = slope_init[k],\
                        lb=lb, ub=ub, model = model, res = res, loss=loss, f_scale=f_scale)
            
            
            if len(colors) == 0:
                graph_detail = 0
                
            if graph_detail == 3:
                plot_parallel(atl03s = atl03s[k],
                              coefs = coefs,
                              colors = colors[k],
                              title_date = title_date,
                              X = plotX[k],
                              Y = plotY[k],
                              beam = beam,
                              file_index = file_index,
                              three = True)
                
            elif graph_detail == 2:
                plot_parallel(atl03s = atl03s[k],
                              coefs = coefs,
                              colors = colors[k],
                              title_date = title_date,
                              X = plotX[k],
                              Y = plotY[k],
                              beam = beam,
                              file_index = file_index)

            # Activate this if you don't want the groundtracks, just the plot
            elif graph_detail == 1:
                plot_graph(coefs = coefs,
                           colors = colors[k],
                           title_date = title_date,
                           X = plotX[k],
                           Y = plotY[k],
                           beam = beam,
                           file_index = file_index)
            
            means = [np.mean(non_negative_subset(meanEgstrong[k])), np.mean(non_negative_subset(meanEgweak[k])),\
                                                                            np.mean(non_negative_subset(meanEvstrong[k])),\
                                                                            np.mean(non_negative_subset(meanEvweak[k]))]
            indices_to_insert = [i + 1 for i, entry in enumerate(asr[k]) if entry == -1]
            for index in indices_to_insert:
                coefs = np.insert(coefs, index, -1)

            y_strong = np.mean(non_negative_subset([coefs[1],coefs[3],coefs[5]]))
            y_weak = np.mean(non_negative_subset([coefs[2],coefs[4],coefs[6]]))
            # print(non_negative_subset(msw_flag[k]),msw_flag[k])
            rows.append(flatten_structure([foldername, table_date, coefs[0], y_strong,y_weak,-y_strong/coefs[0],-y_weak/coefs[0],\
                                           [lon,lat], means,\
                                           np.mean(non_negative_subset(msw_flag[k])), np.mean(non_negative_subset(night_flag[k])),\
                                           np.mean(non_negative_subset(asr[k])), np.mean(non_negative_subset(n_photons[k])),\
                                           data_amount[k]]))
            #print([mid_date, coefs, [lon,lat],means,msw_flag[k],night_flag[k],asr[k],data_amount[k]])
            k+=1
    
    BIG_DF = pd.DataFrame(rows,columns=['camera','date','pvpg','y_strong','y_weak','x_strong','x_weak',\
                                        'longitude','latitude','meanEgstrong','meanEgweak','meanEvstrong','meanEvweak',\
                                        'msw','night','asr','n_photons','data_quantity'])
            
    return BIG_DF

