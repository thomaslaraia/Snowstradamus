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
    

def plot_concise(title_date, beam_names, atl03s, X, Y, A, B, I, file_index, tracks, beam = None, detail = 0, canopy_frac = None):
    if detail == 2:
        fig = plt.figure(figsize=(10, 6))
        ax1 = fig.add_subplot(331)
        ax2 = fig.add_subplot(332)
        ax3 = fig.add_subplot(334)
        ax4 = fig.add_subplot(335)
        ax5 = fig.add_subplot(337)
        ax6 = fig.add_subplot(338)
        axes = [ax1, ax2, ax3, ax4, ax5, ax6]
        ax7 = fig.add_subplot(133)
        
        # Set the figure title
        if file_index != None:
            fig.suptitle(title_date + ' - N = ' + str(file_index), fontsize=16)
        else:
            fig.suptitle(title_date, fontsize=16)
            
        for i, j, atl03 in zip(np.arange(len(I)), I, atl03s):
            if canopy_frac != None:
                atl03.plot_small(axes[j], f"{beam_names[j]} - Canopy Fraction = {round(canopy_frac[j],2)}")
            else:
                atl03.plot_small(axes[j], beam_names[j])
            
            if beam != None:
                if beam == j + 1:
                    ax7.scatter(X[j],Y[j], s=5, color=cmap2(j))
                    ax7.plot(np.array([0,12]), model([A[i], B[i]], np.array([0,12])), label=f"Beam {int(I[i]+1)}", color=cmap2(j), linestyle='--', zorder=2)
                    ax7.annotate(r'$\rho_v/\rho_g \approx {:.2f}$, Beam {}'.format(-A[i], int(i+1)),
                                   xy=(.95,.68-.045*i),
                                   xycoords='axes fraction',
                                   ha='right',
                                   va='top',
                                   fontsize=6,
                                   bbox=dict(boxstyle="round,pad=0.3",
                                             edgecolor="black",
                                             facecolor="white"))
            else:
                ax7.scatter(X[j],Y[j], s=5, color=cmap2(j))
                ax7.plot(np.array([0,12]), model([A[i], B[i]], np.array([0,12])), label=f"Beam {int(I[i]+1)}", color=cmap2(j), linestyle='--', zorder=2)
                ax7.annotate(r'$\rho_v/\rho_g \approx {:.2f}$, Beam {}'.format(-A[i], int(i+1)),
                               xy=(.95,.68-.045*i),
                               xycoords='axes fraction',
                               ha='right',
                               va='top',
                               fontsize=6,
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
        
    elif detail == 1:
        fig = plt.figure(figsize=(10, 6))
        
        # Set the figure title
        if file_index != None:
            fig.suptitle(title_date + ' - N = ' + str(file_index), fontsize=16)
        else:
            fig.suptitle(title_date, fontsize=16)
            
        for i, j, in zip(np.arange(len(I)), I):
            
            if beam != None:
                if beam == j + 1:
                    plt.scatter(X[j],Y[j], s=5, color=cmap2(j))
                    plt.plot(np.array([0,12]), model([A[i], B[i]], np.array([0,12])), label=f"Beam {int(I[i]+1)}", color=cmap2(j), linestyle='--', zorder=2)
                    plt.annotate(r'$\rho_v/\rho_g \approx {:.2f}$, Beam {}'.format(-A[i], int(i+1)),
                                   xy=(.081,0.98-.045*i),
                                   xycoords='axes fraction',
                                   ha='right',
                                   va='top',
                                   fontsize=6,
                                   bbox=dict(boxstyle="round,pad=0.3",
                                             edgecolor="black",
                                             facecolor="white"))
            else:
                plt.scatter(X[j],Y[j], s=5, color=cmap2(j))
                plt.plot(np.array([0,12]), model([A[i], B[i]], np.array([0,12])), label=f"Beam {int(I[i]+1)}", color=cmap2(j), linestyle='--', zorder=2)
                plt.annotate(r'$\rho_v/\rho_g \approx {:.2f}$, Beam {}'.format(-A[i], int(i+1)),
                               xy=(.99,0.69-.045*i),
                               xycoords='axes fraction',
                               ha='right',
                               va='top',
                               fontsize=6,
                               bbox=dict(boxstyle="round,pad=0.3",
                                         edgecolor="black",
                                         facecolor="white"))
        
        plt.title(f"Ev/Eg Rates", fontsize=8)
        plt.xlabel('Eg (returns/shot)')
        plt.ylabel('Ev (returns/shot)')
        plt.xlim(0,8)
        plt.ylim(0,20)
        plt.legend(loc='best')
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust the layout to make room for the suptitle
        plt.show()
    return
    
    
def pvpg_concise(atl03path, atl08path,f_scale = .1, loss = 'arctan', bounds = ([-100, 0], [-1/100, 16]), file_index = None, res = residuals, model = model, rt = None, zeros=None, beam = None, detail = 0, canopy_frac = None):
    """
    Regression of all tracks on a given overpass fit into a more concise visual representation

    atl03path - Path/to/ATL03/file
    atl08path - Path/to/matching/ATL08/file
    f_scale - Parameter in least_squares() function when loss is nonlinear, indiciating the value of the soft margin between inlier and outlier residuals.
    loss - string for loss parameter in least_squares().
    bounds - bounds for slope of line and y-intercept in shape ([--,--],[--,--]), allowing us to restrict the regression to realistic values.
    file_index - Index of file if cycling through an array of filenames, displayed in figure titles for a given file. Allows us to easily pick out strange cases for investigation.
    res - Default holds the ODR residuals function to be used in least_squares(). Can hold adjusted residual functions as well.
    model - model function to be used, e.g. params[0]*x + params[1]
    rt - this will trigger RANSAC regression, and is also equal to the residual threshold of the regression.
    zeros - whether or not to include outliers with zero canopy photon returns. Default = None.
    """
    
    A = h5py.File(atl03path, 'r')
    
    plotX = []
    plotY = []
    
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
    T = []
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
    
    # Extracting date and time from the filename
    title_date = parse_filename_datetime(atl03path)
    
    A = []
    B = []
    I = []
    atl03s = []
    
    if canopy_frac != None:
        B08 = h5py.File(atl08path, 'r')
        canopy_frac = []
    
    for i, gt in enumerate(tracks):
        
        try:
            atl03 = ATL03(atl03path, atl08path, gt)
        except (KeyError, ValueError, OSError) as e:
            plotX.append([])
            plotY.append([])
            if canopy_frac != None:
                canopy_frac.append(-1)
            continue
        if zeros == None:
            atl08 = ATL08(atl08path, gt)
        else:
            atl08 = ATL08_with_zeros(atl08path, gt)
            
        if canopy_frac != None:
            canopy_frac.append(np.array(list(B08[gt]['land_segments']['canopy']['subset_can_flag'])).flatten().mean())


        X = atl08.df.Eg
        Y = atl08.df.Ev
        
        plotX.append(X)
        plotY.append(Y)
        
        init = [-1, np.max(Y)]
        
        a_guess, b_guess = odr(X, Y, res = res, init=init, loss=loss, bounds=bounds, f_scale=f_scale)
        A.append(a_guess)
        B.append(b_guess)
        I.append(int(i))
        T.append(gt)
        atl03s.append(atl03)
        
    plot_concise(title_date=title_date,
                 beam_names = beam_names,
                 atl03s=atl03s,
                 X=plotX,
                 Y=plotY,
                 A = A, B = B,
                 I = I,
                 file_index = file_index,
                 tracks = T,
                 beam=beam,
                 detail = detail,
                 canopy_frac = canopy_frac)
    return A, B
