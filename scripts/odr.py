from scripts.imports import os, glob, pdb, np, h5py, pd, xr, gpd, Proj, Transformer, CRS, \
                        plt, cmap, Model, Data, ODR, datetime, rasterio, show, \
                        ccrs, cfeature
from scripts.classes_fixed import *
from scipy.optimize import least_squares
# from sklearn.metrics import r2_score, mean_squared_error

def model(params, x):
    return params[0]*x + params[1]
    
def residuals(params, x, y):
    return np.abs(model(params, x) - y)/np.sqrt(1 + params[0]**2)
    
def odr(X,Y, init, res = residuals, loss='linear', bounds=([-100, 0], [-1/100, 16]), f_scale=.1):
    result = least_squares(res, init, loss = loss, f_scale=f_scale, args=(X,Y), bounds = bounds)
    a, b = result.x
    return a, b

def parallel_model(params, x, num_datasets):
    common_slope, *intercepts = params
    return [common_slope * x[i] + intercept for i, intercept in enumerate(intercepts)]

def parallel_residuals(params, x, num_datasets, *datasets):
    model_outputs = parallel_model(params, x, num_datasets)
    
    all_residuals=[]
    for model_output, dataset in zip(model_outputs, datasets):
        residuals = model_output - dataset
        non_nan_residuals = residuals[~np.isnan(residuals)]
        all_residuals.extend(non_nan_residuals)
        
    return np.array(all_residuals)

def parallel_odr(datasets, init = -1, lb = -100, ub = -1/100, res = parallel_residuals, loss='linear', f_scale=.1):
    xs, ys = zip(*datasets)
    
    a = [lb] + [0]*len(ys)
    b = [ub] + [16]*len(ys)
    bounds = (a,b)
    
    initial_params = [init] + [np.max(y) for y in ys]
    
    params = least_squares(res, initial_params, loss = loss, f_scale=f_scale, args=(xs, len(ys), *ys), bounds = bounds).x
    
    return params
    
    
"""
def parallel_model(params, x, num_datasets):
    common_slope, *intercepts = params
    return [common_slope*x + intercept 
    
    
def parallel_residuals(params, x, all_y):
    models = parallel_model(params, all_x)
    
    
    
    res = 0
    for i, (x, y) in enumerate(zip(all_x, all_y)):
        res += np.abs(model([params[0],params[1+i]], x) - y)/np.sqrt(1 + params[0]**2)
        print(res)
    return res
    
def parallel_odr(all_X, all_Y, init, res = parallel_residuals, loss='linear', bounds=([-100,0], [-1/100, 16]), f_scale=.1):
    result = least_squares(res, init, loss = loss, f_scale = f_scale, args=(all_X, all_Y), bounds = bounds)
    coefs = result.x
    return coefs
"""
