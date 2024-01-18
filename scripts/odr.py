from scripts.imports import os, glob, pdb, np, h5py, pd, xr, gpd, Proj, Transformer, CRS, \
                        plt, cmap, Model, Data, ODR, datetime, rasterio, show, \
                        ccrs, cfeature
from scripts.classes_fixed import *
from scipy.optimize import least_squares
# from sklearn.metrics import r2_score, mean_squared_error

#This is a fairly standard linear model
def model(params, x):
    return params[0]*x + params[1]

# This defines the residuals orthogonal to the regression line
def residuals(params, x, y):
    return (y - model(params, x))/np.sqrt(1 + params[0]**2)

# This performs ODR regression for Y against X with initial guess init using least_squares()
def odr(X,Y, init, res = residuals, loss='linear', bounds=([-100, 0], [-1/100, 16]), f_scale=.1):
    result = least_squares(res, init, loss = loss, f_scale=f_scale, args=(X,Y), bounds = bounds)
    
    # a is the slope of the line, b is the intercept
    a, b = result.x
    return a, b
