from scripts.imports import os, glob, pdb, np, h5py, pd, xr, gpd, Proj, Transformer, CRS, \
                        plt, cmap, Model, Data, ODR, datetime, rasterio, show, \
                        ccrs, cfeature
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
from scipy.odr import ODR, Model, Data
from scripts.odr import odr

def model(params, x):
    return params[0]*x + params[1]
    
def residuals(params, x, y):
    return np.abs(model(params, x) - y)/np.sqrt(1 + params[0]**2)

def ransac_lin(X, Y, init, res = residuals, loss = 'linear', bounds = ([-100, 0], [-1/100, 16]), f_scale = 0.1):
    #print(X)
    X_ = np.array(X).reshape(-1,1)
    #print(X)
    #Y = np.array(Y).reshape(-1,1)
    #print(Y)

    # apply RANSAC regressor
    ransac = RANSACRegressor(LinearRegression(), random_state=42)
    ransac.fit(X_,Y)
    
    # separate inliers and outliers
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    
    # fit a line through the inliers
    a, b = odr(X[inlier_mask], Y[inlier_mask], res = res, init=init, loss=loss, bounds=bounds, f_scale=f_scale)
    return a, b, inlier_mask, outlier_mask
    
# def ransac_ODR(X, Y, init, res = residuals, loss = 'linear', bounds = ([-100, 0], [-1/100, 16]), f_scale = 0.1):

#     X_ = np.array(X).reshape(-1,1)

#     ransac = RANSACRegressor(min_samples=3, residual_threshold=2.0, is_data_valid=None, is_model_valid=None, random_state=42)
    
#     def fit_odr_model(X,Y):
#         linear = Model(model)
#         data = Data(X,Y)
        
#         odr = ODR(data, linear, beta0=[1.0,np.max(Y)])
#         odr_run = odr.run()
        
#         return odr_run.beta
    
#     ransac.fit(X_, Y, model_estimation_func=fit_odr_model)
    
#     inlier_mask = ransac.inlier_mask_
#     outlier_mask = np.logical_not(inlier_mask)
    
#     a, b = ransac.estimator_.beta
    
#     return a, b, inlier_mask, outlier_mask
