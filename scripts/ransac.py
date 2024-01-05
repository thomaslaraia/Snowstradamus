from scripts.imports import os, glob, pdb, np, h5py, pd, xr, gpd, Proj, Transformer, CRS, \
                        plt, cmap, Model, Data, ODR, datetime, rasterio, show, \
                        ccrs, cfeature
from sklearn.linear_model import RANSACRegressor
from scripts.odr import odr
#from scipy.odr import Model, RealData, ODR
from sklearn.utils import check_random_state

def model(params, x):
    return params[0]*x + params[1]
    
def residuals(params, x, y):
    return np.abs(model(params, x) - y)/np.sqrt(1 + params[0]**2)

# Define a custom ODR estimator
class ODRLinearEstimator:
    def __init__(self):
        self.coef_ = None

    def fit(self, X, y):
        model_func = Model(model)
        data = RealData(X, y)
        odr = ODR(data, model_func, beta0=[0, 1])
        result = odr.run()
        self.coef_ = result.beta
        return self

    def predict(self, X):
        if self.coef_ is None:
            raise ValueError("ODRLinearEstimator is not fitted.")
        return model(self.coef_, X)
        
# RANSAC algorithm with ODR
def ransac_odr(X, y, loss='arctan', min_samples=1, residual_threshold=None, max_trials=100, random_state=None):
    random_state = check_random_state(random_state)
    best_model = None
    best_inlier_mask = None
    best_inlier_count = 0

    for _ in range(max_trials):
        sample_indices = random_state.choice(len(X), size=int(min_samples), replace=False)
        X_sample = X.iloc[sample_indices].values
        y_sample = y.iloc[sample_indices].values

        a, b = odr(X_sample, y_sample, init=[-1, np.max(y)], res=residuals, loss = loss)

        inlier_mask = np.abs(y - model([a, b], X.values)) < residual_threshold
        inlier_count = np.sum(inlier_mask)

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_model = ODRLinearEstimator()  # Instantiate your custom ODRLinearEstimator class
            best_model.coef_ = [a, b]
            best_inlier_mask = inlier_mask

    return best_model, best_inlier_mask

# Run RANSAC with ODR
def run_ransac(X, Y, loss='arctan', rt = 6):
    ransac_model, inlier_mask = ransac_odr(X, Y, loss=loss, min_samples=0.8 * len(X), residual_threshold=rt, max_trials=100, random_state=42)
    a, b = ransac_model.coef_
    return a, b, ransac_model, inlier_mask

def plot_ransac(X, Y, ransac_model, inlier_mask, ax):
    
    # Plot the original data
    ax.scatter(X.values, Y.values, label='Original Data')
    
    # Plot the inliers identified by RANSAC with ODR
    ax.scatter(X.values[inlier_mask], Y.values[inlier_mask], color='red', label='Inliers')
    
    # Plot the RANSAC regression line with ODR
    x_range = np.linspace(0, 12, 100).reshape(-1, 1)
    y_pred = ransac_model.predict(x_range)
    ax.plot(x_range, y_pred, color='green', label='RANSAC Regression with ODR')
    
    ax.set_xlabel('Eg')
    ax.set_ylabel('Ev')
    #ax.legend()
    ax.set_title('RANSAC Regression with Orthogonal Distance Regression (ODR)')
    
    return ax
