from scripts.FSC_dataframe_phoreal import *

from scipy.optimize import fsolve

def hist_plot(loc_df, hue_labels, X, Hue, save=None, plot=True):
    plt.figure()

    Plot = sns.histplot(loc_df, x=X, hue=Hue, kde=True, palette='tab10')

    # Extracting the KDE lines from the seaborn plot
    lines = Plot.get_lines()

    # Getting the x and y data for each KDE line
    kde_data = {}
    for line in lines:
        label = line.get_label()
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        kde_data[label] = (x_data, y_data)

    plt.close()

    # Function to find intersections between two KDE lines
    def find_intersections(x1, y1, x2, y2):
        # Ensure arrays are numpy arrays
        x1, y1, x2, y2 = np.array(x1), np.array(y1), np.array(x2), np.array(y2)
        
        # Create an interpolation function for each KDE
        f1 = np.interp
        f2 = np.interp
        
        # Define the function to find the roots (intersections)
        def func(x):
            return f1(x, x1, y1) - f2(x, x2, y2)
        
        # Find the range where intersections might occur
        x_min = max(min(x1), min(x2))
        x_max = min(max(x1), max(x2))
        
        # Generate a range of x values within the overlap
        x_values = np.linspace(x_min, x_max, 1000)
        
        # Use fsolve to find the roots
        intersections = []
        for x in x_values:
            root = fsolve(func, x)
            if x_min <= root <= x_max:
                intersections.append(root[0])
        
        # Remove duplicate intersections
        intersections = sorted(set(intersections))
        
        return intersections
    
    # Function to find the peak of a KDE
    def find_peak(x, y):
        peak_index = np.argmax(y)
        return x[peak_index], y[peak_index]
    
    # Identify the dominant KDE at each peak and find the earliest intersection to the right
    first_intersections = []
    labels = list(kde_data.keys())
    dominant_peaks = []
    
    for label1 in labels:
        x1, y1 = kde_data[label1]
        peak_x1, peak_y1 = find_peak(x1, y1)
        dominant_peaks.append((peak_x1, peak_y1, label1))
    
    # Sort peaks by x value
    dominant_peaks.sort()
    
    # Find the earliest intersection to the right of each dominant peak
    for peak_x1, peak_y1, label1 in dominant_peaks:
        x1, y1 = kde_data[label1]
        earliest_intersection = None
        for label2 in labels:
            if label1 == label2:
                continue
            x2, y2 = kde_data[label2]
            
            intersections = find_intersections(x1, y1, x2, y2)
            
            for x in intersections:
                if x > peak_x1:
                    if earliest_intersection is None or x < earliest_intersection:
                        earliest_intersection = x
                    break
    
        if earliest_intersection is not None:
            first_intersections.append(earliest_intersection)
    
    # Sort the first intersections in increasing order
    first_intersections = sorted(first_intersections)

    if plot == False:
        return first_intersections
    
    # Plotting the KDE with intersection points
    plt.figure()
    Plot = sns.histplot(loc_df, x=X, hue=Hue, kde=True, palette='tab10')
    for x in first_intersections:
        plt.axvline(x, color='red', linestyle='--')  # Red dashed lines for intersections
    if save != None:
        plt.savefig(f'./images{save}')
    plt.show()
    return first_intersections