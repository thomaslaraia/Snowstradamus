from scripts.FSC_dataframe_phoreal import *

from scipy.optimize import fsolve

def hist_plot(loc_df, hue_labels, X, Hue, save=None, plot=True, xlim=None, ylim=None, preset=True):
    if not preset:
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

        # Combine KDE data for all labels to find overall max density at each x
        all_x = np.unique(np.concatenate([x for x, _ in kde_data.values()]))
        combined_y = {label: np.interp(all_x, x, y) for label, (x, y) in kde_data.items()}

        # Find the dominant label at each x
        dominant_labels = []
        for x in all_x:
            max_y = 0
            dominant_label = None
            for label, y in combined_y.items():
                if y[np.where(all_x == x)[0][0]] > max_y:
                    max_y = y[np.where(all_x == x)[0][0]]
                    dominant_label = label
            dominant_labels.append(dominant_label)

        # Find transitions in the dominant label
        transitions = []
        for i in range(1, len(dominant_labels)):
            if dominant_labels[i] != dominant_labels[i - 1]:
                # Find intersection between the previous and current dominant label
                x1, y1 = kde_data[dominant_labels[i - 1]]
                x2, y2 = kde_data[dominant_labels[i]]
                intersections = find_intersections(x1, y1, x2, y2)
                for x in intersections:
                    if all_x[i - 1] < x < all_x[i]:  # Ensure the intersection is within the range
                        transitions.append(x)
                        break

    else:
        if Hue == 'FSC':
            transitions = [0.14, 0.72]
        elif Hue == 'JointSnow':
            transitions = [0.15, 0.43, 0.82]

    if not plot:
        return transitions

    # Plotting the KDE with intersection points
    plt.figure()
    

# scatter.legend(handles=handles, labels=new_labels, loc='upper right')
    Plot = sns.histplot(loc_df, x=X, hue=Hue, kde=True, palette='tab10')
    
    # handles, labels = Plot.get_legend_handles_labels()
    # print(labels)
    # new_labels = [hue_labels[label] if label in hue_labels else label for label in labels]
    # Plot.legend(handles=handles, labels=new_labels, loc='upper right')
    for x in transitions:
        plt.axvline(x, color='red', linestyle='--')  # Red dashed lines for intersections
    if save is not None:
        plt.savefig(f'./images{save}')
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    plt.show()
    return transitions