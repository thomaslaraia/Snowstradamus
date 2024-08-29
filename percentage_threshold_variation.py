from scripts.elevation_masking import *

start_date = '2018-11-01'
end_date = '2023-11-30'
center_lat = 45.8238
center_lon = 7.5609
radius = 5000  # 5 km radius
location_elevation = 2091
percentages = np.arange(0.4,0.91,0.05)
hdf_dir = '../data_store/data/MOD10A1F_torgnon/'
tif_path = '../data_store/data/torgnon/elevation/w50535_s10.tif'
tolerances = np.arange(50, 251, 10)

for percentage in percentages:
    monthly_dissimilarity_results, monthly_unmasked_dissimilarity = perform_analysis(
        start_date, end_date, center_lat, center_lon, radius, location_elevation, hdf_dir, tif_path, tolerances, percentage
    )
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab20.colors[:12]
    
    for month in range(1, 13):
        plt.plot(tolerances, monthly_dissimilarity_results[month], marker='o', color=colors[month-1], label=f'Month {month} (Masked)')
        plt.plot([tolerances.min(), tolerances.max()], [monthly_unmasked_dissimilarity[month]]*2, color=colors[month-1], linestyle='--', label=f'Month {month} (Unmasked)')
    
    plt.xlabel('Elevation Tolerance (m)')
    plt.ylabel('Average Within-Cluster Dissimilarity')
    plt.title(f'Average Dissimilarity between MOD10A1F and Elevation Threshold by Month - {percentage*100}% Pixel within Threshold')
    plt.legend(title='Month', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.savefig(f'./images/elevation_tolerance_{int(percentage*100)}_percent.png')
    plt.close()
    # plt.show()
