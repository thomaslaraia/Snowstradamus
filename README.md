# Snowstradamus

<p align="center">
  <img src="images/snowstradamus.png" alt="Project Logo" width="200"/>
</p>

Using ICESat-2 Lidar data to model fractional snow cover beneath forest canopies.

## Table of Contents

- [About](#about)
- [Prerequisites](#prerequisites)
- [Notebooks](#notebooks)
- [Scripts](#scripts)
- [Archive](#archive)
- [Acknowledgements](#acknowledgements)

## About

This project holds my work towards my PhD project at the University of Edinburgh, supervised by Steven Hancock. The purpose is to try to use satellite lidar to model fractional snow cover (FSC) beneath forest canopies. This is a known limitation of hydrology models, and an improvement to subcanopy snow modeling would improve the accuracy of such models. This would, for example, potentially give better estimates of water availability to regions that depend on snowpack for fresh water and aid in drought prediction. To the best of my knowledge, the current best model is SCAmod (Metsämäki et al., 2012), which uses satellite imagery (e.g. MODIS data) and transmissivity maps through an algorithm to estimate FSC and compare to FSC from Finnish snow course observations, Finnish weather station observations, and Landsat/ETM+ scenes (which are treated as "ground truth"). This model will be used as a benchmark for comparison.

## Prerequisites

To use these scripts, you need to have ATL03 and corresponding ATL08 data, which are available through NASA Earthdata. In most cases, the spatial subsetting tool in NASA Earthdata was used to preprocess the data to be restricted to a given region of interest. There is potential to investigate whether downloading the entire file and subsetting the data through a Python script could prevent data loss, but I haven't done that yet.

## Notebooks

'manual_pvpg_computation.ipynb': This is a template for later work if I want to pursue manual computation of the canopy:ground reflectance ratio for a given land segment. This is opposed to using ATL08 data for 100m land segments.

'parallel_regression.ipynb': This notebook uses scripts to perform ODR regression on all six (or however many exist in the ROI to a maximum of six) groundtracks simultaneously so that they have the same slope. This assumes that the atmospheric conditions are virtually identical for each track on a given overpass.

'quality_flagging.ipynb': This covers an investigation into the variables within the ATL03 and ATL08 data products to consider what may be useful to separate good quality tracks from back quality tracks. It also has some quick notes about variables that may be useful in future classification algorithms.

'regression_ODR_with_arctan_loss.ipynb': This notebook shows flaws in orthogonal distance regression with a linear loss function used in literature, demonstrates the use of a function to plot groundtracks over an ROI map (assuming you have a valid geotiff, I used Sentinel-2 data), and shows results from orthogonal distance regression using an arctan loss function to limit the impact of outliers on the regression. The fit of the line is closer to expectations. Additionally, in literature, data points with 0 canopy photons returned were removed entirely to deal with outliers such as lakes, which give extremely high ground photon rates and skew the regression. This method is more capable of dealing with such outliers instead of removing them. The used function allows for choice of loss function. Validation must still be performed on this method.

'shapefile_generation.ipynb': This generates files you can use to spatially subset regions in NASA Earthdata. I found that if I used the inbuilt tool on NASA Earthdata, I couldn't spatially subset when downloading data. This should really be a script, I'll get to that later.

'tracks_missing_ROI.ipynb': This is a brief investigation of the ATL03 files in the region of interest that data download failed to produce an ATL08 file for. The cause is not tracks missing the ROI but falling within the square block that the ROI is in, as surmised. Perhaps more work to come.

## Scripts

'classes_fixed.py': Script to define ATL03 and ATL08 classes, which are used to read the relevant h5py files. Based on code provided by Matt Purslow, which as been adjusted to allow a choice between removing outliers (original design) or keeping them.

'imports.py': Super boring script to import packages that I found myself importing a lot, but I wanted to take up less space.

'odr.py': Script to perform orthogonal distance regression with a loss function of your choice using 'scipy.optimize.least_squares'. Also home to the parallel regression functions used in 'parallel_regression.ipynb'.

'parallel.py': This contains functions to perform parallel slopes regression using two different methods. The methods, unfortunately, do not give the same results (problem), and they are also extremely sensitive to initial parameters (huge problem). This is far from complete.

'pvpg_fixed.py': Holds several functions involved with extracting canopy:ground reflectance ratios from a groundtrack.
The function 'pvpg' performs linear ODR on each groundtrack the function can load without much discrimination.
The function 'pvpg_flagged' does the same, but indicates which groundtracks have been flagged, as described in the 'quality_flagging.ipynb' notebook.
The function 'pvpg_penalized_flagged' skips over all files where at least one groundtrack has been flagged, and takes in input parameters to adjust the loss function, f_scale, bounds of parameters, residual function, model function, and whether RANSAC regression should be used (hint: it shouldn't).
The function 'pvpg_concise' manages to display all the information from 'pvpg_penalized_flagged' into a smaller, generally easier to digest figure.

'ransac.py': It was a pain to use orthogonal distance regression in a RANSACRegressor(), but as far as I know, it works. It didn't do that well in this context though, it's more complex and seems worse than the arctan ODR approach.

'show_tracks.py': Contains functions 'map_setup' and 'show_tracks' to visualize the tracks on a geotiff map that the user must provide. Users can use a colourmap to colour the points by canopy photon return rates or ground photon return rates to easily investigate outliers in the data (spoiler, usually lakes and marshes). This also now contains a function 'show_tracks_only_atl03' that shows the tracks that don't have matching ATL08 files.

'track_pairs.py': You have downloaded all your ATL03 and ATL08 files and thrown them into a data folder, but whoops, for some reason there's way more ATL03 files than ATL08 files! That's weird! Well, good thing you have this script to look through your folder, check which ATL03 files have a corresponding ATL08 file, and save these into an array. It even sorts it by date for you. There is also a parameter to devide if you also want to have a list of all the non-matching files.

## Archive

'regression_ransac_with_ODR_and_arctan_loss.ipynb.': I tried really hard to make RANSAC work, I really though that the having a regression method that automatically removes outliers would be good. Turns out it's hard to tune the initial regression to find the outliers that I want to be outliers. Just wasn't consistent enough.

'rovaniemi_tracks.ipynb': This is a pretty early exploratory notebook. In a more general sense, you can take a quick look at your tracks and an approximate pv/pg estimation using function 'pvpg'. Trust me, just use one of the non-archived notebooks instead to do the same thing but better.

'rovaniemi_tracks_phoreal.ipynb': Same note about choosing not to pursue using PhoREAL directly.

'strong_weak_beam_comparison.ipynb': It took me way longer than I'd like to admit to realize that the right and left beams could switch which one was strong and which one was weak. Explains some weird observations I found.

## Acknowledgements

I would like to thank the Centre for Satellite Data in Environmental Science (SENSE) CDT for funding my PhD research and my supervisors Steven Hancock (University of Edinburgh), Richard Essery (University of Edinburgh), Amy Neuenschwander (University of Texas at Austin), and Andrew Ross (University of Leeds) for providing guidance throughout my work. I would also like to thank Matthew Purslow, one of Steven's former PhD students who worked with ICESat-2 and canopy:ground reflectance ratios and helped me get the ball rolling at the start of my studies.