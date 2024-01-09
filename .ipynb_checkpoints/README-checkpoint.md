# Snowstradamus

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