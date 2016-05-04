Ocean Ship Logbooks 1750-1850 Data Mining Project
=================================================
1) Routes: Provides a visualization of all the logbook entries on a map. Utilized matplotlib and basemap modules

2) Clusters: Holds the dataset using NumPy and uses sklearn libraries to run the KNeighborsClassifier to cluster the entries

3) Classification: Utilizes NumPy and sklearn to implement Gaussian Naive Bayes to create a classification model



To run the program, each script like so when in the directory,
Make sure the data (CLIWOC15.csv) is in src directory:

        $ python Routes.py
        $ python Clusters.py
        $ python Classification.py

Outputs files according to the script used.

If the file is not present, it can be retrieved from the following Kaggle link:
https://www.kaggle.com/kaggle/climate-data-from-ocean-ships