import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# read in data
shipdata = pd.read_csv('CLIWOC15.csv')
lat = shipdata.Lat3
lon = shipdata.Lon3
coord = np.column_stack((list(lon), list(lat)))
nation = shipdata.Nationality
utc = shipdata.UTC
year = shipdata.Year
month = shipdata.Month


# take out lon/lat nan and voyageFrom/To nan
utc = utc[~np.isnan(coord).any(axis=1)]
utc = utc.astype(np.float)

year = year[~np.isnan(coord).any(axis=1)]
year = year.astype(np.float)

month = month[~np.isnan(coord).any(axis=1)]
month = month.astype(np.float)

coord = coord[~np.isnan(coord).any(axis=1)]
coord = coord.astype(np.float)

nation = nation[~np.isnan(coord).any(axis=1)]

data = np.column_stack((coord, year, month, utc, nation))

# set X as the feature object
X = data[:, 0:4]
# normalize the data attributes
normalized_X = preprocessing.normalize(X)
# standardize the data attributes
standardized_X = preprocessing.scale(X)

# set y as the target variable
y = data[:, 5]

# run k-means
model = KNeighborsClassifier(metric='euclidean', weights='distance', n_neighbors=7)
model.fit(X, y)

expected = y
predicted = model.predict(X)

# output info
text_file = open("OutputKMeans.txt", "w")
text_file.write("Ocean Ship Logbooks 1750-1850 K-means implementation\n\n")
text_file.write(str(model))
text_file.write("\n\nFeature vector: latitude, longitude, UTC, year, month\n")
text_file.write("Target variable: Nationality\n\n")
text_file.write(metrics.classification_report(expected, predicted))
text_file.write("\nConfusion Matrix\n")
text_file.write(str(metrics.confusion_matrix(expected, predicted)))
text_file.close()


