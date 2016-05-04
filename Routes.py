import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# read in data
shipdata = pd.read_csv('CLIWOC15.csv')
lat = shipdata.Lat3
lon = shipdata.Lon3
coord = np.column_stack((list(lon), list(lat)))
ship = shipdata.ShipName
nation = shipdata.Nationality
utc = shipdata.UTC
year = shipdata.Year
month = shipdata.Month

# take out lon/lat nan and voyageFrom/To nan
utc = utc[~np.isnan(coord).any(axis=1)]
ship = ship[~np.isnan(coord).any(axis=1)]
nation = nation[~np.isnan(coord).any(axis=1)]
year = year[~np.isnan(coord).any(axis=1)]
year = year.astype(np.int)
month = month[~np.isnan(coord).any(axis=1)]
coord = coord[~np.isnan(coord).any(axis=1)]

data = np.column_stack((coord, ship, year, month, utc, nation))

# sets up the base map
m = Basemap(projection='robin', lon_0=0, resolution='c', llcrnrlon=120, urcrnrlon=-30)
m.drawcoastlines()
m.drawcountries()
m.fillcontinents(color='grey')
m.drawmeridians(np.arange(0, 360, 30))
m.drawparallels(np.arange(-90, 90, 30))

nationList = np.matrix([['Spanish', 'yellow'], ['French', 'cyan'], ['Swedish', 'magenta'], ['Dutch', 'red'],
                        ['British', 'blue'], ['Danish', 'green'], ['Hamburg', 'grey']])

# plot data
for nation in nationList:
    temp = data[data[:, 6] == nation.item(0)]
    #temp = data[data[:, 3] < 1800]

    # sort time
    temp = temp[temp[:, 5].argsort()]

    # draw  the paths on the background
    x, y = m(temp[:, 0], temp[:, 1])
    m.plot(x, y, '.', color=nation.item(1), alpha=0.2, markersize=.5, label=nation.item(0))

plt.title('Nation Routes')
plt.savefig('Ship Routes.jpeg', dpi=600)
plt.show()
plt.close()
