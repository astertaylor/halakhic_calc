import pandas as pd
import numpy as np

citiesdat = pd.read_csv("data/worldcities.csv")
# Create a mapping from city names to coordinates

pops = citiesdat['population'].values
pops[np.isnan(pops)] = 0.0
inds = np.argsort(pops)[::-1]  # Sort indices by population in descending order
keep_inds = []
countries = []
for ind in inds:
    if countries.count(citiesdat['country'][ind]) < 25:
        keep_inds.append(ind)
        countries.append(citiesdat['country'][ind])

    if len(keep_inds) >= 1000:
        break

keep_inds.insert(0, citiesdat[citiesdat['city_ascii']=="Jerusalem"].index[0])
keep_inds.pop(430)

citiesdat.iloc[keep_inds].to_csv("data/cities_list.csv", index=False)

