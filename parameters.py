# %%
# (additional) indices and sets
import geopandas as gdp
import numpy as np
import pandas as pd
import itertools
import sys, trace
import matplotlib.pyplot as plt

# Load the GeoJSON file
location_nodes = gdp.read_file("location_nodes.geojson")


# Add x and y coordinates as new columns
location_nodes.loc[:, 'x'] = location_nodes.geometry.x
location_nodes.loc[:, 'y'] = location_nodes.geometry.y

# Sort 
location_nodes = location_nodes.sort_values(by=['y', 'x']).reset_index(drop=True)

# Label sorted demand points
demand_points_gdf = location_nodes.loc[location_nodes.type_f == "demand_point"].copy()
demand_points_gdf['label'] = ['i' + str(i + 1) for i in range(len(demand_points_gdf))]


# # Plot demand points and location nodes
# plt.figure(figsize=(10, 10))
# plt.scatter(location_nodes['x'], location_nodes['y'], c='blue', label='Location Nodes')
# plt.scatter(demand_points_gdf['x'], demand_points_gdf['y'], c='red', label='Demand Points')

# # Annotate demand points with labels
# for idx, row in demand_points_gdf.iterrows():
#     plt.annotate(row['label'], (row['x'], row['y']), textcoords="offset points", xytext=(0,10), ha='center')

# plt.xlabel('Longitude')
# plt.ylabel('Latitude')
# plt.legend()
# plt.title('Demand Points and Location Nodes')
# plt.show()

# Save demand point labels to a Numpy Array
dps = demand_points_gdf['label'].to_numpy()

# Take subsets for HPs, HCs and HFs (comprising both HPs and HCs)
hps_gdf = location_nodes[location_nodes.type_f == "HP"]
hcs_gdf = location_nodes[location_nodes.type_f == "HC"]
hfs_gdf = location_nodes[(location_nodes.type_f == "HC") | (location_nodes.type_f == "HP")]

# Remove duplicates in HFs (those candidate locations that serve both for HPs and HCs should only appear once)
hfs_gdf = hfs_gdf.drop_duplicates(subset='geometry').reset_index(drop=False)

# Label candidate locations
hfs_gdf['label'] = ['j' + str(j + 1) for j in range(len(hfs_gdf))]

# Save candidate location names for HFs to a Numpy Array
hfs = hfs_gdf['label'].to_numpy()

# Save candidate location names for HPs and HCs to a Numpy Array
hps = hfs_gdf[hfs_gdf['geometry'].isin(hps_gdf['geometry'])]['label'].to_numpy()
hcs = hfs_gdf[hfs_gdf['geometry'].isin(hcs_gdf['geometry'])]['label'].to_numpy()


# %%
# Services
services = ['basic','maternal1','maternal2']

# Health workers
health_workers = ['doctor','nurse','midwife']

# Types/Levels of HFs
levels = ['hp', 'hc']

# HFs to locate (at most)
HFs_to_locate = [4,2]

# Distance matrix
distance_matrix = pd.read_excel('distance_matrix_ij.xlsx', index_col=0)
"""
E.g., to access elements:
distance_matrix_df.loc['dp3','hc1']
"""

# Number of health workers of each type to allocate
workers_to_allocate = [50, 90, 90]

# Lower bounds on the number of workers per HF-type
lb_workers = np.array([
    [0,1], # doctor
    [1,2], # nurse
    [1,2], # midwife
])

lb_workers_df = pd.DataFrame(lb_workers, index=health_workers)

lb_workers = {(health_workers[p], levels[l]): lb_workers_df.iloc[p, l] 
      for p, l in itertools.product(range(len(health_workers)), range(len(levels)))}


# Where can each service be provided?
services_at_HFs = np.array([
    [1,0], # basic
    [1,1], # maternal1
    [0,1], # maternal2
])

services_at_HFs_df = pd.DataFrame(services_at_HFs, index=services)

a_HF = {(services[s], levels[l]): services_at_HFs_df.iloc[s, l] 
      for s, l in itertools.product(range(len(services)), range(len(levels)))}


# Which health worker can deliver each service?
services_per_worker = np.array([
    [1,0,1], # doctor
    [1,1,0], # nurse
    [0,1,1], # midwife
])

services_per_worker_df = pd.DataFrame(services_per_worker, index=health_workers)

a_W = {(health_workers[p],services[s]): services_per_worker_df.iloc[p, s] 
      for p, s in itertools.product(range(len(health_workers)), range(len(services)))}


# Maximum coverage distance for first assignment
t1max = 20

# Service time
service_time = [0.5, 1, 2]

# Working hours per day per health worker type
working_hours = [7, 8, 8]
  


########### Included on Feb 19 for the new model (test.tex) ###########
total_population = {(key): 10000 for key in dps}

# Demand rate per types of services during opening hours (oh)
demand_rate_opening_hours = np.array([
    [0.001] * len(dps), # basic
    [0.002] * len(dps), # maternal1
    [0.003] * len(dps), # maternal2
])

demand_rate_opening_hours_df = pd.DataFrame(demand_rate_opening_hours, index=services)
demand_rate_opening_hours_df = demand_rate_opening_hours_df.T

dr_oh = {(dps[i], services[s]): demand_rate_opening_hours_df.iloc[i, s] 
      for i, s in itertools.product(range(len(dps)), range(len(services)))}

dd_oh = {(key): int(round(total_population[i] * dr_oh[key])) for i in dps for key in dr_oh} # I needed to use round instead of np.floor to get my desired results


# Demand rate per types of services outside opening hours (closing hours ---> ch)
demand_rate_closing_hours = np.array([
    [0.001] * len(dps), # basic
    [0.002] * len(dps), # maternal1
    [0.003] * len(dps), # maternal2
])

demand_rate_closing_hours_df = pd.DataFrame(demand_rate_closing_hours, index=services)
demand_rate_closing_hours_df = demand_rate_closing_hours_df.T

dr_ch = {(dps[i], services[s]): demand_rate_closing_hours_df.iloc[i, s] 
      for i, s in itertools.product(range(len(dps)), range(len(services)))}

dd_ch = {(key): int(round(total_population[i] * dr_ch[key])) for i in dps for key in dr_ch} # I needed to use round instead of np.floor to get my desired results

