# parameters_dynamic.py

import geopandas as gdp
import numpy as np
import pandas as pd
import itertools
import sys, trace
import matplotlib.pyplot as plt
from scenarios import scenarios  # Import scenario configurations


# Select Scenario
scenario_name = "baseline"  # Change this to switch scenarios
params = scenarios[scenario_name]  # Load selected scenario parameters


# Load the GeoJSON file
location_nodes = gdp.read_file("location_nodes.geojson")

# Add x and y coordinates
location_nodes.loc[:, 'x'] = location_nodes.geometry.x
location_nodes.loc[:, 'y'] = location_nodes.geometry.y

# Sort 
location_nodes = location_nodes.sort_values(by=['y', 'x']).reset_index(drop=True)

# Label sorted demand points
demand_points_gdf = location_nodes.loc[location_nodes.type_f == "demand_point"].copy()
demand_points_gdf['label'] = ['i' + str(i + 1) for i in range(len(demand_points_gdf))]

# Save demand point labels to a Numpy Array
dps = demand_points_gdf['label'].to_numpy()

# Subset location types
hps_gdf = location_nodes[location_nodes.type_f == "HP"]
hcs_gdf = location_nodes[location_nodes.type_f == "HC"]
hfs_gdf = location_nodes[(location_nodes.type_f == "HC") | (location_nodes.type_f == "HP")].drop_duplicates(subset='geometry').reset_index(drop=False)

# Label candidate locations
hfs_gdf['label'] = ['j' + str(j + 1) for j in range(len(hfs_gdf))]

# Save location labels
hfs = hfs_gdf['label'].to_numpy()
hps = hfs_gdf[hfs_gdf['geometry'].isin(hps_gdf['geometry'])]['label'].to_numpy()
hcs = hfs_gdf[hfs_gdf['geometry'].isin(hcs_gdf['geometry'])]['label'].to_numpy()

# Distance matrix
distance_matrix = pd.read_excel('distance_matrix_ij.xlsx', index_col=0)

# Health services and workers
services = ['basic','maternal1','maternal2']
health_workers = ['doctor','nurse','midwife']
levels = ['hp', 'hc']



# Assign scenario parameters
HFs_to_locate = params["HFs_to_locate"]
t1max = params["t1max"]
workers_to_allocate = params["workers_to_allocate"]
working_hours = params["working_hours"]
service_time = params["service_time"]

# Lower bound workers per HF type
lb_workers_df = pd.DataFrame(params["lb_workers"], index=health_workers)
lb_workers = {(health_workers[p], levels[l]): lb_workers_df.iloc[p, l] 
      for p, l in itertools.product(range(len(health_workers)), range(len(levels)))}

# Where can each service be provided?
services_at_HFs_df = pd.DataFrame(params["services_at_HFs"], index=services)
a_HF = {(services[s], levels[l]): services_at_HFs_df.iloc[s, l] 
      for s, l in itertools.product(range(len(services)), range(len(levels)))}

# Which health worker can deliver each service?
services_per_worker_df = pd.DataFrame(params["services_per_worker"], index=health_workers)
a_W = {(health_workers[p], services[s]): services_per_worker_df.iloc[p, s] 
      for p, s in itertools.product(range(len(health_workers)), range(len(services)))}

# Demand rates
total_population = {(key): params["total_population"] for key in dps}

# Opening hours
demand_rate_opening_hours_df = pd.DataFrame([params["demand_rate_opening_hours"]] * len(dps), index=dps, columns=services)
dr_oh = {(dps[i], services[s]): demand_rate_opening_hours_df.iloc[i, s] 
      for i, s in itertools.product(range(len(dps)), range(len(services)))}

dd_oh = {(key): int(round(total_population[i] * dr_oh[key])) for i in dps for key in dr_oh}

# Closing hours
demand_rate_closing_hours_df = pd.DataFrame([params["demand_rate_closing_hours"]] * len(dps), index=dps, columns=services)
dr_ch = {(dps[i], services[s]): demand_rate_closing_hours_df.iloc[i, s] 
      for i, s in itertools.product(range(len(dps)), range(len(services)))}

dd_ch = {(key): int(round(total_population[i] * dr_ch[key])) for i in dps for key in dr_ch}
