# parameters_dynamic2.py

import geopandas as gdp
import numpy as np
import pandas as pd
import itertools
import sys, trace
import matplotlib.pyplot as plt
from scenarios import scenarios  # Import scenario configurations
from scipy.spatial.distance import cdist
import os

# Select Scenario
# scenario_name = "baseline2"  # Change this to switch scenarios
# Define a default scenario, but allow overriding dynamically

try:
    scenario_name
except NameError:
    scenario_name = "baseline2"  # Default scenario

params = scenarios[scenario_name]  # Load selected scenario parameters



# Load the GeoJSON file
# location_nodes = gdp.read_file("location_nodes.geojson")
#location_nodes = gdp.read_file("location_refcamps.geojson")
location_nodes =  gdp.read_file(params["location_file"])
location_nodes

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
 

camps = set(location_nodes["Camp"].unique())
camps
camp_demand_labels = demand_points_gdf.groupby("Camp")["label"].apply(set).to_dict()
camp_demand_labels
camp_candidate_location_labels = hfs_gdf.groupby("Camp")["label"].apply(set).to_dict()
camp_candidate_location_labels 
 
"""
# This is _just_ to have insights on which values would be adequate for t'max and t''max 

from shapely.geometry import Point

# Initialize dictionaries to store results
avg_dist_demand_to_hp = {}
avg_dist_demand_to_hc = {}
min_intercamp_distance = {} # For distances between different camps
max_withincamp_distance = {} # For maximum distance within the same camp

# Compute distances for each camp
for camp in camps:
    # Filter locations by camp
    camp_demand_points = demand_points_gdf[demand_points_gdf["Camp"] == camp]
    camp_hps = hps_gdf[hps_gdf["Camp"] == camp]
    camp_hcs = hcs_gdf[hcs_gdf["Camp"] == camp]

    # Compute distances between demand points and HPs
    if not camp_hps.empty:
        avg_distances_per_demand_point_to_hp = camp_demand_points.to_crs(epsg=3857).geometry.apply(lambda dp: camp_hps.to_crs(epsg=3857).geometry.distance(dp).mean())
        avg_dist_demand_to_hp[camp] = avg_distances_per_demand_point_to_hp.mean()
    else:
        avg_dist_demand_to_hp[camp] = None  # No HPs in this camp

    # Compute distances between demand points and HCs
    if not camp_hcs.empty:
        avg_distances_per_demand_point_to_hc = camp_demand_points.to_crs(epsg=3857).geometry.apply(lambda dp: camp_hcs.to_crs(epsg=3857).geometry.distance(dp).mean())
        avg_dist_demand_to_hc[camp] = avg_distances_per_demand_point_to_hc.mean()
    else:
        avg_dist_demand_to_hc[camp] = None  # No HCs in this camp

    # Now calculate maximum within-camp distance (distance between any two locations within the same camp)
    locations_camp = location_nodes[location_nodes["Camp"] == camp]
    
    # Reproject to EPSG:3857 (meters) and compute the maximum pairwise distance within the camp
    max_within_distance = locations_camp.to_crs(epsg=3857).geometry.apply(
        lambda loc: locations_camp.to_crs(epsg=3857).geometry.distance(loc).max()
    ).max()
    
    # Store the maximum distance within the camp
    max_withincamp_distance[camp] = max_within_distance

# Compute minimum distance between any location in different camps
for camp1 in camps:
    for camp2 in camps:
        if camp1 != camp2:
            locations_camp1 = location_nodes[location_nodes["Camp"] == camp1]
            locations_camp2 = location_nodes[location_nodes["Camp"] == camp2]
            min_distance = locations_camp1.to_crs(epsg=3857).geometry.apply(lambda loc: locations_camp2.to_crs(epsg=3857).geometry.distance(loc).min()).min()
            min_intercamp_distance[(camp1, camp2)] = min_distance

# Now find the overall minimum distance across all inter-camp pairs
overall_min_distance = min(min_intercamp_distance.values())

# Find the overall maximum within-camp distance
overall_max_withincamp_distance = max(max_withincamp_distance.values())


# # Convert results to DataFrames for better visualization
# df_avg_dist_hp = pd.DataFrame(list(avg_dist_demand_to_hp.items()), columns=["Camp", "Avg_Dist_Demand_to_HP"])
# df_avg_dist_hc = pd.DataFrame(list(avg_dist_demand_to_hc.items()), columns=["Camp", "Avg_Dist_Demand_to_HC"])
# df_min_intercamp = pd.DataFrame(list(overall_min_distance.items()), columns=["Camp_Pair", "Min_Distance"])
# df_max_withincamp = pd.DataFrame(list(overall_max_withincamp_distance.items()), columns=["Camp", "Max_Distance"])

# Display results
print(avg_dist_demand_to_hp)
print(avg_dist_demand_to_hc)
print(min_intercamp_distance)
print(overall_min_distance)
print(max_withincamp_distance)
print(overall_max_withincamp_distance)
"""


def compute_distance_matrix(demand_points_gdf, hfs_gdf):
    """
    Compute the distance matrix between demand points and candidate health facility locations.

    Parameters:
    - demand_points_gdf: GeoDataFrame containing demand points with 'geometry'.
    - hfs_gdf: GeoDataFrame containing candidate health facility locations with 'geometry'.

    Returns:
    - distance_matrix: 2D NumPy array of distances (rows: demand points, columns: health facilities).
    """
    # Extract coordinates as NumPy arrays directly from the geometry column
    demand_coords = np.array(demand_points_gdf.geometry.apply(lambda point: (point.x, point.y)).tolist())
    hfs_coords = np.array(hfs_gdf.geometry.apply(lambda point: (point.x, point.y)).tolist())
    
    # Compute the distance matrix using cdist with Euclidean metric
    distance_matrix = cdist(demand_coords, hfs_coords, metric='euclidean')
    
    # Create a labeled DataFrame
    distance_df = pd.DataFrame(distance_matrix, index=dps, columns=hfs)

    return distance_df

# Example usage:
distance_df = compute_distance_matrix(demand_points_gdf, hfs_gdf)

# To save the above matrix into an Excel file to subsequently read
# distance_df.to_excel('distance_matrix_refcamps.xlsx', sheet_name='DistanceMatrixRefCamps')#, float_format="%.2f")

# Distance matrix
# distance_matrix = pd.read_excel('distance_matrix_ij.xlsx', index_col=0)
# distance_matrix = pd.read_excel('distance_matrix_refcamps.xlsx', index_col=0)



import numpy as np

def compute_distance_matrix_meters(demand_points_gdf, hfs_gdf, crs_epsg=3857):
    """
    Compute the distance matrix between demand points and candidate health facility locations.

    Parameters:
    - demand_points_gdf: GeoDataFrame containing the demand points with geometry (usually point geometries).
    - hfs_gdf: GeoDataFrame containing the candidate health facility locations with geometry.
    - crs_epsg: The EPSG code to which the geometries will be reprojected. Default is 3857 (Web Mercator).

    Returns:
    - distance_df: A pandas DataFrame where the rows are demand points, the columns are health facilities,
                   and the values are the distances between them.
    """
    # Reproject both demand points and health facilities to the target CRS (e.g., EPSG:3857 for meters)
    demand_points_gdf = demand_points_gdf.to_crs(epsg=crs_epsg)
    hfs_gdf = hfs_gdf.to_crs(epsg=crs_epsg)

    # Initialize an empty distance matrix with dimensions (num_demand_points x num_health_facilities)
    num_demand_points = len(demand_points_gdf)
    num_health_facilities = len(hfs_gdf)
    distance_matrix = np.zeros((num_demand_points, num_health_facilities))

    # Compute distances
    for i, demand_point in enumerate(demand_points_gdf.geometry):
        for j, hf_location in enumerate(hfs_gdf.geometry):
            distance_matrix[i, j] = demand_point.distance(hf_location)
    
    # Create a DataFrame with labeled indices and columns
    distance_df = pd.DataFrame(distance_matrix, index=dps, columns=hfs) 


    return distance_df


# Example usage:
distance_df = compute_distance_matrix_meters(demand_points_gdf, hfs_gdf)
distance_df

# To save the above matrix into an Excel file to subsequently read
# distance_df.to_excel('distance_matrix_refcamps_meters.xlsx', sheet_name='DistanceMatrixRefCamps')#, float_format="%.2f")
# distance_df.to_excel('distance_matrix_tierkidi.xlsx', sheet_name='DistanceMatrixTierkidi')#, float_format="%.2f")

# Distance matrix
# distance_matrix = pd.read_excel('distance_matrix_ij.xlsx', index_col=0)
# distance_matrix = pd.read_excel('distance_matrix_refcamps_meters.xlsx', index_col=0)
distance_matrix = pd.read_excel(params["distance_matrix"], index_col=0)

distance_matrix




######################################

# Health services and workers
services = ['basic','maternal1','maternal2']
health_workers = ['doctor','nurse','midwife']
levels = ['hp', 'hc']



# Assign scenario parameters
HFs_to_locate = params["HFs_to_locate"]
t1max = params["t1max"]
t2max = params["t2max"]
workers_to_allocate = params["workers_to_allocate"]
working_hours = params["working_hours"]
service_time = params["service_time"]

# Lower bound workers per HF type
lb_workers_df = pd.DataFrame(params["lb_workers"], index=health_workers)
lb_workers = {(health_workers[p], levels[l]): lb_workers_df.iloc[p, l] 
      for p, l in itertools.product(range(len(health_workers)), range(len(levels)))}

# Upper bound workers per HF type
ub_workers_df = pd.DataFrame(params["ub_workers"], index=health_workers)
ub_workers = {(health_workers[p], levels[l]): ub_workers_df.iloc[p, l] 
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
