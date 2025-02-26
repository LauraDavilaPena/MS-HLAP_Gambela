
from fastkml import kml
import geopandas as gpd
import numpy as np
from shapely.geometry import shape
from shapely.geometry import Point
from shapely.geometry import Polygon
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from haversine import haversine, Unit
import contextily as ctx
import matplotlib.patches as patches


work = False  
if work:
    with open(r'C:/Users/ld514/OneDrive - University of Kent/PROJECTS/CUAMM - Ethiopia/model python/data/Gambella.kml', 'rb') as file:
        kml_content = file.read()
else:
    with open(r'C:/Users/laura/OneDrive - University of Kent/PROJECTS/CUAMM - Ethiopia/model python/data/Gambella.kml', 'rb') as file:
        kml_content = file.read()

# Parse the KML
k = kml.KML()
k.from_string(kml_content)


# Extract polygons
polygons = []
for feature in list(k.features()):
    for placemark in list(feature.features()):
        if placemark.geometry.geom_type == 'Polygon':
            polygons.append(shape(placemark.geometry))

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(geometry=polygons)
gdf

# Save to a shapefile or any other format if needed
#gdf.to_file("polygons.shp")

# Save as CSV (with WKT for geometries)
# gdf.to_csv("polygons.csv", index=False)


def generate_points_within_polygon(polygon, n_points=100):
    minx, miny, maxx, maxy = polygon.bounds
    points = []
    while len(points) < n_points:
        p = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        if polygon.contains(p):
            points.append((p.x, p.y))
    return np.array(points)


def cluster_polygon(polygon, n_clusters):
    # Generate points within the polygon
    points = generate_points_within_polygon(polygon)
    
    # Handle cases with fewer points than the number of clusters
    if len(points) < n_clusters:
        print(f"Polygon with {len(points)} points cannot be clustered into {n_clusters} clusters.")
        return None, None

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(points)
    
    # Get the centroids of the clusters
    centroids = kmeans.cluster_centers_
    return points, centroids, kmeans.labels_


# Example usage
n_clusters = [5,3,4,3,1,3,1]  # Number of clusters (population zones) desired
results = []

for idx, row in gdf.iterrows():
    polygon = row['geometry']
    points, centroids, labels = cluster_polygon(polygon, n_clusters[idx])
    
    if centroids is not None:
        results.append({
            'polygon': polygon,
            'points': points,
            'centroids': centroids,
            'labels': labels
        })

results


def plot_clustering_results(polygon, points, centroids, labels, title):
    """
    Visualize clustering results with an optional title.
    
    Parameters:
        polygon: Shapely polygon object representing the boundary.
        points: Array of clustered points (NumPy array).
        centroids: Array of centroids (NumPy array).
        labels: Cluster labels for the points.
        title: Title for the plot.
    """
    plt.figure()
    plt.title(title)

    # Plot the polygon
    x, y = polygon.exterior.xy
    plt.plot(x, y, color='black', linestyle='-', linewidth=1, label='Polygon')
    
    # Plot the clustered points
    scatter = plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='viridis', label='Clustered Points')
    plt.colorbar(scatter, label='Cluster ID')
    
    # Plot centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100, label='Centroids')
    
    plt.legend()
    plt.show()


# Plot the results for the first polygon as an example
if results:
    for i, result in enumerate(results):
        camps_names = ['Nguenyyiel', 'Kule', 'Tierkidi', 'Jewi', 'Punydo II','Pinyudo', 'Okugo']  # Camp names
        
        # Use camp names for titles
        title = f"Population zones (clusters) for refugee camp {camps_names[i]}"
        
        # Plot the results for the current polygon
        plot_clustering_results(
            result['polygon'], 
            result['points'], 
            result['centroids'], 
            result['labels'],
            title=title
        )



# Initialize an empty list to hold all centroids
all_centroids = []

# Iterate over each entry in the results
for index in range(len(results)):
    # Extract the centroids for the current index
    centroids = results[index]['centroids']
    
    # Append the centroids to the list
    all_centroids.extend(centroids.tolist())  # Convert array to list and extend


population_zones = all_centroids
population_names = ['Nguenyyiel 1', 'Nguenyyiel 2', 'Nguenyyiel 3', 'Nguenyyiel 4', 'Nguenyyiel 5',
               'Kule 1', 'Kule 2', 'Kule 3', 
               'Tierkidi 1', 'Tierkidi 2', 'Tierkidi 3', 'Tierkidi 4', 
               'Jewi 1', 'Jewi 2', 'Jewi 3', 'Pinyudo-II 1',
                'Pinyudo 1', 'Pinyudo 2', 'Pinyudo 3',
                'Okugo 1']
gambella_hospital = np.array([[34.580385,8.242747699999999]])
all_zones = population_zones
all_zones.extend(gambella_hospital.tolist()) 
all_names = population_names
all_names.append('Gambella-Hospital')
"""
Note the order of the refugee camps is: Nguenyyiel, Kule, Tierkidi, Jewi, Pinyudo-II, Pinyudo, Okugo
"""



# Calculate the Haversine distance matrix (again, I used the Haversine distance for illustrative purposes, but probably CUAMM
# has real data on distances, based on the transportation network in Ethiopia)
def get_dist_matrix(all_zones, all_names):
    dist_matrix = []
    for i in range(len(all_zones)):
        row = []
        for j in range(len(all_zones)):
            distance = haversine(all_zones[i], all_zones[j], unit=Unit.KILOMETERS) 
            row.append(distance)
        dist_matrix.append(row)

    # Creating a DataFrame for the distance matrix
    dist_df = pd.DataFrame(dist_matrix, index=all_names, columns=all_names)
    return dist_matrix, dist_df

# Calculate the distance matrix with the above function and data
distance_matrix = get_dist_matrix(all_zones, all_names)[1]
distance_matrix


# Let see if I can visualize Gambella

# Convert to GeoDataFrame
geometry_popzones = [Point(xy) for xy in all_zones]
gdf_popzones = gpd.GeoDataFrame(geometry=geometry_popzones)

# Set the coordinate reference system (CRS) to WGS84 (EPSG:4326)
gdf_popzones.set_crs(epsg=4326, inplace=True)

# Convert the GeoDataFrame to the Web Mercator projection (EPSG:3857)
gdf_popzones = gdf_popzones.to_crs(epsg=3857)
gdf_popzones


# Create the plot
fig, ax = plt.subplots(figsize=(10, 10))

# Plot all points except the last one
gdf_popzones.iloc[:-1].plot(ax=ax, color='blue', markersize=2)

# Plot the last point with a custom "doctor's cross" marker
last_point = gdf_popzones.iloc[-1]

# Define the doctor's cross marker (red cross)
marker_cross = plt.Line2D((0,1),(0,0), color='red', lw=5) # Use Line2D to create a cross marker

ax.scatter(last_point.geometry.x, last_point.geometry.y, s=500, color='red', marker='P')


# Add basemap from contextily using OpenStreetMap
ctx.add_basemap(ax, crs=gdf_popzones.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)

# Set axis labels and title
ax.set_title('Gambella Region - Selected Points')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

plt.show()


#If I want to disregard Okugo:

# Exclude the one but last point
gdf_popzones_filtered = gdf_popzones.drop(gdf_popzones.index[-2])

# Create the plot
fig, ax = plt.subplots(figsize=(10, 10))

# Plot all points except the last one
gdf_popzones_filtered.iloc[:-1].plot(ax=ax, color='blue', markersize=2)

# Plot the last point with a custom "doctor's cross" marker
last_point = gdf_popzones_filtered.iloc[-1]

# Coordinates for the last point
x, y = last_point.geometry.x, last_point.geometry.y

# Create the doctor's cross as a combination of two rectangles
cross_size = 1000  # Adjust size of the cross here
cross_width = cross_size / 5  # Adjust width of the cross arms here

# Create horizontal and vertical bars for the cross
horizontal_bar = patches.Rectangle((x - cross_size / 2, y - cross_width / 2), cross_size, cross_width, color='red')
vertical_bar = patches.Rectangle((x - cross_width / 2, y - cross_size / 2), cross_width, cross_size, color='red')

# Add the bars to the plot
ax.add_patch(horizontal_bar)
ax.add_patch(vertical_bar)

# Add basemap from contextily using OpenStreetMap
ctx.add_basemap(ax, crs=gdf_popzones.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)

# Set axis labels and title
ax.set_title('Gambella Region - Selected Points (Okugo excluded)')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')

plt.show()

# %%

# Note the order of the refugee camps is:
# Nguenyyiel, Kule, Tierkidi, Jewi, Pinyudo-II,
# Pinyudo, Okugo (the one in all_names)
#
# The population of each refugee camp is:
d = np.array([112000, 52959, 72438, 67896, 11392, 51239, 13954])
n_clusters = [5,3,4,3,1,3,1] 
# Create the repeated vector
repeated_demand = np.repeat(d, n_clusters)
repeated_clusters = np.repeat(n_clusters, n_clusters)

# We uniformly distribute the demand of each refugee camp
# over each of the population zones within each refugee camp
demand_per_population_zone = repeated_demand/repeated_clusters

# Now, for each of the specialities, we calculate the
# demand per population zone
basic_care_demand = df = np.ceil(demand_per_population_zone)
fertility_rate = 0.035
maternal_demand = np.ceil(demand_per_population_zone*fertility_rate)
under18_rate = 0.66
pediatric_demand = np.ceil(demand_per_population_zone*under18_rate)
tachioma_rate = 0.172
eyecare_demand = np.ceil(demand_per_population_zone*tachioma_rate)

# Create the DataFrame
data = {
    'Basic Care': basic_care_demand,
    'Maternal Care': maternal_demand,
    'Pediatric Care': pediatric_demand,
    'Eye Care': eyecare_demand
}

demand = pd.DataFrame(data)

# Display the DataFrame
print(demand)

"""
In each HP we need to have:
- Basic care: between 10 and 20 supportive staff
- Maternal: between 1 and 2 midwives
- Pediatric: between 2 and 5 nurses
- Eye care: no more than 2 professionals

In each HC we need to have: 
- Basic care: between 20 and 30 supportive staff
- Maternal: between 2 and 5 midwives
- Pediatric: between 4 and 8 nurses
- Eye care: between 1 and 4 professionals

At the hospital we need to have:
- Basic care: more than 50 supportive staff
- Maternal: more than 5 midwives
- Pediatric: between 6 and 12 nurses
- Eye care: between 3 and 6 professionals
"""

lb_specialists_HPs = [10, 1, 2, 0] 
ub_specialists_HPs = [20, 2, 5, 2]
lb_specialists_HCs = [20, 2, 4, 1]
ub_specialists_HCs = [30, 5, 8, 4]
lb_specialists_Hosp = [50, 5, 6, 3]
ub_specialists_Hosp = [1000, 1000, 12, 6]


# Combine the lists into a 2D matrix
lb_specialists = np.array([lb_specialists_HPs, lb_specialists_HCs, lb_specialists_Hosp])
ub_specialists = np.array([ub_specialists_HPs, ub_specialists_HCs, ub_specialists_Hosp])

# The number of specialists to locate (per 'level of service') is s_l
s_l = [197, 14, 19, 12]

# Service rates for each professional (to define somehow 
# the capacity of the system)
# rates expressed in patients/day per professional
rates_specialists_HPs = [42, 28, 28, 42]
rates_specialists_HCs = [42, 10, 28, 14]
rates_specialists_Hosp = [42, 4, 28, 14]

rates_specialists = np.array([rates_specialists_HPs, rates_specialists_HCs, rates_specialists_Hosp])

facilities_to_locate = [4,2,1]
rates_referrals = np.array([[0,0.1,0], [0,0,0.3], [0,0,0]])


# Function to get camps within a specified distance

def within_distance(camps_coord, camps_names, max_distance):
    distance_matrix = get_dist_matrix(camps_coord, camps_names)[1]
    camps_within_distance = {}
    for i in camps_names:
        within_distance_aux = distance_matrix[i][distance_matrix[i] <= max_distance].index.tolist()
        within_distance_aux.remove(i)  # Remove the camp itself from the list
        camps_within_distance[i] = within_distance_aux
    # Create a mapping from names to indices
    name_to_index = {name: index + 1 for index, name in enumerate(camps_within_distance.keys())}
    # Create a new dictionary with indices
    indexed_dict = {name_to_index[key]: [name_to_index[neighbor] for neighbor in neighbors]
                for key, neighbors in camps_within_distance.items()}
    return camps_within_distance, indexed_dict

zones_within_d1  = within_distance(all_zones, all_names, max_distance=50)[0]
facilities_within_d2  = within_distance(all_zones, all_names, max_distance=100)[0]
indexed_dict_zones = within_distance(all_zones, all_names, max_distance=50)[1]
indexed_dict_facilities = within_distance(all_zones, all_names, max_distance=100)[1]

d_dict = {index + 1: value for index, value in enumerate(d)}



