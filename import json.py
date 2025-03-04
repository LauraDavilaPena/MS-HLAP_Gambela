import json
import matplotlib.pyplot as plt
import numpy as np

# Load the JSON data
with open('location_refcamps2.geojson', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Extract coordinates and camp names
coordinates = []
camps = []

for feature in data['features']:
    coords = feature['geometry']['coordinates']
    camp = feature['properties'].get('Camp', 'Unknown')
    coordinates.append((coords[0], coords[1]))
    camps.append(camp)

# Convert to numpy array for easier handling
coordinates = np.array(coordinates)

# Assign a unique color to each camp
unique_camps = list(set(camps))
camp_colors = {camp: plt.cm.get_cmap('tab10')(i) for i, camp in enumerate(unique_camps)}

# Plot
plt.figure(figsize=(12, 8))
for i, (x, y) in enumerate(coordinates):
    plt.scatter(x, y, color=camp_colors[camps[i]], label=camps[i] if camps[i] not in plt.gca().get_legend_handles_labels()[1] else "")

# Add legend and labels
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Locations Colored by Camp')
plt.legend(title='Camp', loc='upper right', bbox_to_anchor=(1.3, 1))

# Show the plot
plt.grid(True)
plt.show()
