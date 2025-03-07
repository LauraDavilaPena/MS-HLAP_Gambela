import pandas as pd
import geopandas as gpd
from shapely import wkt  # Use if geometry is stored as WKT strings
#%% Convert json to Excel%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Load the GeoJSON file
gdf = gpd.read_file('tierkidi.geojson')

# Save to Excel
gdf.to_excel('tierkidi_output.xlsx', index=False, engine='openpyxl')

print("GeoJSON converted to Excel and saved as 'tierkidi_output.xlsx'")


#%% Convert Excel to json%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Read Excel file
df = pd.read_excel('tierkidi_output.xlsx')

# If geometry is stored as WKT string, you can convert it
if 'geometry' in df.columns and isinstance(df['geometry'].iloc[0], str):
    df['geometry'] = df['geometry'].apply(wkt.loads)

# Convert DataFrame to GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry='geometry')

# Save to GeoJSON
gdf.to_file('modified_file.geojson', driver='GeoJSON')

print("Excel file successfully converted to GeoJSON.")
