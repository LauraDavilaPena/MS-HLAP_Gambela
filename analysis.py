
import matplotlib.pyplot as plt
import contextily as ctx
import pyomo.environ as pyo
import pandas as pd

from matplotlib.patches import Circle


# Plot the solution

def plot_solution(model, demand_points_gdf, hfs_gdf):
    """
    Visualise the solution of the Pyomo model without location names and without the surrounding box.
    
    Parameters:
        model: Solved Pyomo model
        demand_points_gdf: GeoDataFrame with demand points (geometry and labels)
        hfs_gdf: GeoDataFrame with potential locations for health posts (HP) and health centres (HC) (geometry and labels)
    """

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot demand points (without names)
    demand_points_gdf.plot(ax=ax, color='blue', markersize=50)

    # Plot HPs (triangles) and HCs (squares) without labels
    hps_gdf = hfs_gdf[hfs_gdf['label'].isin(model.J_HP)]
    hcs_gdf = hfs_gdf[hfs_gdf['label'].isin(model.J_HC)]
    hps_gdf.plot(ax=ax, color='green', marker='^', markersize=80)
    hcs_gdf.plot(ax=ax, color='orange', marker='s', markersize=80)

    # Plot assignments with arrows
    for i in model.I:
        for j in model.J:
            if model.x1[i, j].value > 0 or model.x2[i, j].value > 0:
                # Get coordinates for the demand point and facility
                dp_coords = demand_points_gdf.loc[demand_points_gdf['label'] == i, 'geometry'].values[0]
                hf_coords = hfs_gdf.loc[hfs_gdf['label'] == j, 'geometry'].values[0]

                # Determine arrow colour
                if model.x1[i, j].value > 0 and model.x2[i, j].value > 0:
                    arrow_color = 'black'
                    linewidth = 2
                elif model.x1[i, j].value > 0:
                    arrow_color = 'yellow'
                    linewidth = 1
                elif model.x2[i, j].value > 0:
                    arrow_color = 'red'
                    linewidth = 1

                ax.annotate(
                    '', xy=(hf_coords.x, hf_coords.y), xytext=(dp_coords.x, dp_coords.y),
                    arrowprops=dict(arrowstyle='->', color=arrow_color, lw=linewidth),
                    zorder=1
                )

    # Add information text for each facility (demand and workers), without the facility name
    for j in model.J:
        assigned_demand_points = sum(model.x1[i, j].value > 0 or model.x2[i, j].value > 0 for i in model.I)
        workers = {p: model.w[j, p].value for p in model.P}
        status = "Open" if sum(model.y[j, l].value for l in model.L) > 0 else "Closed"
        hf_coords = hfs_gdf.loc[hfs_gdf['label'] == j, 'geometry'].values[0]
        ax.text(
            hf_coords.x + 0.01, hf_coords.y,
            f"{status}\nDemand: {assigned_demand_points}\nWorkers: {workers}",
            fontsize=8, color='black'
        )

    # Add information text for each demand point (flows), without the demand point name
    for i in model.I:
        f1_sum = sum(model.f1[i, j, s].value for j in model.J for s in model.S)
        f2_sum = sum(model.f2[i, j, s].value for j in model.J for s in model.S)
        total_demand = sum(model.d1[i, s] + model.d2[i, s] for s in model.S)
        dp_coords = demand_points_gdf.loc[demand_points_gdf['label'] == i, 'geometry'].values[0]
        ax.text(
            dp_coords.x + 0.01, dp_coords.y,
            f"f': {f1_sum}\nf'': {f2_sum}\nTotal: {total_demand}",
            fontsize=8, color='blue'
        )

    # Remove the axes (box) for a cleaner look
    ax.set_axis_off()

    plt.title("Model Solution: Demand and Facility Assignments")
    plt.show()



# Plot the solution with a map background (only if our demand points and facilities have real-world coordinates; otherwise, use the code above)

def plot_solution_with_map(model, demand_points_gdf, hfs_gdf, show_arrows = False, show_HF_text = False):
    """
    Visualize the solution of the Pyomo model on a real-world map using Contextily

    Parameters:
        model: Solved Pyomo model
        demand_points_gdf: GeoDataFrame with demand points (geometry and labels)
        hfs_gdf: GeoDataFrame with potential locations for health posts (HP) and health centres (HC) (geometry and labels)
    """

    # Convert to Web Mercator projection for map compatibility
    demand_points_gdf = demand_points_gdf.to_crs(epsg=3857)
    hfs_gdf = hfs_gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot demand points (without names)
    demand_points_gdf.plot(ax=ax, color='red', markersize=50)

    # Plot HPs (blue triangles) and HCs (black squares) without labels
    hps_gdf = hfs_gdf[hfs_gdf['label'].isin(model.J_HP)]
    hcs_gdf = hfs_gdf[hfs_gdf['label'].isin(model.J_HC)]

    hcs_gdf.plot(ax=ax, color='black', marker='s', markersize=80)
    hps_gdf.plot(ax=ax, color='blue', marker='^', markersize=80)

    
    # Plot assignments with arrows
    if show_arrows:
        for i in model.I:
            for j in model.J:
                if model.x1[i, j].value > 0 or model.x2[i, j].value > 0:
                    # Get coordinates for the demand point and facility
                    dp_coords = demand_points_gdf.loc[demand_points_gdf['label'] == i, 'geometry'].values[0]
                    hf_coords = hfs_gdf.loc[hfs_gdf['label'] == j, 'geometry'].values[0]

                    # Determine arrow colour and width
                    if model.x1[i, j].value > 0 and model.x2[i, j].value > 0:
                        arrow_color = 'black'
                        linewidth = 2
                    elif model.x1[i, j].value > 0:
                        arrow_color = 'green'
                        linewidth = 1
                    elif model.x2[i, j].value > 0:
                        arrow_color = 'orange'
                        linewidth = 1

                    ax.annotate(
                        '', xy=(hf_coords.x, hf_coords.y), xytext=(dp_coords.x, dp_coords.y),
                        arrowprops=dict(arrowstyle='->', color=arrow_color, lw=linewidth),
                        zorder=1
                    )
    
    
    # # Add information text for each facility (demand and workers), without the facility name
    if show_HF_text: 
        for j in model.J:
            assigned_demand_points = sum(model.x1[i, j].value > 0 or model.x2[i, j].value > 0 for i in model.I)
            workers = {p: model.w[j, p].value for p in model.P}
            status = "Open" if sum(model.y[j, l].value for l in model.L) > 0 else "Closed"
            hf_coords = hfs_gdf.loc[hfs_gdf['label'] == j, 'geometry'].values[0]
            ax.text(
                hf_coords.x + 0.1, hf_coords.y,
                f"{status}\nDemand: {assigned_demand_points}\nWorkers: {workers}",
                fontsize=8, color='black'
            )

        # Add information text for each demand point (flows), without the demand point name
        for i in model.I:
            f1_sum = sum(model.f1[i, j, s].value for j in model.J for s in model.S)
            f2_sum = sum(model.f2[i, j, s].value for j in model.J for s in model.S)
            total_demand = sum(model.d1[i, s] + model.d2[i, s] for s in model.S)
            dp_coords = demand_points_gdf.loc[demand_points_gdf['label'] == i, 'geometry'].values[0]
            ax.text(
                dp_coords.x + 0.1, dp_coords.y,
                f"f': {f1_sum}\nf'': {f2_sum}\nTotal: {total_demand}",
                fontsize=8, color='blue'
            )


    # Add a basemap
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    # Remove the axes (box) for a cleaner look
    ax.set_axis_off()

    plt.title(f"Demand points and candidate locations for health facilites")

    
    #plt.savefig("terkidi_OSM.png", format='png', dpi=300, bbox_inches='tight')

    plt.show()



# Plot the solution with a map background and a circle of radius cov_radius around HPs

def plot_solution_with_covering_radius(model, demand_points_gdf, hfs_gdf, cov_radius):
    """
    Visualize the solution of the Pyomo model on a real-world map using Contextily.

    Parameters:
        model: Solved Pyomo model
        demand_points_gdf: GeoDataFrame with demand points (geometry and labels)
        hfs_gdf: GeoDataFrame with potential locations for health posts (HP) and health centres (HC) (geometry and labels)
        cov_radius: the covering radius for the first assignment (t1max)
    """

    # Convert to Web Mercator projection for map compatibility
    demand_points_gdf = demand_points_gdf.to_crs(epsg=3857)
    hfs_gdf = hfs_gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot demand points (without names)
    demand_points_gdf.plot(ax=ax, color='red', markersize=50)

    # Plot open HPs (blue triangles) and HCs (black squares) without labels
    for j in model.J:
        for l in model.L:
            if model.y[j, l].value > 0:  # Only plot open facilities
                hf_coords = hfs_gdf.loc[hfs_gdf['label'] == j, 'geometry'].values[0]
                
                if l == 'hp':  # Plot a circle around HP with radius t1max
                    # Create a circle patch with light blue fill and no edge color
                    circle = Circle(
                        (hf_coords.x, hf_coords.y), radius=cov_radius, color='lightblue', alpha=0.3
                    )
                    ax.add_patch(circle)

                if l == 'hp':
                    ax.plot(hf_coords.x, hf_coords.y, marker='^', color='blue', markersize=8)
                elif l == 'hc':
                    ax.plot(hf_coords.x, hf_coords.y, marker='s', color='black', markersize=8)

    # Add a basemap
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    # Remove the axes (box) for a cleaner look
    ax.set_axis_off()

    plt.title(f"Open facilities and covering radius of {cov_radius}m")
    
    #plt.savefig("terkidi_add_1HC_add_1doctor_change_t1max_add_4nurses4midwives_upper_bounds_OSM_solution.png", format='png', dpi=300, bbox_inches='tight')

    plt.show()



# Plot the solution with a map background and zoom in around a specified open facility

def plot_solution_with_map_zoom(model, demand_points_gdf, hfs_gdf, zoom_factor = 0.1, buffer_size = 500, open_facility_label = None):
    """
    Visualize the solution of the Pyomo model on a real-world map using Contextily,
    zooming in around a selected open facility.

    Parameters:
        model: Solved Pyomo model
        demand_points_gdf: GeoDataFrame with demand points (geometry and labels)
        hfs_gdf: GeoDataFrame with potential locations for health posts (HP) and health centres (HC) (geometry and labels)
        zoom_factor: Proportion of extra space to add around the open facility (default 10%)
        open_facility_label: The label of the open facility to zoom in on. If None, zoom in on all open facilities.
    """

    # Convert to Web Mercator projection for map compatibility
    demand_points_gdf = demand_points_gdf.to_crs(epsg=3857)
    hfs_gdf = hfs_gdf.to_crs(epsg=3857)

    # Get open facilities
    open_hfs = hfs_gdf[[sum(model.y[j, l].value for l in model.L) > 0 for j in hfs_gdf['label']]]

    if open_facility_label:
        if isinstance(open_facility_label, str):
            open_facility_label = [open_facility_label]  # Ensure it's a list
        open_facility = open_hfs[open_hfs['label'].isin(open_facility_label)]
    
        # Select the specific open facility (or facilities) to zoom in on
        #open_hfs[open_hfs['label'] == open_facility_label]
    else:
        # Zoom in on all open facilities
        open_facility = open_hfs

    # Compute zoom-in limits based on the selected facility or all open facilities
    if not open_facility.empty:
        min_x, min_y, max_x, max_y = open_facility.total_bounds  # Bounding box of selected/open facilities

        # If it's a single point, add a small buffer for zooming
        if  len(open_facility) == 1:
            min_x -= buffer_size
            max_x += buffer_size
            min_y -= buffer_size
            max_y += buffer_size

        x_margin = (max_x - min_x) * zoom_factor
        y_margin = (max_y - min_y) * zoom_factor

    else:
        # Default to full view if no open facilities
        min_x, min_y, max_x, max_y = hfs_gdf.total_bounds
        x_margin = (max_x - min_x) * zoom_factor
        y_margin = (max_y - min_y) * zoom_factor


    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot demand points (without names)
    demand_points_gdf.plot(ax=ax, color='red', markersize=50)

    # Plot HPs (blue triangles) and HCs (black squares) without labels
    hps_gdf = hfs_gdf[hfs_gdf['label'].isin(model.J_HP)]
    hcs_gdf = hfs_gdf[hfs_gdf['label'].isin(model.J_HC)]
    hps_gdf.plot(ax=ax, color='blue', marker='^', markersize=80)
    hcs_gdf.plot(ax=ax, color='black', marker='s', markersize=80)

    # Plot assignments with arrows
    for i in model.I:
        for j in model.J:
            if model.x1[i, j].value > 0 or model.x2[i, j].value > 0:
                # Get coordinates for the demand point and facility
                dp_coords = demand_points_gdf.loc[demand_points_gdf['label'] == i, 'geometry'].values[0]
                hf_coords = hfs_gdf.loc[hfs_gdf['label'] == j, 'geometry'].values[0]

                # Determine arrow colour and width
                if model.x1[i, j].value > 0 and model.x2[i, j].value > 0:
                    arrow_color = 'black'
                    linewidth = 2
                elif model.x1[i, j].value > 0:
                    arrow_color = 'green'
                    linewidth = 1
                elif model.x2[i, j].value > 0:
                    arrow_color = 'orange'
                    linewidth = 1

                ax.annotate(
                    '', xy=(hf_coords.x, hf_coords.y), xytext=(dp_coords.x, dp_coords.y),
                    arrowprops=dict(arrowstyle='->', color=arrow_color, lw=linewidth),
                    zorder=1
                )


    # Add information text for each facility (demand and workers), without the facility name
    for j in model.J:
        assigned_demand_points = sum(model.x1[i, j].value > 0 or model.x2[i, j].value > 0 for i in model.I)
        workers = {p: model.w[j, p].value for p in model.P}
        status = "Open" if sum(model.y[j, l].value for l in model.L) > 0 else "Closed"
        hf_coords = hfs_gdf.loc[hfs_gdf['label'] == j, 'geometry'].values[0]
        ax.text(
            hf_coords.x + 100, hf_coords.y - 300,
            f"{status}\n#assigned_demand_points: {assigned_demand_points}\nWorkers: {workers}",
            fontsize=8, color='black',
            clip_on=True  # Ensures text is clipped to the zoomed-in area
        )

    # Add information text for each demand point (flows), without the demand point name
    for i in model.I:
        f1_sum = sum(model.f1[i, j, s].value for j in model.J for s in model.S)
        f2_sum = sum(model.f2[i, j, s].value for j in model.J for s in model.S)
        total_demand = sum(model.d1[i, s] + model.d2[i, s] for s in model.S)
        dp_coords = demand_points_gdf.loc[demand_points_gdf['label'] == i, 'geometry'].values[0]
        ax.text(
            dp_coords.x, dp_coords.y + 120,
            f"f': {f1_sum}\nf'': {f2_sum}\nTotal: {total_demand}",
            fontsize=8, color='blue',
            clip_on=True  # Ensures text is clipped
        )

    
    # Add a basemap
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    # Apply zoom to selected facility or all open facilities
    ax.set_xlim(min_x - x_margin, max_x + x_margin)
    ax.set_ylim(min_y - y_margin, max_y + y_margin)

    # Remove the axes (box) for a cleaner look
    ax.set_axis_off()

    plt.title(f"Model Solution: Demand & Facility Assignments - Zoomed on {open_facility_label if open_facility_label else 'All Open Facilities'}")
    plt.show()



# Plot the solution with a map background and covering radius from each of the facilities that depend on the model solution

def plot_solution_with_model_dependent_covering_radius(model, demand_points_gdf, hfs_gdf, plot_first_cov_radius, plot_second_cov_radius, plot_color_first_assignment, plot_color_second_assignment):
    """
    Visualize the solution of the Pyomo model on a real-world map using Contextily.
    Now, cov_radius for each open HP and HC is the distance to the furthest allocated demand point.
    
    Parameters:
        model: Solved Pyomo model
        demand_points_gdf: GeoDataFrame with demand points (geometry and labels)
        hfs_gdf: GeoDataFrame with potential locations for health posts (HP) and health centres (HC) (geometry and labels)
    """

    # Convert to Web Mercator projection for map compatibility
    demand_points_gdf = demand_points_gdf.to_crs(epsg=3857)
    hfs_gdf = hfs_gdf.to_crs(epsg=3857)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot demand points (without names)
    #demand_points_gdf.plot(ax=ax, color='red', markersize=50)

    # Plot demand points colored by their assigned facility
    for i in model.I:
        first_assignment = None
        second_assignment = None
        
        for j in model.J:
            if model.x1[i, j].value > 0:
                first_assignment = j
            if model.x2[i, j].value > 0:
                second_assignment = j

        demand_coords = demand_points_gdf.loc[demand_points_gdf['label'] == i, 'geometry'].values[0]
        
        # Determine color based on first or second assignment
        plot_color = None
        
        if plot_color_first_assignment and first_assignment is not None:
            assigned_facility_type = next(l for l in model.L if model.y[first_assignment, l].value > 0)
            plot_color = 'lightblue' if assigned_facility_type == 'hp' else 'orange'
        
        elif not plot_color_first_assignment and plot_color_second_assignment and second_assignment is not None:
            assigned_facility_type = next(l for l in model.L if model.y[second_assignment, l].value > 0)
            plot_color = 'lightgreen' if assigned_facility_type == 'hc' else None 

         # Plot only if a color is assigned
        if plot_color:
            ax.scatter(demand_coords.x, demand_coords.y, color=plot_color, s=50, edgecolor='black')

    # Plot open HPs (blue triangles) and HCs (black squares) without labels
    for j in model.J:
        for l in model.L:
            if model.y[j, l].value > 0:  # Only plot open facilities
                hf_coords = hfs_gdf.loc[hfs_gdf['label'] == j, 'geometry'].values[0]
                
                # Get the demand points assigned to the facility (use model variables to determine assignments)
                first_assigned_demand_points = [i for i in model.I if model.x1[i, j].value > 0]
                second_assigned_demand_points = [i for i in model.I if model.x2[i, j].value > 0]

                # Compute the distance to the furthest allocated demand point
                first_max_distance = 0
                second_max_distance = 0
                first_furthest_demand = None
                second_furthest_demand = None
                for i in first_assigned_demand_points:
                    demand_coords = demand_points_gdf.loc[demand_points_gdf['label'] == i, 'geometry'].values[0]
                    distance = hf_coords.distance(demand_coords)
                    if distance > first_max_distance:
                        first_max_distance = distance
                        first_furthest_demand = demand_coords

                for i in second_assigned_demand_points:
                    demand_coords = demand_points_gdf.loc[demand_points_gdf['label'] == i, 'geometry'].values[0]
                    distance = hf_coords.distance(demand_coords)
                    if distance > second_max_distance:
                        second_max_distance = distance
                        second_furthest_demand = demand_coords

                # Now max_distance is the cov_radius for this facility
                first_cov_radius = first_max_distance
                second_cov_radius = second_max_distance

                # Plot the covering radius as a circle around the facility
                if l == 'hp' and plot_first_cov_radius:  # Plot a circle around HP
                    circle = Circle(
                        (hf_coords.x, hf_coords.y), radius=first_cov_radius, color='lightblue', alpha=0.6
                    )
                    ax.add_patch(circle)

                elif l == 'hc':
                    # Plot covering radius for first assignment (x1) if demand points are assigned
                    if plot_first_cov_radius:
                        for i in first_assigned_demand_points:
                            if model.x1[i, j].value > 0:  # First assignment for HC
                                circle = Circle(
                                    (hf_coords.x, hf_coords.y), radius=first_cov_radius, color='orange', alpha=0.2
                                )
                                ax.add_patch(circle)
                                break  # First assignment is plotted once

                    # Plot covering radius for second assignment (x2) if demand points are assigned
                    if plot_second_cov_radius:
                        for i in second_assigned_demand_points:
                            if model.x2[i, j].value > 0:  # Second assignment for HC
                                circle = Circle(
                                    (hf_coords.x, hf_coords.y), radius=second_cov_radius, color='lightgreen', alpha=0.2
                                )
                                ax.add_patch(circle)
                                break  # Second assignment is plotted once

                # Plot the actual facility location (marker for HP or HC)
                if l == 'hp':
                    ax.plot(hf_coords.x, hf_coords.y, marker='^', color='blue', markersize=8)
                elif l == 'hc':
                    ax.plot(hf_coords.x, hf_coords.y, marker='s', color='black', markersize=8)
                                # Add the line to the furthest demand point


                if first_furthest_demand is not None and plot_first_cov_radius:
                    ax.plot([hf_coords.x, first_furthest_demand.x], [hf_coords.y, first_furthest_demand.y], color='blue', linestyle='-', linewidth=0.5)
                    ax.text((hf_coords.x + first_furthest_demand.x) / 2, (hf_coords.y + first_furthest_demand.y) / 2,
                            f'{first_max_distance:.2f} m', color='blue', fontsize=10, ha='center')

                if second_furthest_demand is not None and plot_second_cov_radius:
                    ax.plot([hf_coords.x, second_furthest_demand.x], [hf_coords.y, second_furthest_demand.y], color='green', linestyle='-', linewidth=1)
                    ax.text((hf_coords.x + second_furthest_demand.x) / 2, (hf_coords.y + second_furthest_demand.y) / 2,
                            f'{second_max_distance:.2f} m', color='green', fontsize=10, ha='center')


    # Add a basemap
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

    # Remove the axes (box) for a cleaner look
    ax.set_axis_off()

    plt.title("Open facilities and covering radius based on assigned demand points")
    
    # Compute the bounding box of demand points and facilities
    all_x = list(demand_points_gdf.geometry.x) + list(hfs_gdf.geometry.x)
    all_y = list(demand_points_gdf.geometry.y) + list(hfs_gdf.geometry.y)

    # Determine the largest covering radius among all facilities
    max_covering_radius = 0

    for j in model.J:
        for l in model.L:
            if model.y[j, l].value > 0:  # Only consider open facilities
                hf_coords = hfs_gdf.loc[hfs_gdf['label'] == j, 'geometry'].values[0]

                # Compute the maximum distance to assigned demand points
                first_max_distance = max(
                    (hf_coords.distance(demand_points_gdf.loc[demand_points_gdf['label'] == i, 'geometry'].values[0])
                    for i in model.I if model.x1[i, j].value > 0), default=0
                )
                second_max_distance = max(
                    (hf_coords.distance(demand_points_gdf.loc[demand_points_gdf['label'] == i, 'geometry'].values[0])
                    for i in model.I if model.x2[i, j].value > 0), default=0
                )

                # Update max covering radius
                max_covering_radius = max(max_covering_radius, first_max_distance, second_max_distance)

    # Set fixed margins considering the largest covering radius
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    # Compute center and max span
    x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
    
    # Set zoom factor: reduce the impact of max_covering_radius
    zoom_factor = 0.35  # Adjust this value (lower = more zoomed in)
    max_span = max((x_max - x_min), (y_max - y_min)) / 2 + zoom_factor * max_covering_radius


    # Set limits to maintain a square aspect ratio
    ax.set_xlim(x_center - max_span, x_center + max_span)
    ax.set_ylim(y_center - max_span, y_center + max_span)

    # Ensure square aspect ratio
    ax.set_aspect("equal")

    plt.show()



# Plot solution; this is intended for the first (toy) example of the grid. Not if we have a real-world coordinate system.

def plot_solution_v1(model, demand_points_gdf, hfs_gdf, show_first_assignment=False, show_second_assignment=False):
    # Extract open facilities from the model
    open_facilities = {
        j: l
        for (j, l) in model.y.keys()
        if pyo.value(model.y[j, l]) > 0  # Only keep open facilities
    }

    # Open facilities GeoDataFrame
    open_hfs_gdf = hfs_gdf[hfs_gdf['label'].isin(open_facilities.keys())].copy()


    # Add a column for facility type (HP or HC) based on l
    open_hfs_gdf['facility_type'] = open_hfs_gdf['label'].map(open_facilities)


    # Extract first assignments from the model
    assignments1 = {
        (i, j): pyo.value(model.x1[i, j])
        for (i, j) in model.x1.keys()
        if pyo.value(model.x1[i, j]) > 0  # Only keep active first assignments
    }

    # Extract second assignments from the model
    assignments2 = {
        (i, j): pyo.value(model.x2[i, j])
        for (i, j) in model.x2.keys()
        if pyo.value(model.x2[i, j]) > 0  # Only keep active second assignments
    }

    # Create a list of connections (coordinates) for first assignment
    connections1 = [
        (
            demand_points_gdf.loc[demand_points_gdf['label'] == i, 'geometry'].values[0],  # Demand point coords
            hfs_gdf.loc[hfs_gdf['label'] == j, 'geometry'].values[0]  # Facility coords
        )
        for (i, j) in assignments1.keys()
    ]

    # Create a list of connections (coordinates) for second assignment
    connections2 = [
        (
            demand_points_gdf.loc[demand_points_gdf['label'] == i, 'geometry'].values[0],  # Demand point coords
            hfs_gdf.loc[hfs_gdf['label'] == j, 'geometry'].values[0]  # Facility coords
        )
        for (i, j) in assignments2.keys()
    ]


    fig, ax = plt.subplots(figsize=(5,5))

    # Plot the grid and demand points
    demand_points_gdf.plot(ax=ax, color='red', label='Demand points')

    # Add labels for demand points
    for idx, row in demand_points_gdf.iterrows():
        ax.text(row.geometry.x + 0.05, row.geometry.y + 0.05, row['label'], fontsize=8, color='red')


    # Plot open HPs and HCs separately, only if they are not empty

    hps_open_gdf = open_hfs_gdf[open_hfs_gdf['facility_type'] == 'hp']
    hcs_open_gdf = open_hfs_gdf[open_hfs_gdf['facility_type'] == 'hc']

    if not hps_open_gdf.empty:
        hps_open_gdf.plot(ax=ax, color='blue', marker='^', label='HPs')

    if not hcs_open_gdf.empty:
        hcs_open_gdf.plot(ax=ax, color='black', marker='s', label='HCs')

    # Add labels for open facilities
    for idx, row in open_hfs_gdf.iterrows():
        ax.text(row.geometry.x + 0.05, row.geometry.y + 0.05, row['label'], fontsize=8, color='black')

    
    # Conditionally add first assignment arrows
    if show_first_assignment:
        for (start, end) in connections1:
            ax.annotate(
                '', xy=(end.x, end.y), xytext=(start.x, start.y),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1),
                zorder=1
            )

    # Conditionally add second assignment arrows
    if show_second_assignment:
        for (start, end) in connections2:
            ax.annotate(
                '', xy=(end.x, end.y), xytext=(start.x, start.y),
                arrowprops=dict(arrowstyle='->', color='orange', lw=1),
                zorder=1
            )

    # Set axis limits based on your grid dimensions (6x6)
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.5, 6.5)

    # Add gridlines at every 0.5 unit
    ax.set_xticks([x * 0.5 for x in range(13)])  # From 0 to 6 with a step of 0.5
    ax.set_yticks([y * 0.5 for y in range(13)])
    ax.grid(color='lightgray', linestyle='--', linewidth=0.5)

    # Add legend and show plot
    ax.legend()
    #plt.title("Location-Allocation Solution with Grid")

    plt.show()



# Create summary table

def generate_facility_summary(model):
    """
    Generates a summary table for facilities in a Pyomo model and saves it as an HTML file.
    
    Parameters:
    - model: Pyomo model containing decision variables and parameters.
    - q: Dictionary with service time requirements per service type.
    - services: List of service types.
    """ 

    rows = []

    # Compute uncovered percentage
    covered = {
        (i, j): pyo.value(model.x1[i, j])
        for (i, j) in model.x1.keys()
        if pyo.value(model.x1[i, j]) > 0  # Only keep covered population zones
    }

    # Compute the total demand for each i by summing over all service types s
    demand_covered = {
        i: sum(model.d1[(i, s)] for s in model.S)  # Adjust service types as needed
        for i in set(i for i, _ in covered.keys())  # Only for demand points in uncovered
    }

    total_demand = sum(model.d1.values())
    total_covered_demand = sum(demand_covered.values())
    uncovered_percentage = 100 * (1 - (total_covered_demand / total_demand))

    # Create HTML snippet for uncovered percentage
    uncovered_html = f"<h3>Uncovered Percentage: {uncovered_percentage:.1f}%</h3>"

    for j in model.J:
        # Determine if facility j is open by checking if any y[j,l] > 0.
        facility_type = None
        for l in model.L:
            if model.y[j, l].value is not None and model.y[j, l].value > 0:
                facility_type = l
                break  # Only one type is assigned per facility.
        if facility_type is None:
            continue  # Skip facilities that are not open.

        # --- Count demand points assigned to the facility ---
        n_assigned_demand_points = sum(
            1 for i in model.I if (
                (model.x1[i, j].value is not None and model.x1[i, j].value > 0) or 
                (model.x2[i, j].value is not None and model.x2[i, j].value > 0)
            )
        )

        # --- Satisfied Demand: compute f1 and f2 sums per service ---
        f1_sums = {}
        f2_sums = {}
        for s in model.S:
            f1_total = 0
            f2_total = 0
            for i in model.I:
                f1_val = model.f1[i, j, s].value if model.f1[i, j, s].value is not None else 0
                f2_val = model.f2[i, j, s].value if model.f2[i, j, s].value is not None else 0
                f1_total += f1_val
                f2_total += f2_val
            f1_sums[s] = f1_total
            f2_sums[s] = f2_total

        overall_f1 = sum(f1_sums[s] for s in model.S)
        overall_f2 = sum(f2_sums[s] for s in model.S)
        overall_satisfied = overall_f1 + overall_f2

        # --- Total Demand Assigned: compute d1 and d2 sums per service ---
        assigned_demand_d1 = {}
        assigned_demand_d2 = {}
        for s in model.S:
            total_d1 = 0
            total_d2 = 0
            for i in model.I:
                d1_val = model.d1[i, s] if model.d1[i, s] is not None else 0
                d2_val = model.d2[i, s] if model.d2[i, s] is not None else 0
                x1_val = model.x1[i, j].value if model.x1[i, j].value is not None else 0
                x2_val = model.x2[i, j].value if model.x2[i, j].value is not None else 0
                total_d1 += d1_val * x1_val
                total_d2 += d2_val * x2_val
            assigned_demand_d1[s] = total_d1
            assigned_demand_d2[s] = total_d2

        overall_d1 = sum(assigned_demand_d1[s] for s in model.S)
        overall_d2 = sum(assigned_demand_d2[s] for s in model.S)
        overall_assigned = overall_d1 + overall_d2

        # --- Compute capacity per service (number of services available) ---
        capacity_per_service = {}
        for s in model.S:
            available_time = 0
            for p in model.P:
                w_val = model.w[j, p].value if model.w[j, p].value is not None else 0
                available_time += model.h[p] * model.a_W[p, s] * w_val
            service_time = model.q[s]
            capacity = int(available_time / service_time) if service_time > 0 else 0
            capacity_per_service[s] = capacity

        # --- Compute Efficiency as (overall satisfied)/(overall assigned) ---
        efficiency = overall_satisfied / overall_assigned if overall_assigned > 0 else None

        # --- Compute new Utilization (Service/Personnel) ---
        # Total service time provided = sum_{s in S} ( (f1_s + f2_s) * q[s] )
        total_service_time = sum((f1_sums[s] + f2_sums[s]) * model.q[s] for s in model.S)
        # Total personnel time = sum_{p in P} (w[j,p] * h[p])
        total_personnel_time = 0
        for p in model.P:
            personnel = model.w[j, p].value if model.w[j, p].value is not None else 0
            total_personnel_time += personnel * model.h[p]
        new_utilization = total_service_time / total_personnel_time if total_personnel_time > 0 else None

        # --- Compute maximum distance from facility j to any assigned demand point ---
        assigned_distances = []
        for i in model.I:
            if ((model.x1[i, j].value is not None and model.x1[i, j].value > 0) or 
                (model.x2[i, j].value is not None and model.x2[i, j].value > 0)):
                assigned_distances.append(model.t[i, j])
        max_distance = round(max(assigned_distances),2) if assigned_distances else 0

        # --- Build the row for facility j ---
        row = {
            "Facility": j,
            "Type": facility_type,
            "#Assigned Demand Points": n_assigned_demand_points,  # New column for demand points assigned
            "Satisfied Demand": f"{int(overall_f1)}; {int(overall_f2)}",
            "Total Demand": f"{int(overall_d1)}; {int(overall_d2)}",
            "Efficiency (%)": f"{efficiency*100:.1f}%" if efficiency is not None else "N/A",
            "Utilization (Service/Personnel)": f"{new_utilization*100:.1f}%" if new_utilization is not None else "N/A",
            "Max Distance": max_distance
        }
        
        # For each service, show satisfied demand as "f1; f2 (Capacity)"
        for s in model.S:
            row[f"Demand_{s}"] = f"{int(f1_sums[s])}; {int(f2_sums[s])} ({capacity_per_service[s]})"
        
        # Add personnel columns (as integers)
        for p in model.P:
            personnel = model.w[j, p].value
            personnel_int = int(personnel) if personnel is not None else 0
            row[f"Personnel_{p}"] = personnel_int
        
        rows.append(row)

    # Create a DataFrame from the collected rows.
    summary_table = pd.DataFrame(rows)

    # --- Enhance headers for managerial insight ---
    header_mapping = {
        "Facility": "Facility ID",
        "Type": "Facility Type",
        "#Assigned Demand Points": "#Assigned Demand Points",
        "Satisfied Demand": "Satisfied Demand (f1; f2)",
        "Total Demand": "Total Demand (d1; d2)",
        "Efficiency (%)": "Efficiency (%)",
        "Utilization (Service/Personnel)": "Utilization (Service/Personnel)",
        "Max Distance": "Max Distance"
    }
    for s in model.S:
        header_mapping[f"Demand_{s}"] = f"Demand - {s.capitalize()} (Capacity)"
    for p in model.P:
        header_mapping[f"Personnel_{p}"] = f"Personnel - {p.capitalize()}"

    summary_table.rename(columns=header_mapping, inplace=True)

    # --- Define a helper function for conditional formatting ---
    def highlight_diff(val):
        try:
            # Expecting format like "a,b (cap)" or "number (cap)"
            left, right = val.split('(')
            left = left.strip()
            capacity_val = int(right.split(')')[0].strip())
            if ',' in left:
                parts = left.split(',')
                demand_val = sum(int(x.strip()) for x in parts)
            else:
                demand_val = int(left)
            diff = capacity_val - demand_val
            if diff < 0:
                return 'background-color: salmon'   # Demand exceeds capacity.
            elif diff > 0:
                return 'background-color: lightgreen'  # Spare capacity available.
            else:
                return ''
        except Exception:
            return ''

    # --- Apply Pandas styling (to output as HTML) ---
    styled_table = summary_table.style.set_table_styles([
        {'selector': 'th',
        'props': [('background-color', '#4F81BD'),
                ('color', 'white'),
                ('font-size', '12pt'),
                ('text-align', 'center'),
                ('padding', '8px')]}
    ]).set_properties(**{'text-align': 'center', 'font-size': '11pt'})

    # Apply conditional formatting on the per-service demand columns.
    for col in summary_table.columns:
        if "Demand - " in col:
            styled_table = styled_table.map(highlight_diff, subset=[col])

    styled_table = styled_table.set_caption("Facility Summary Table - Managerial Insights")

    # Convert the styled table to HTML
    table_html = styled_table.to_html()

    # --- Combine uncovered percentage HTML with the table HTML ---
    html_content = uncovered_html + table_html


    # --- Save the styled table as an HTML file (openable in any browser) ---
    #html = styled_table.to_html()
    with open("facility_summary_with_uncovered_percentage.html", "w") as f:
        f.write(html_content)

    print("Summary table with uncovered percentage saved as 'facility_summary_with_uncovered_percentage.html'.")

    return summary_table

# Optionally, export the raw table to CSV and Excel.
# summary_table.to_csv("facility_summary_improved.csv", index=False)
# summary_table.to_excel("facility_summary_improved.xlsx", index=False)


# Display only the selected variables 
def display_selected_variables(model):
    """
    Display only the selected variables from the Pyomo model.
    """
    print("Selected variables (y_jl = 1, x1_ij = 1, x2_ij = 1, f1_ijs > 0, f2_ijs > 0, w_jp > 0, taumax > 0, deltamax > 0):")

    for j in model.J:
            for l in model.L:
                if model.y[j, l].value is not None and model.y[j, l].value > 0:
                    print(f"y[{j},{l}] = {model.y[j, l].value}")

    for i in model.I:
            for j in model.J:
                    if model.x1[i, j].value is not None and model.x1[i, j].value > 0:
                        print(f"x1[{i},{j}] = {model.x1[i, j].value}") 

    for i in model.I:
            for j in model.J:
                    if model.x2[i, j].value is not None and model.x2[i, j].value > 0:
                        print(f"x2[{i},{j}] = {model.x2[i, j].value} Demand Location is in camp ") 

    for i in model.I:
            for j in model.J:
                for s in model.S:
                    if model.f1[i, j, s].value is not None and model.f1[i, j, s].value > 0:
                        print(f"f1[{i},{j},{s}] = {model.f1[i, j, s].value}") 

    for i in model.I:
            for j in model.J:
                for s in model.S:
                    if model.f2[i, j, s].value is not None and model.f2[i, j, s].value > 0:
                        print(f"f2[{i},{j},{s}] = {model.f2[i, j, s].value}") 

    for j in model.J:
            for p in model.P:
                    if model.w[j, p].value is not None and model.w[j, p].value > 0:
                        print(f"w[{j},{p}] = {model.w[j, p].value}") 
                    
    for c in model.C:
        if model.taumax[c].value is not None and model.taumax[c].value > 0:
            print(f"taumax[{c}] = {model.taumax[c].value}")

    if model.deltamax.value is not None and model.deltamax.value > 0:
        print(f"deltamax = {model.deltamax.value}") 


def display_selected_variables_CAC(model):
    """
    Display only the selected variables from the Pyomo model.
    """
    print("Selected variables (y_jl = 1, x1_ij = 1, x2_ij = 1, f1_ijs > 0, f2_ijs > 0, w_jp > 0, taumax > 0, deltamax > 0):")

    for j in model.J:
            for l in model.L:
                if model.y[j, l].value is not None and model.y[j, l].value > 0:
                    print(f"y[{j},{l}] = {model.y[j, l].value}")

    for i in model.I:
            for j in model.J:
                    if model.x1[i, j].value is not None and model.x1[i, j].value > 0:
                        print(f"x1[{i},{j}] = {model.x1[i, j].value}") 

    for i in model.I:
            for j in model.J:
                    if model.x2[i, j].value is not None and model.x2[i, j].value > 0:
                        print(f"x2[{i},{j}] = {model.x2[i, j].value} Demand Location is in camp ") 

    for i in model.I:
            for j in model.J:
                for s in model.S:
                    if model.f1[i, j, s].value is not None and model.f1[i, j, s].value > 0:
                        print(f"f1[{i},{j},{s}] = {model.f1[i, j, s].value}") 

    for i in model.I:
            for j in model.J:
                for s in model.S:
                    if model.f2[i, j, s].value is not None and model.f2[i, j, s].value > 0:
                        print(f"f2[{i},{j},{s}] = {model.f2[i, j, s].value}") 

    for j in model.J:
            for p in model.P:
                    if model.w[j, p].value is not None and model.w[j, p].value > 0:
                        print(f"w[{j},{p}] = {model.w[j, p].value}") 
                    
    for c in model.C:
        if model.tau1max[c].value is not None and model.tau1max[c].value > 0:
            print(f"tau1max[{c}] = {model.tau1max[c].value}")

    for c in model.C:
        if model.tau2max[c].value is not None and model.tau2max[c].value > 0:
            print(f"tau2max[{c}] = {model.tau2max[c].value}")
