import rasterio.plot
import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import itertools
from scipy.spatial import cKDTree
import geopy.distance
import matplotlib.pyplot as plt
from operator import itemgetter
import rasterio
from pyproj import Transformer
from shapely.geometry import box
from rasterio.mask import mask

# Title of the app
st.title("Automatic Compliance Monitoring for Brick Kilns")

# Create two columns with a 40% - 60% split
col1, col2 = st.columns([0.35, 0.65])
def plot_raster_with_data(tif_file, brick_kilns, bbox):
    """
    Plots a raster image from a TIFF file using a bounding box.
    """
    with rasterio.open(tif_file) as src:
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create a GeoDataFrame with the bounding box
        geo = gpd.GeoDataFrame({'geometry': [bbox]}, crs=src.crs)
        
        # Crop the raster
        out_image, out_transform = mask(src, geo.geometry, crop=True)
        
        # Update metadata
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        # Plot the cropped image
        if out_image.any():
            rasterio.plot.show(out_image, ax=ax, cmap='gray', alpha = 0.2)
            ax.set_title("Cropped TIF Image")
            ax.axis('off')
        else:
            st.error("No data within the specified bounding box.")
            return
        
        brick_kilns = gpd.GeoDataFrame(brick_kilns, geometry=gpd.points_from_xy(brick_kilns.lon, brick_kilns.lat), crs="EPSG:4326")

        
        brick_kilns_compliant = brick_kilns[brick_kilns['compliant']]
        ax.scatter(brick_kilns_compliant.geometry.x, brick_kilns_compliant.geometry.y, color='green', s=10, marker='o', label='Compliant Brick Kilns', zorder=20)

        # Plot non-compliant brick kilns in red
        brick_kilns_non_compliant = brick_kilns[~brick_kilns['compliant']]
        ax.scatter(brick_kilns_non_compliant.geometry.x, brick_kilns_non_compliant.geometry.y, color='red', s=10, marker='o', label='Non-Compliant Brick Kilns', zorder=20)
        


        # Display the plot in Streamlit
        st.pyplot(fig)

def create_bbox_from_coordinates(tif_file, min_lon, min_lat, max_lon, max_lat):
    """
    Creates a bounding box from given coordinates transformed to the raster's CRS.
    """
    with rasterio.open(tif_file) as src:
        raster_crs = src.crs
        transformer = Transformer.from_crs("epsg:4326", raster_crs, always_xy=True)
        min_x, min_y = transformer.transform(min_lon, min_lat)
        max_x, max_y = transformer.transform(max_lon, max_lat)
        
    bbox = box(min_x, min_y, max_x, max_y)
    return bbox
# def crop_and_display_tif(tif_file, bbox):
    """
    Crops a TIFF file using a bounding box and displays the cropped image in Streamlit.

    Args:
    - tif_file (str): Path to the TIFF file.
    - bbox (shapely.geometry.box): Bounding box for cropping.

    Returns:
    - None
    """
    with rasterio.open(tif_file) as src:
        # Create a GeoDataFrame with the bounding box
        geo = gpd.GeoDataFrame({'geometry': [bbox]}, crs=src.crs)
        
        # Crop the raster
        out_image, out_transform = mask(src, geo.geometry, crop=True)
        
        # Update the metadata
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        # Plot the cropped image
        fig, ax = plt.subplots(figsize=(10, 10))
        plt.show(out_image, ax=ax)
        ax.set_title("Cropped TIF Image")
        ax.axis('off')

        # Display the plot in Streamlit
        st.pyplot(fig)

with col1:
    # Dropdown for selecting the state
    # state = st.selectbox("Select State", ["Punjab", "Haryana", "Bihar"])  # Update the list as needed
    state = st.selectbox("Select State", ["All", "Punjab", "Haryana", "Bihar", "West Bengal", "Uttar Pradesh"])  # Update the list as needed

    # Checkboxes for different compliance criteria
    distance_kilns = st.checkbox("Inter-brick kiln distance < 1km")
    distance_hospitals = st.checkbox("Distance to Hospitals < 800m")
    distance_water_bodies = st.checkbox("Distance to Water bodies < 500m")

     # Add a horizontal line to separate sections
    st.markdown("<hr>", unsafe_allow_html=True)

    st.subheader("Show Population Density")
    population_density = False

    population_density = st.checkbox("Yes")
        
with col2:
    fp2 = "/home/shataxi.dubey/shataxi_work/India_State_Shapefile/India_State_Shapefile/India_State_Boundary.shp"
    data2 = gpd.read_file(fp2)

    waterways_path = '/home/shataxi.dubey/shataxi_work/compliance_analysis/waterways/waterways.shp'
    waterways = gpd.read_file(waterways_path)

    tif_file = "/home/shahshrimay/landscan-global-2022-colorized.tif"

    # Load brick kilns data
    brick_kilns_paths = {
        'All' : '/home/shahshrimay/Compliance-app/combined_file.csv',
        "Punjab": '/home/patel_zeel/compass24/exact_latlon/punjab.csv',
        "Haryana": '/home/patel_zeel/compass24/exact_latlon/haryana.csv',
        "Bihar": '/home/patel_zeel/compass24/exact_latlon/bihar.csv',
        "Uttar Pradesh": '/home/patel_zeel/compass24/exact_latlon/uttar_pradesh.csv',
        "West Bengal": '/home/patel_zeel/compass24/exact_latlon/west_bengal.csv',
    }

    # Load brick kilns data for the selected state
    brick_kilns_path = brick_kilns_paths[state]
    brick_kilns = pd.read_csv(brick_kilns_path)

    hospital_df = pd.read_csv('/home/rishabh.mondal/bkdb/India_Hospital_Data.csv')
    hospital_df = hospital_df.rename(columns={'lon': 'Longitude', 'lat': 'Latitude'})

    def ckdnearest(brick_kilns, rivers, gdfB_cols=['geometry']):
        A = np.vstack([np.array(geom) for geom in brick_kilns[['lon', 'lat']].values])
        B = [np.array(geom.coords) for geom in rivers.geometry.to_list()]
        B_ix = tuple(itertools.chain.from_iterable(
            [itertools.repeat(i, x) for i, x in enumerate(list(map(len, B)))]))
        B = np.concatenate(B)
        ckd_tree = cKDTree(B)
        dist, river_point_idx = ckd_tree.query(A, k=1)
        closest_river_point = B[river_point_idx]
        river_origin_idx = itemgetter(*river_point_idx)(B_ix)
        gdf = pd.concat(
            [brick_kilns, rivers.loc[river_origin_idx, gdfB_cols].reset_index(drop=True),
             pd.DataFrame(closest_river_point, columns=['closest_river_point_long', 'closest_river_point_lat']),
             pd.Series(dist, name='dist')], axis=1)
        return gdf

    def ckdnearest_hospital(brick_kilns, hospital_df):
        A = np.vstack([np.array(geom) for geom in brick_kilns[['lon', 'lat']].values])
        B = np.vstack([np.array(geom) for geom in hospital_df[['Longitude', 'Latitude']].values])
        ckd_tree = cKDTree(B)
        dist, hospital_idx = ckd_tree.query(A, k=1)
        closest_hospital_point = B[hospital_idx]
        gdf = pd.concat(
            [brick_kilns,
             pd.DataFrame(closest_hospital_point, columns=['closest_hospital_long', 'closest_hospital_lat']),
             pd.Series(dist, name='dist')], axis=1)
        return gdf

    def cal_bk_hosp_dist(path, hospital_df):
        state_bk = pd.read_csv(path)
        bk_hospital_mapping = ckdnearest_hospital(state_bk, hospital_df)
        bk_hospital_mapping['distance_km'] = 0
        for i in range(len(bk_hospital_mapping)):
            bk_hospital_mapping['distance_km'][i] = geopy.distance.distance(
                (bk_hospital_mapping['lat'][i], bk_hospital_mapping['lon'][i]),
                (bk_hospital_mapping['closest_hospital_lat'][i], bk_hospital_mapping['closest_hospital_long'][i])
            ).km
        return bk_hospital_mapping

    def cal_bk_river_dist(path, waterways):
        state_bk = pd.read_csv(path)
        bk_river_mapping = ckdnearest(state_bk, waterways)
        bk_river_mapping['distance'] = 0
        for i in range(len(state_bk)):
            bk_river_mapping['distance'][i] = geopy.distance.distance(
                (bk_river_mapping['lat'][i], bk_river_mapping['lon'][i]),
                (bk_river_mapping['closest_river_point_lat'][i], bk_river_mapping['closest_river_point_long'][i])
            ).km
        return bk_river_mapping

    def ckdnearest_brick_kilns(brick_kilns):
        A = brick_kilns[['lat', 'lon']].values
        ckd_tree = cKDTree(A)
        dist, idx = ckd_tree.query(A, k=2)
        distances = []
        for i in range(len(A)):
            point1 = (A[i, 0], A[i, 1])
            point2 = (A[idx[i, 1], 0], A[idx[i, 1], 1])
            geo_distance = geopy.distance.distance(point1, point2).km
            distances.append(geo_distance)
        closest_kiln_point = A[idx[:, 1]]
        gdf = pd.concat(
            [brick_kilns,
             pd.DataFrame(closest_kiln_point, columns=['closest_kiln_long', 'closest_kiln_lat']),
             pd.Series(distances, name='dist')], axis=1)
        return gdf

    bk_river_mapping = cal_bk_river_dist(brick_kilns_path, waterways)
    bk_hospital_mapping = cal_bk_hosp_dist(brick_kilns_path, hospital_df)
    bk_kiln_mapping = ckdnearest_brick_kilns(pd.read_csv(brick_kilns_path))

    brick_kilns['compliant'] = True
    if distance_kilns:
        brick_kilns['compliant'] &= bk_kiln_mapping['dist'] >= 1
    if distance_hospitals:
        brick_kilns['compliant'] &= bk_hospital_mapping['distance_km'] >= 0.8
    if distance_water_bodies:
        brick_kilns['compliant'] &= bk_river_mapping['distance'] >= 0.5

    # Plotting the results
    if (population_density == False):
        fig, ax = plt.subplots(figsize=(10, 6))
        data2.plot(ax=ax, cmap='Pastel2', edgecolor='black', linewidth=0.1)  # State map
        waterways.plot(ax=ax, color='blue', linewidth=0.2)  # Water bodies

        # Plot all brick kilns in green
        brick_kilns_compliant = brick_kilns[brick_kilns['compliant']]
        ax.scatter(brick_kilns_compliant['lon'], brick_kilns_compliant['lat'], color='green', s=10, marker='o', label='Compliant Brick Kilns')

        # Plot non-compliant brick kilns in red
        brick_kilns_non_compliant = brick_kilns[~brick_kilns['compliant']]
        ax.scatter(brick_kilns_non_compliant['lon'], brick_kilns_non_compliant['lat'], color='red', s=10, marker='o', label='Non-compliant Brick Kilns')

        if state == 'All': #(plot india map)
            ax.set_xlim(66, 98)
            ax.set_ylim(7, 38)
        elif state == 'Bihar':
            ax.text(83, 25.8, 'Uttar\n Pradesh')
            ax.text(85.5, 25.5, 'Bihar')
            ax.text(87.9, 25.3, 'West\n Bengal')
            ax.set_xlim(83, 88.6)
            ax.set_ylim(24.25, 27.53)
        elif state == 'Haryana':
            ax.text(77.3, 29.5, 'Uttar \nPradesh')
            ax.text(74.5, 28.5, 'Rajasthan')
            ax.text(75.7, 30.7, 'Punjab')
            ax.text(76.2, 29.4, 'Haryana')
            ax.set_xlim(74.2, 78.1)
            ax.set_ylim(27.6, 31)
        elif state == 'Punjab':
            ax.text(76.3, 31.5, 'Himachal Pradesh')
            ax.text(75, 30.8, 'Punjab')
            ax.text(74.3, 30.4, 'Haryana')
            ax.set_xlim(73.8, 77)
            ax.set_ylim(29.5, 32.6)
        elif state == 'Uttar Pradesh':
            ax.text(76.3, 31.5, 'Himachal Pradesh')
            ax.text(75, 30.8, 'Punjab')
            ax.text(74.3, 30.4, 'Haryana')
            ax.set_xlim(76, 85)
            ax.set_ylim(23.6, 31)
        elif state == 'West Bengal':
            # ax.text(76.3, 31.5, 'Himachal Pradesh')
            # ax.text(75, 30.8, 'Punjab')
            # ax.text(74.3, 30.4, 'Haryana')
            ax.set_xlim(85.7, 90)
            ax.set_ylim(21.3, 27.5)

        plt.legend()
        plt.title(f"{state} Brick Kilns Compliance")
        st.pyplot(fig)

    else:
        brick_kilns_non_compliant = brick_kilns[~brick_kilns['compliant']]
        
        if state == 'All':
            bbox1 = create_bbox_from_coordinates(tif_file, min_lon= 60.0, max_lon=98.0, min_lat=7.0, max_lat=48.0 )
            plot_raster_with_data(tif_file, brick_kilns, bbox1)


    
    # Display the number of non-compliant kilns
    num_brick_kilns = len(brick_kilns)
    st.markdown(f"""
        <div style="text-align: center; font-size: 18px;">
            Number of brick kilns: {num_brick_kilns}
        </div>
    """, unsafe_allow_html=True)

    num_non_compliant = len(brick_kilns_non_compliant)
    st.markdown(f"""
        <div style="text-align: center; font-size: 18px;">
            Number of non-compliant brick kilns: {num_non_compliant}
        </div>
    """, unsafe_allow_html=True)