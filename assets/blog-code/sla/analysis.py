# Script for sla analysis and figures used in blog post

# %%
from pathlib import Path
from dask.distributed import Client, LocalCluster
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import rioxarray as rxr
from sklearn.neighbors import BallTree
from sklearn.mixture import GaussianMixture
import scipy.stats as stats
import holoviews as hv
hv.extension('bokeh')
import geoviews as gv
gv.extension('bokeh')

# Define ROOT and DATA directories
ROOT_DIR = Path().absolute().parents[0]
DATA_DIR = Path(
    '/media/durbank/WARP/Research/Glaciers', 
    'sla-estimate/data')

# Initialize dask client
cluster = LocalCluster(processes=False)
client = Client(cluster)
# cluster = LocalCluster()
# client = Client(cluster)

# %%
# Define Landsat directory
LS_dir = DATA_DIR.joinpath(
    'Landsat_8-9_OLI_TIRS_C2_L2')

# Define QA dictionary
# Info taken from [Landsat 8 C2 L2 Science Product Guide (Table 6-3)](https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/atoms/files/LSDS-1619_Landsat8-C2-L2-ScienceProductGuide-v2.pdf)
qa_dict = {
    'Fill': [1], 
    'Clear': [21824, 21826], 
    'Water': [21888, 21890, 23952], 
    'Snow': [30048],
    'Cloud_all': [
        22080, 22144, 22280, 24088, 24216, 24344, 
        24472, 54596, 54852, 55052, 56856, 56984, 
        57240],
    # 'Cloud_mid_conf': [22080, 22144, 24088, 24216],
    # 'Cloud_hi_conf': [22280], 
    'Cloud_shadow': [23888], 
    'NULL': [np.nan]}

def calc_SCA(QA_path, QA_dict):
    """Function to calculate the snow covered area in an Landsat tile (as a fraction of the total usuable pixels) using the provided Landsat pixel QA raster.
    Valid pixels are those that are not designated as padding fill, water, cloud, or cloud shadow.

    Args:
        QA_path (pathlib.PosixPath): The path to the Landsat pixel QA raster
        QA_dict (dict): A dictionary for which raster pixel values correpsond to different classification types.

    Returns:
        fSCA (numpy.float64): The snow-covered area in the Landsat tile, as a fraction of the total valid pixels.    
    """
    QA_pxl = rxr.open_rasterio(
        QA_path, chunks='auto').squeeze().reset_coords(
        'band', drop=True)
    masked_vals = QA_dict["Fill"] + QA_dict["Water"] + QA_dict["Cloud_all"] + QA_dict["Cloud_shadow"] + QA_dict['NULL']
    mask_valid = np.invert(QA_pxl.isin(masked_vals).data)
    mask_snow = QA_pxl.isin(qa_dict["Snow"]).data
    fSCA = mask_snow.sum() / mask_valid.sum()

    return fSCA

def LS_import(tile_path, req_bands=False, band_names=False, band_nums=False):
    """Function to import Landsat tile and format into a xarray object.
    Will also convert data to surface reflectance (or temperature).

    Args:
        tile_path (pathlib.PosixPath): Path to Landsat directory containing raster bands to import
        req_bands (tuple, optional): Tuple of str that are the filename endings of bands of interest to import. If not given, will default to import all tif files in tile path. Defaults to False.
        band_names (list, optional): Desired names to give to imported bands. If not given, will simply assign bands number names based on order of import. Defaults to False.
        band_nums (list, optional): List of band numbers to assign. Currently unused in fucntion. Defaults to False.
    Returns:
        DS (xarray.core.dataset.DataSet): DataSet containing the requested bands and assigned to given band names.
    """
    if req_bands:
        bands_fn = [path for path in tile_path.glob('*') if str(path).endswith(req_bands)]
    else:
        bands_fn = [path for path in tile_path.glob('*') if str(path).endswith('.TIF')]

    if not band_names:
        tmp_bands = np.arange(len(bands_fn))+1
        band_names = [str(el) for el in tmp_bands.tolist()]

    if not band_nums:
        band_nums = (np.arange(len(bands_fn))+1).tolist()

    Bs = []
    for i,band in enumerate(bands_fn):
        # Load current band
        tmp = rxr.open_rasterio(
            band, chunks='auto').squeeze().reset_coords(
            'band', drop=True)
        
        # Assign scaling factors and valid ranges based on raster type (reflectance vs. temperature)
        if '_ST_' in str(band):
            m_scale = 0.00341802
            a_scale = 149.0
            val_min = 1
            val_max = 65535
        elif '_SR_' in str(band):
            m_scale = 0.0000275
            a_scale = -0.2
            val_min = 7273
            val_max = 43636
        else:
            m_scale = 1
            a_scale = 0
            val_min = 1
            val_max = 65535
        
        # Assign invalid data to nan and append to dataarray list
        val_nan = 0
        nan_idx = (tmp.data == val_nan) + (tmp.data < val_min) + (tmp.data > val_max)
        tmp.data = tmp.data.astype('float')
        tmp.data[nan_idx] = np.nan
        tmp.data = m_scale*tmp.data + a_scale
        Bs.append(tmp)

    # Zip dataarrays into combined dataset
    DS = xr.Dataset(
        dict(zip(band_names, Bs)))
    return DS

def get_snowice(data_df):
    """Function to differentiate snow and ice in Landsat pixels.

    Args:
        data_df (pandas.core.frame.DataFrame): DataFrame (converted from xarray) containing x-y coordinates and raster band values of potential snow/ice pixels.

    Returns:
        pandas.core.frame.DataFrame: Returns dataframe (with input dataframe index) for snow/ice class assignment and snow/ice class probability estimates.
    """
    # Create a 2-component GM model and fit to data
    GM = GaussianMixture(
        n_components=2, max_iter=25, 
        random_state=777)
    assignment = GM.fit_predict(data_df)
    k_densities = GM.predict_proba(data_df)

    # Assign factor labels to predicted clusters
    class_dict = {
        GM.means_.mean(axis=1).argmin():'ice', 
        GM.means_.mean(axis=1).argmax():'snow'}

    # Create df for data output
    output_df = pd.DataFrame(
        {'class': assignment}, index=data_df.index)
    output_df.replace(
        {'class':class_dict}, inplace=True)

    # Assign class probabilities to each data point
    output_df['snow_proba'] = k_densities[:,[
        key for key,val in class_dict.items() 
        if val is 'snow']]
    output_df['ice_proba'] = k_densities[:,[
        key for key,val in class_dict.items() 
        if val is 'ice']]

    return output_df

def get_snowline(data_df, buffer_sz=100, df_crs=None):

    # Convert class data to gdf
    gdf = data_df.reset_index()
    gdf = gpd.GeoDataFrame(
        gdf.drop(['x','y'], axis=1), 
        geometry=gpd.points_from_xy(gdf.x, gdf.y), 
        crs=df_crs)

    # Produce gdfs for snow/ice and poly gdfs based on buffers
    snow_pts = gdf[gdf['class']=='snow']
    snow_poly = gpd.GeoDataFrame(
        snow_pts.drop('class', axis=1), 
        geometry=snow_pts.buffer(buffer_sz))
    ice_pts = gdf[gdf['class']=='ice']
    ice_poly = gpd.GeoDataFrame(
        ice_pts.drop('class', axis=1), 
        geometry=ice_pts.buffer(buffer_sz))

    # Find snow/ice pixels within buffer of the other class
    sl_ice = gpd.sjoin(
        ice_pts, snow_poly, predicate='within').drop_duplicates(
            subset='geometry')
    sl_ice = sl_ice.drop(
        sl_ice.filter(regex='_right').columns.append(
            sl_ice.filter(regex='_left').columns), 
        axis=1)
    sl_snow = gpd.sjoin(
        snow_pts, ice_poly, predicate='within').drop_duplicates(
            subset='geometry')
    sl_snow = sl_snow.drop(
        sl_snow.filter(regex='_right').columns.append(
            sl_snow.filter(regex='_left').columns), 
        axis=1)

    # Make gdf of snowline pixels based on spatial proximity of assigned class
    sl_gdf = pd.concat([sl_ice, sl_snow]).sort_index()
    sl_gdf['ice_proba'] = gdf.loc[sl_gdf.index, 'ice_proba']
    # sl_gdf['snow_proba'] = gdf.loc[sl_gdf.index, 'snow_proba']

    return sl_gdf

def get_nearest(
    src_points, candidates, k_neighbors=1):
    """Find nearest neighbors for all source points from a set of candidate points"""

    # Create tree from the candidate points
    tree = BallTree(candidates, leaf_size=15, metric='haversine')

    # Find closest points and distances
    distances, indices = tree.query(src_points, k=k_neighbors)

    # Transpose to get distances and indices into arrays
    distances = distances.transpose()
    indices = indices.transpose()

    if k_neighbors==2:
        # Select 2nd closest (as first closest will be the same point)
        closest = indices[1]
        closest_dist = distances[1]
    else:
        # Get closest indices and distances (i.e. array at index 0)
        # note: for the second closest points, you would take index 1, etc.
        closest = indices[0]
        closest_dist = distances[0]

    # Return indices and distances
    return (closest, closest_dist)

def extract_at_pts(
    xr_ds, gdf_pts, coord_names=['lon','lat'], 
    return_dist=False, planet_radius=6371000):
    """
    Function where, given an xr-dataset and a Point-based geodataframe, extract all values of variables in xr-dataset at pixels nearest the given points in the geodataframe.
    xr_ds {xarray.core.dataset.Dataset}: Xarray dataset containing variables to extract.
    gdf_pts {geopandas.geodataframe.GeoDataFrame} : A Points-based geodataframe containing the locations at which to extract xrarray variables.
    coord_names {list}: The names of the longitude and latitude coordinates within xr_ds.
    return_dist {bool}: Whether function to append the distance (in meters) between the given queried points and the nearest raster pixel centroids. 
    NOTE: This assumes the xr-dataset includes lon/lat in the coordinates 
    (although they can be named anything, as this can be prescribed in the `coord_names` variable).
    """

    # Convert xr dataset to df and extract coordinates
    xr_df = xr_ds.to_dataframe().reset_index()
    xr_coord = xr_df[coord_names]

    # Ensure gdf_pts is in lon/lat and extract coordinates
    crs_end = gdf_pts.crs 
    gdf_pts.to_crs(epsg=4326, inplace=True)
    pt_coord = pd.DataFrame(
        {'Lon': gdf_pts.geometry.x, 
        'Lat': gdf_pts.geometry.y}).reset_index(drop=True)

    # Convert lon/lat points to RADIANS for both datasets
    xr_coord = xr_coord*np.pi/180
    pt_coord = pt_coord*np.pi/180

    # Find xr data nearest given points
    xr_idx, xr_dist = get_nearest(pt_coord, xr_coord)

    # Drop coordinate data from xr (leaves raster values)
    cols_drop = list(dict(xr_ds.coords).keys())
    xr_df_filt = xr_df.iloc[xr_idx].drop(
        cols_drop, axis=1).reset_index(drop=True)
    
    # Add raster values to geodf
    gdf_return = gdf_pts.reset_index(
        drop=True).join(xr_df_filt)
    
    # Add distance between raster center and points to gdf
    if return_dist:
        gdf_return['dist_m'] = xr_dist * planet_radius
    
    # Reproject results back to original projection
    gdf_return.to_crs(crs_end, inplace=True)

    return gdf_return

# %%

# Calculate snow-covered areas for tiles and assign lowest as tile to process
yr_dirs = [path for path in LS_dir.glob('*') if path.is_dir]
tiles = []
for yr in yr_dirs:

    qa_paths = [
        path for path in yr.glob('**/*') 
        if str(path).endswith('_QA_PIXEL.TIF')]
    fSCAs = [calc_SCA(path, qa_dict) for path in qa_paths]
    tiles.append(qa_paths[np.argmin(fSCAs)].parents[0])


# Desired tile/bands to import
my_bands = (
    'B2.TIF', 'B3.TIF', 'B4.TIF', 'B5.TIF', 
    'B6.TIF', 'B7.TIF', 'B10.TIF')
band_names = [
    'Blue', 'Green', 'Red', 'NIR', 
    'SWIR1', 'SWIR2', 'TIR']

# Import first end-of-summer tile
tile = tiles[0]
DS_tile = LS_import(tile, req_bands=my_bands, band_names=band_names)

# Define Aletsch Glacier boundaries
bb = np.array([7.9, 46.4, 8.2, 46.6])
poly = Polygon(
    [(bb[0], bb[1]), (bb[2], bb[1]), (bb[2], bb[3]), (bb[0], bb[3])])
poly_gdf = gpd.GeoDataFrame([1], geometry=[poly], crs="EPSG:4326")
poly_gdf.to_crs(DS_tile.rio.crs, inplace=True)
bb_trans = poly_gdf.total_bounds

# Clip Landsat to Aletsch bounds
DS = DS_tile.rio.clip_box(
    minx=bb_trans[0], miny=bb_trans[1], maxx=bb_trans[2], maxy=bb_trans[3])

# Import QA band and clip to Aletsch bounds
qa_fn = [
    path for path in tile.glob('*') 
    if str(path).endswith('_QA_PIXEL.TIF')].pop()
QA_pxl = rxr.open_rasterio(
    qa_fn).squeeze().reset_coords('band', drop=True).rio.clip_box(
        minx=bb_trans[0], miny=bb_trans[1], 
        maxx=bb_trans[2], maxy=bb_trans[3])

# Subset data to just cryosphere pixels
SNOW = DS.where(QA_pxl.isin(qa_dict["Snow"]), drop=True)

# %% Gaussian mixture model

# Extract snow pixels to df
snow_vars = ['Blue', 'Green', 'Red', 'NIR']
df = SNOW[snow_vars].to_dataframe().dropna().drop(
    columns='spatial_ref')

# Assign snow/ice pixel classes
class_df = get_snowice(df)

# Add classes and class probabilities to original xarray DataSet
SNOW = SNOW.merge(class_df.to_xarray())

# Define snowline proxy locations
sla_gdf = get_snowline(class_df, df_crs=SNOW.rio.crs, buffer_sz=50)

# %% Extract elevation values for snowline

# Import DEM
dem_paths = [
    DATA_DIR.joinpath(
        'DEM/COP_30m-DGED/', 
        'DEM1_SAR_DGE_30_20110104T172529_20140913T171753_ADS_000000_Ctjc.DEM', 
        'Copernicus_DSM_10_N46_00_E007_00/DEM/', 
        'Copernicus_DSM_10_N46_00_E007_00_DEM.tif'), 
    # DATA_DIR.joinpath(
    #     'DEM/COP_30m-DGED/', 
    #     'DEM1_SAR_DGE_30_20110126T172440_20140913T171753_ADS_000000_rZF5.DEM', 
    #     'Copernicus_DSM_10_N45_00_E007_00/DEM/', 
    #     'Copernicus_DSM_10_N45_00_E007_00_DEM.tif'), 
    DATA_DIR.joinpath(
        'DEM/COP_30m-DGED/', 
        'DEM1_SAR_DGE_30_20110408T171540_20140913T171753_ADS_000000_hGDU.DEM', 
        'Copernicus_DSM_10_N46_00_E008_00/DEM/', 
        'Copernicus_DSM_10_N46_00_E008_00_DEM.tif')
]
dems = []
for path in dem_paths:
    dem_i = rxr.open_rasterio(
    path, chunks='auto').squeeze().reset_coords('band', drop=True)
    dems.append(dem_i)
dem_full = dems[0]
for tile in dems[1::]:
    dem_full = dem_full.combine_first(tile)

dem = dem_full.rio.clip_box(minx=bb[0], miny=bb[1], maxx=bb[2], maxy=bb[3])

# Extract elevations at snow/ice pixels, only keeping those within 30-m of dem pixel
sla_all = extract_at_pts(
    xr.Dataset({'elev':dem}), sla_gdf, 
    coord_names=['x','y'], return_dist=True)
sla_gdf = sla_all.loc[sla_all['dist_m']<30].drop(columns=['dist_m'])

# %% Reproject all the things!!! (for plotting)

DS = DS.rio.reproject("EPSG:4326")
QA_pxl = QA_pxl.rio.reproject("EPSG:4326")
# SNOW = SNOW.rio.reproject("EPSG:4326")
sla_gdf.to_crs(epsg=4326, inplace=True)

# %%
fig, ax = plt.subplots()
sla_gdf[sla_gdf['class']=='snow']['elev'].plot(
    kind='kde', color='cyan', ax=ax, linestyle='-.', linewidth=0.5)
sla_gdf[sla_gdf['class']=='ice']['elev'].plot(
    kind='kde', color='blue', ax=ax, linestyle='-.', linewidth=0.5)

def kde_weighted(w_vals, dat_vals):

    weights = w_vals / w_vals.sum()
    kernel = stats.gaussian_kde(dat_vals, weights=w_vals)
    X = dat_vals.sort_values()
    Y = kernel(X)
    weights = weights.loc[X.index]
    return (weights, X, Y)

w_snow, X_snow, Y_snow = kde_weighted(
    w_vals=(1-sla_gdf[sla_gdf['class']=='snow']['ice_proba']), 
    dat_vals=sla_gdf[sla_gdf['class']=='snow']['elev'])
w_ice, X_ice, Y_ice = kde_weighted(
    w_vals=(sla_gdf[sla_gdf['class']=='ice']['ice_proba']), 
    dat_vals=sla_gdf[sla_gdf['class']=='ice']['elev'])

(w_snow*X_snow).sum()
(w_ice*X_ice).sum()

# %% Some figures

# True color image
true_plt = gv.RGB(
    (DS['x'], DS['y'], DS['Red'], DS['Green'], DS['Blue'])).opts(
        height=450, width=450)

# Elevation plot for region
elev_plt = gv.Image(dem).opts(
    cmap='bmy', width=500, height=450, colorbar=True, 
    clabel="Elevation (meters)", tools=['hover'])

study_plt = (true_plt + elev_plt)

# Each band used in the mixture model
nrgb_plts = (
    gv.Image(DS['NIR']).opts(
        height=450, width=450, tools=['hover'], 
        cmap='oranges', title="Near IR Band") 
    + gv.Image(DS['Red']).opts(
        height=450, width=450, tools=['hover'], 
        cmap='reds', title="Red Band") 
    + gv.Image(DS['Green']).opts(
        height=450, width=450, tools=['hover'], 
        cmap='greens', title="Green Band") 
    + gv.Image(DS['Blue']).opts(
        height=450, width=450, tools=['hover'], 
        cmap='blues', title="Blue Band")
).cols(2)

# Histograms of reflectance in cryosphere pixels for bands with strongest snow/ice contrast
snow_vars = ['Blue', 'Green', 'Red', 'NIR']
color_dict = {
    'Blue':'blue', 'Green':'green', 
    'Red':'red', 'NIR': 'orange'}

fig_hist, axs = plt.subplots(2,2, figsize=(16,12))
for i,ax in enumerate(axs.reshape(-1)):
    var = snow_vars[i]
    ax.hist(
        SNOW[var].compute().data.flatten(), 
        color=color_dict[var])
    ax.set_title(list(color_dict.keys())[i])

# Location of snow/ice pixels elevations
snow_elev = gv.Points(sla_gdf[sla_gdf['class']=="snow"], vdims=['elev']).opts(
    color='elev', cmap='bmy', 
    colorbar=True, clabel="Elevation (meters)", 
    tools=['hover']).redim.range(
        elev=elev_plt.range(dim='z'))
ice_elev = gv.Points(sla_gdf[sla_gdf['class']=="ice"], vdims=['elev']).opts(
    color='elev', cmap='bmy', 
    colorbar=True, clabel="Elevation (meters)", 
    tools=['hover']).redim.range(
        elev=elev_plt.range(dim='z'))
elev_dist = (hv.Distribution(snow_elev, kdims=['elev']).opts(
                alpha=0.5, color='cyan') *
            hv.Distribution(ice_elev, kdims=['elev']).opts(
                alpha=0.5, color='cornflowerblue')).redim.range(
        elev=elev_plt.range(dim='z'))
sla_plt = (
    true_plt * (snow_elev * ice_elev)
    ).opts(width=650, height=600) << elev_dist.opts(height=600, width=150)

# Class probability-weighted ELA estimates
snow_dist = (
    hv.Distribution((X_snow, Y_snow)).opts(
        color='cyan', alpha=0.5) * 
    hv.Distribution(sla_gdf[sla_gdf['class']=="snow"]['elev']).opts(
        filled=False, line_color='cyan', line_dash='dotted', 
        line_width=4, tools=['hover']) *
    hv.VLine(X_snow.iloc[Y_snow.argmax()]).opts(
        color='black', line_width=2, line_dash='dashed')
    ).opts(xlabel='Snowline elevation (meters)')
ice_dist = (
    hv.Distribution((X_ice, Y_ice)).opts(
        color='cornflowerblue', alpha=0.5) * 
    hv.Distribution(sla_gdf[sla_gdf['class']=="ice"]['elev']).opts(
        filled=False, line_color='cornflowerblue', line_dash='dotted', 
        line_width=4, tools=['hover']) * 
    hv.VLine(X_ice.iloc[Y_ice.argmax()]).opts(
        color='black', line_width=2, line_dash='dashed')
    ).opts(xlabel='Iceline elevation (meters)')

ela_dist = (
    snow_dist.opts(width=450, height=450) + 
    ice_dist.opts(width=450, height=450))

print(
    f"The final weighted estimate of the snowline altitude is " + 
    f"{X_snow.iloc[Y_snow.argmax()]:.0f}\u00B1" + 
    f"{sla_gdf.groupby('class').std()['elev']['snow']:.0f} meters.")
print(
    f"The final weighted estimate of the iceline altitude is " + 
    f"{X_ice.iloc[Y_ice.argmax()]:.0f}\u00B1" + 
    f"{sla_gdf.groupby('class').std()['elev']['ice']:.0f} meters.")

# %% Save figures for later use

gv.save(study_plt, ROOT_DIR.joinpath('figs/study-plt.html'))
gv.save(nrgb_plts, ROOT_DIR.joinpath('figs/bands-plt.html'))
fig_hist.savefig(ROOT_DIR.joinpath('figs/snow-hists.png'))
gv.save(sla_plt, ROOT_DIR.joinpath('figs/sla-plt.html'))
hv.save(ela_dist, ROOT_DIR.joinpath('figs/ela-dist.html'))
