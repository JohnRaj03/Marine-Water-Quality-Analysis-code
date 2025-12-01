from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from IPython.display import JSON
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import earthaccess
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time


start_time = time.time()
auth = earthaccess.login(persist=True)
results = earthaccess.search_datasets(keyword="L2 ocean color", instrument="MODIS")
print(f"Authenticated and searched datasets in {time.time() - start_time} seconds.")


tspan = ("2013-01-01", "2023-12-31")
bbox = (-6.0, 49.8, -4.0, 49.8)
cc = (0, 50)


start_time = time.time()
chl_results = earthaccess.search_data(
    short_name="MODISA_L2_OC",
    temporal=tspan,
    bounding_box=bbox,
    cloud_cover=cc
)
print(f"Searched for chlorophyll-a data in {time.time() - start_time} seconds.")


start_time = time.time()
sst_results = earthaccess.search_data(
    short_name="MODISA_L2_SST",
    temporal=tspan,
    bounding_box=bbox,
    cloud_cover=cc
)
print(f"Searched for SST data in {time.time() - start_time} seconds.")


start_time = time.time()
chl_paths = earthaccess.download(chl_results, "chlor_a")
sst_paths = earthaccess.download(sst_results, "sst")
print(f"Downloaded data in {time.time() - start_time} seconds.")

start_time = time.time()
prod_chl = xr.open_dataset(chl_paths[0])
obs_chl = xr.open_dataset(chl_paths[0], group="geophysical_data")
nav_chl = xr.open_dataset(chl_paths[0], group="navigation_data")
print(f"Opened chlorophyll-a datasets in {time.time() - start_time} seconds.")


start_time = time.time()
prod_sst = xr.open_dataset(sst_paths[0])
obs_sst = xr.open_dataset(sst_paths[0], group="geophysical_data")
nav_sst = xr.open_dataset(sst_paths[0], group="navigation_data")
print(f"Opened SST datasets in {time.time() - start_time} seconds.")


start_time = time.time()
nav_chl = nav_chl.set_coords(("longitude", "latitude"))
dataset_chl = xr.merge((prod_chl, obs_chl, nav_chl.coords))
print(f"Merged chlorophyll-a datasets in {time.time() - start_time} seconds.")

start_time = time.time()
nav_sst = nav_sst.set_coords(("longitude", "latitude"))
dataset_sst = xr.merge((prod_sst, obs_sst, nav_sst.coords))
print(f"Merged SST datasets in {time.time() - start_time} seconds.")

start_time = time.time()
array_chl = np.log10(dataset_chl["chlor_a"])
array_chl.attrs.update({"units": f'log10({dataset_chl["chlor_a"].attrs["units"]})'})
print(f"Processed chlorophyll-a data in {time.time() - start_time} seconds.")


start_time = time.time()
array_sst = dataset_sst["sst"]
print(f"Processed SST data in {time.time() - start_time} seconds.")


start_time = time.time()
array_pic = dataset_chl["pic"]
print(f"Processed PIC data in {time.time() - start_time} seconds.")


start_time = time.time()
array_poc = dataset_chl["poc"]
print(f"Processed POC data in {time.time() - start_time} seconds.")


start_time = time.time()
df = pd.DataFrame({
    'chlor_a': array_chl.values.flatten(),
    'sst': array_sst.values.flatten(),
    'pic': array_pic.values.flatten(),
    'poc': array_poc.values.flatten()
})
df_cleaned = df.dropna()
print(f"Prepared data for Random Forest in {time.time() - start_time} seconds.")

X = df_cleaned[['chlor_a', 'sst', 'pic', 'poc']]
y = np.random.random(len(df_cleaned)) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Split the data into training and testing sets.")

start_time = time.time()
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print(f"Trained the Random Forest model in {time.time() - start_time} seconds.")


start_time = time.time()
y_pred = rf.predict(X_test)
print(f"Made predictions in {time.time() - start_time} seconds.")


mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


pred_pfz = rf.predict(X)


pred_pfz_full = np.full(df.shape[0], np.nan)
pred_pfz_full[df_cleaned.index] = pred_pfz


pred_pfz_reshaped = pred_pfz_full.reshape(array_chl.shape)

fig, axs = plt.subplots(2, 3, figsize=(18, 10), subplot_kw={'projection': ccrs.PlateCarree()})


ax = axs[0, 0]
mesh_chl = ax.pcolormesh(nav_chl['longitude'], nav_chl['latitude'], array_chl, cmap='jet', transform=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, edgecolor='black')
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.RIVERS)
ax.gridlines(draw_labels=True)
ax.set_title('Chlorophyll-a Concentration')
fig.colorbar(mesh_chl, ax=ax, orientation='vertical', label='Chlorophyll-a (mg/m³)')


ax = axs[0, 1]
mesh_sst = ax.pcolormesh(nav_sst['longitude'], nav_sst['latitude'], array_sst, cmap='jet', transform=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, edgecolor='black')
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.RIVERS)
ax.gridlines(draw_labels=True)
ax.set_title('Sea Surface Temperature')
fig.colorbar(mesh_sst, ax=ax, orientation='vertical', label='SST (°C)')


ax = axs[0, 2]
mesh_pic = ax.pcolormesh(nav_chl['longitude'], nav_chl['latitude'], array_pic, cmap='jet', transform=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, edgecolor='black')
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.RIVERS)
ax.gridlines(draw_labels=True)
ax.set_title('PIC Concentration')
fig.colorbar(mesh_pic, ax=ax, orientation='vertical', label='PIC (units)')


ax = axs[1, 0]
mesh_poc = ax.pcolormesh(nav_chl['longitude'], nav_chl['latitude'], array_poc, cmap='jet', transform=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, edgecolor='black')
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.RIVERS)
ax.gridlines(draw_labels=True)
ax.set_title('POC Concentration')
fig.colorbar(mesh_poc, ax=ax, orientation='vertical', label='POC (units)')


ax = axs[1, 1]
mesh_pfz = ax.pcolormesh(nav_chl['longitude'], nav_chl['latitude'], pred_pfz_reshaped, cmap='jet', transform=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, edgecolor='black')
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.RIVERS)
ax.gridlines(draw_labels=True)
ax.set_title('Predicted Potential Fishing Zones')
fig.colorbar(mesh_pfz, ax=ax, orientation='vertical', label='PFZ Prediction')


axs[1, 2].axis('off')

plt.tight_layout()
plt.show()
