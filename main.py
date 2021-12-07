import pandas as pd
import geopandas as gpd
import utils

rest_areas = pd.read_csv('data/_demand_calculated.csv')
dir_0, dir_1 = utils.split_by_dir(rest_areas, col_dir='dir')





