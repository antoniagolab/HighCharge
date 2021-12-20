import pandas as pd
import numpy as np
import geopandas as gpd
import psycopg2 as ps
from shapely.ops import linemerge
import matplotlib.pyplot as plt
from shapely.geometry import MultiPolygon, Polygon, MultiLineString, Point

# get highways
conn = ps.connect("port=5432 dbname=osm_probe user=postgres password=antI1995")
highways_roads = gpd.GeoDataFrame.from_postgis("select * from planet_osm_roads where "
                                                       "highway='motorway' or highway = 'trunk'", conn, geom_col="way", crs=3857)

highways_lines = gpd.GeoDataFrame.from_postgis("select * from planet_osm_line where "
                                                       "highway='motorway' or highway = 'trunk'", conn, geom_col="way", crs=3857)
highways_total = highways_roads.append(highways_lines)
highways_total.to_file("highways_austria_osm/highways_austria_osm.shp")