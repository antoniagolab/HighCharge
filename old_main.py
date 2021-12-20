import numpy as np
import pandas as pd
import psycopg2 as ps
import geopandas as gp
from shapely.geometry import MultiPolygon, Polygon, MultiLineString, Point
import folium
import matplotlib.pyplot as plt


conn = ps.connect("port=5432 dbname=osm_probe user=postgres password=antI1995")

strassen_oe = ['S16', 'A12', 'A25', 'S7', 'A9', 'A7', 'D4', 'A23', 'A22', 'S31','S5', 'A3', 'A5', 'A13', 'S6', 'A6', 'S3', 'B37', 'A21', 'S1', 'S4', 'S2', 'S35', 'A10', 'S37', 'S10', 'S33', 'A2', 'B37a', 'A1', 'A11', 'S36', 'A14', 'A8', 'A4']
table = gp.GeoDataFrame.from_postgis('select * from "planet_osm_roads"', conn, geom_col="way", crs=3857)
table_2 = gp.GeoDataFrame.from_postgis('select * from "planet_osm_line"', conn, geom_col="way", crs=3857)

highways = table[table['ref'].isin(strassen_oe)]
highways_2 = table_2[table_2['ref'].isin(strassen_oe)]

# https://www.bev.gv.at/portal/page?_pageid=713,1573889&_dad=portal&_schema=PORTAL
streets = gp.read_file('KM500-V_01.2021/Verkehr/Straßen.shp')
border = gp.read_file('KM500-V_01.2021/Grenzen/Staatsgrenzen.shp')
border_austria = border[border['OBJEKTART']=="Staatsgrenze Österreich"]
border_linestrings = MultiLineString(border_austria.geometry.to_list())
border_austria.to_file('austrian_border.shp')

first_points = []
last_points = []
for l in border_linestrings:
    ps = [(l.xy[0][ij], l.xy[1][ij]) for ij in range(0, len(l.xy))]
    first_points.append(ps[0])
    last_points.append(ps[1])

all_points = first_points
all_points.extend(last_points)
kl = 0
current_ind = 0
link_front = True
border_line = []
while kl < len(border_linestrings):
    curr_point = Point(all_points[current_ind])
    dists = [curr_point.distance(Point(p)) for p in all_points]
    indx_next_line = np.argsort(dists)[1]
    if indx_next_line < len(border_linestrings):
        current_ind = indx_next_line
        next_linestring = border_linestrings[indx_next_line]
    else:
        current_ind = indx_next_line-len(border_linestrings)
        next_linestring = border_linestrings[indx_next_line-len(border_linestrings)]
    border_line.append(next_linestring)
    kl = kl + 1

highways = streets[streets['OBJEKTART'].isin(['Autobahn', 'Schnellstraße'])]



# Raststaetten

service_stations_polygons = gp.GeoDataFrame.from_postgis("select * from planet_osm_polygon where highway='services' or highway='rest_area'", conn, geom_col="way", crs=3857)
service_stations_lines = gp.GeoDataFrame.from_postgis("select * from planet_osm_line where highway='services' or highway='rest_area'", conn, geom_col="way", crs=3857)


for _, r in service_stations_4326.iterrows():
    # Without simplifying the representation of each borough,
    # the map might not be displayed
    sim_geo = gpd.GeoSeries(r['geometry']).simplify(tolerance=0.001)
    geo_j = sim_geo.to_json()
    geo_j = folium.GeoJson(data=geo_j,
                           style_function=lambda x: {'fillColor': 'orange'})
    folium.Popup(r['BoroName']).add_to(geo_j)
    geo_j.add_to(m)
