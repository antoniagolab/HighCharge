import geopandas as gpd
from shapely.geometry import LineString, Point
import numpy as np
from shapely import wkt

reference_coord_sys = "EPSG:31287"


def pd2gpd(dataframe, geom_col_name="geometry"):
    """
    turns pandas.DataFrame into geopandas.GeoDataFrame
    :param dataframe: pandas.DataFrame
    :param geom_col_name: String of name of column containing geometry objects
    :return: geopandas.GeoDataFrame
    """
    dataframe[geom_col_name] = dataframe[geom_col_name].apply(wkt.loads)
    dataframe = gpd.GeoDataFrame(dataframe, geometry=geom_col_name)
    return dataframe.set_crs(reference_coord_sys)


def segments(curve):
    """
    extracts all segments from a LineString, segments = LineString consisting of only two points
    :param curve: shapely.geometry.LineString
    :return: list of LineString elements
    """
    return list(map(LineString, zip(curve.coords[:-1], curve.coords[1:])))


def get_leftmost_coord(geom):
    """
    gets the leftmost coordinate of a geometry object
    :param geom: shapely.geometry.LineString/MultiLineString/Point
    :return: float
    """
    return min(geom.xy[0])


def get_rightmost_coord(geom):
    """
    gets rightmost coordinate of a geometry object
    :param geom: shapely.geometry.LineString/MultiLineString/Point
    :return: float
    """
    return max(geom.xy[0])


def sort_left_to_right(geom_list):
    """
    sorting geometries from left to right
    :param geom_list: list containing shapely.geometry.LineString/MultiLineString/Point
    :return: list of sorted shapely.geometry objects, numpy.array() of argsort indices of input list
    """
    rightest_coords = [get_leftmost_coord(geom) for geom in geom_list]
    sort_indx = np.argsort(rightest_coords)
    sorted_list = [geom_list[ij] for ij in sort_indx]
    return sorted_list, sort_indx


def finding_segments_point_lies_on(point_list, linestring_gdf):
    """

    :param point_list: list with shapely.geometry.Point objects
    :param linestring_gdf: geopandas.GeoDataFrame
    :return:
    """
    linestring_list = linestring_gdf.geometry.to_list()
    linestring_ids = linestring_gdf["ID"].to_list()
    connections = {}
    for ij in range(0, len(point_list)):
        current_p = point_list[ij]
        id_list = []
        for kl in linestring_ids:
            current_l = linestring_list[kl]
            if current_l.distance(current_p) < 1e-3:
                id_list.append(kl)
        connections[ij] = id_list

    return connections


def get_leftmost_end_point(linestring):
    """
    extracts leftmost end point of linestring
    :param linestring: shapely.geometry.LineString
    :return: shapely.geometry.Point
    """
    point_0_x = linestring.xy[0][0]
    point_1_x = linestring.xy[0][-1]
    if point_0_x <= point_1_x:
        return point_0_x
    else:
        return point_1_x


def get_rightmost_end_point(linestring):
    """
    extracts rightmost end point of linestring
    :param linestring: shapely.geometry.LineString
    :return: shapely.geometry.Point
    """
    point_0_x = linestring.xy[0][0]
    point_1_x = linestring.xy[0][-1]
    if point_0_x >= point_1_x:
        return point_0_x
    else:
        return point_1_x


def sort_points_in_linestring_l2r(linestring):
    """

    :param linestring:
    :return:
    """
    point_left = get_leftmost_end_point(linestring)
    nb_p = len(linestring.xy[0])
    point_list = [Point(linestring.xy[0][ij], linestring.xy[1][ij]) for ij in range(0, nb_p)]
    if not (point_list[0].x == point_left):
        point_list.reverse()

    linestring_l2r = LineString(point_list)
    return linestring_l2r


def is_right(p1, p2, p3):
    """
    checks position of p1 relative to vector p2->p3
    :param p1: shapely.geometry.Point
    :param p2: shapely.geometry.Point
    :param p3: shapely.geometry.Point
    :return: boolean
    """
    pos_value = (p2.x - p1.x) * (p3.y - p1.y) - (p3.x - p1.x) * (p2.y - p1.y)
    if pos_value < 0:
        return True
    else:
        return False


def classify_dir(segment, point):
    dist_along_seg = segment.project(point)
    projected_point = segment.interpolate(dist_along_seg)
    point_ahead = segment.interpolate(dist_along_seg + 100)
    if is_right(projected_point, point_ahead, point):
        return 0
    else:
        return 1


def calculate_distances_along_segments(segment_df, poi_df, info_column="on_segment"):
    poi_geoms = poi_df.geometry.to_list()
    segm_geoms = segment_df.geometry.to_list()
    on_which_segments = poi_df[info_column].to_list()
    dists = []
    for ij in range(0, len(poi_geoms)):
        dists.append(segm_geoms[on_which_segments[ij]].project(poi_geoms[ij]))

    return dists



