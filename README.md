# Highcharge - a fast-charging allocation model for high-level road networks

## Overview

The scripts in this depository were developed in course of the work described in Golab et al., 2022 <sup>[1](#myfootnote1)</sup>.

__Abstract:__ Given the on-going transformation of the transport sector toward electrification, expansion of the current charging infrastructure is essential to meet future charging demands. The lack of fast-charging infrastructure along highways and motorways is a particular obstacle for long-distance travel with battery electric vehicles (BEVs). In this context, we propose a charging infrastructure allocation model that allocates and sizes fast-charging stations along high-level road networks while minimizing the costs for infrastructure investment. The modeling framework is applied to the Austrian highway and motorway network, and the needed expansion of the current fast-charging infrastructure in place is modeled under different future scenarios for 2030. Within these, the share of BEVs in the car fleet, developments in BEV technology and road traffic load changing in the face of future modal shift effects are altered. In particular, we analyze the change in the requirements for fast-charging infrastructure in response to enhanced driving range and growing BEV fleet. The results indicate that improvements in the driving range of BEVs will have limited impact and hardly affect future costs of the expansion of the fast-charging infrastructure. On the contrary, the improvements in the charging power of BEVs have the potential to reduce future infrastructure costs.

We refer to the manuscript for explanations on the validation process, demand calculation and optimization model.

The three core scripts of this work are:

- _data_pre-pre-processing_Austrian_highwaynetwork.py_: The calculation procedures conducted in this script are referenced in Section **Demand calculation**.
- _demand&#95;calculation.py_: Using this script, charging demand for all nodes in the highway network is obtained. This charging demand is for a day and given by a car fleet with 100% of electric vehicles. 
- _parameter_calculation.py_: Evaluation of input parameters for scenario calculations. 
- _validation.py_: Recreation of currently required fast-charging infrastructure. 
- _analysis.py_: This script encompasses all calculations for the analysis presented during in the Results section of the manuscript.


## Complementary information - details on pre-processing

Geographic data on Austrian highways and motorways, and service areas (also referred to as "rest areas" in the scripts) where retrieved from OpenStreetMap (OSM) data using [https://www.geofabrik.de/de/data/download.html]. All way instances with keys *highway='motorway'* and *highway = 'trunk'* were retrieved and used for as the shape of the high-level road network in further processing. This shape was carefully edited by hand using the software ArcGIS Pro. During this edit, gaps in line segments were filled and redundant lines were deleted, in order to finally obtain a clean shape representation of the road network, moreover, including the representation of both driving directions through parallel line shapes. Areas and way instances representing service area geometries were retrieved using keywords *highway='services'* and *highway='rest_area'*. 

Information on existing service area and their name, accessibility and relative position on highways and motorways were retrieved from [https://www.asfinag.at/parken-rasten/rastanlagensuche/]. Based on highway geometries, the service area information was matched with the OSM geometries. The geometry of a service area was then simplified to an areas centroid. Further, data on traffic counts was retrieved from [https://www.asfinag.at/verkehr-sicherheit/verkehrszahlung/]. As the exact geographic positions of traffic counters is unknown, but only the highway section on which the traffic counter is installed, the absolute traffic counter positions were approximated based on the relative distances on highways defined in the files from [https://www.asfinag.at/verkehr-sicherheit/verkehrszahlung/]. 


Based on (a) road network geometry and (b) service area geometries, the following four important data blocks are obtained:

- **_Intersections (junctions)_**: From the road network geometry, junctions of highways/motorways are retrieved. 
- **_Segments_**: The shape representing the road network is splitteds into segments using the obtained intersections. Therefore, segments are sections of the road network between intersections and ending points of the road network. Each segment's geometry is a shapely.geometry.LineString object which is West-East oriented. Based on these driving direction (0,1), dirving direction 0 is West-East and driving direction 1 is East-West along a segment. 
- **_Accessebility of service areas_**: The driving direction from which service areas are able to be accesses needs to be specified according to the driving directions set for the segments. Therefore, the relative position of the centroids representing the service areas is determined. This is done for all service areas which are exclusively accessible from only one direction.  
- **_Positioning of traffic counters_**: Based on the obtained segment geometries, the relative positions of all traffic counters is determined and, in particular, on which segments the traffic lie. 
- **_Sequence of node visits_**: In order to account for traffic flow movement, a sequence of node visits is obtained, for each group of nodes grouped by the segment they lie on and the driving direction they are accessible from. This is pre-calculated in order to later reduce overall calculation time of the optimization. 


This information is obtained by running *data_pre-pre-processing_Austrian_highwaynetwork.py*. In reference to the graph-based representation of this model, the following is important to note here: Vertices in the graph represent intersections, ending points of the road network and service areas (also referred to as nodes), while edges represent road connections between these vertices. 


# References 


<a name="myfootnote1">1</a>: Golab, A.; Zwickl-Bernhard, S.; Auer, H., 2022. *Minimum-cost fast-charging infrastructure planning for electric vehicles along the Austrian high-level road network*, submitted to *Energies*, Manuscript No. energies-1632966 , under review




