import streamlit as st
import math
import random
import pydeck as pdk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
import networkx as nx
import osmnx as ox

def get_map_bounds(gdf_nodes, shortest_route, optimized_route):
	#Crop target image by finding bounding box of paths + border
	max_x = -1000
	min_x = 1000
	max_y = -1000
	min_y = 1000

	for i in (shortest_route + optimized_route):
		row = gdf_nodes.loc[i]
		temp_x = row['x']
		temp_y = row['y']

	if temp_x > max_x:
		max_x = temp_x
	if temp_x < min_x:
		min_x = temp_x
	if temp_y > max_y:
		max_y = temp_y
	if temp_y < min_y:
		min_y = temp_y

	#Add border
	min_x -= 0.005
	min_y -= 0.005
	max_x += 0.005
	max_y += 0.005

	return min_x, max_x, min_y, max_y


def nodes_to_lats_lons(nodes, list_of_path_nodes):
	source_lats = []
	source_lons = []
	dest_lats = []
	dest_lons = []
   	
	for i in range(0,len(list_of_path_nodes)-2):
		source_lats.append(nodes.loc[list_of_path_nodes[i]]['y'])
		source_lons.append(nodes.loc[list_of_path_nodes[i]]['x'])
		dest_lats.append(nodes.loc[list_of_path_nodes[i+1]]['y'])
		dest_lons.append(nodes.loc[list_of_path_nodes[i+1]]['x'])

	return (source_lats, source_lons, dest_lats, dest_lons)

st.cache()
def get_coords(s,e,w1,w2,w3):
	if (s==''):
		s = 'Boston College'
	if (e==''):
		e = '280 Summer St Boston'

	#Turn addresses into coordinates - ultimately user inputs
	#If many requests, user_agent should be email address
	print('1')
	geolocator = Nominatim(user_agent='brainiacevad@gmail.com')
	start_location = geolocator.geocode(s)
	end_location = geolocator.geocode(e)
	start_coords = (start_location.latitude, start_location.longitude)
	end_coords = (end_location.latitude, end_location.longitude)

	#Load graph from graphml
	G = ox.load_graphml(filename='greater_boston')

	#Snap addresses to graph nodes
	start_node = ox.get_nearest_node(G, start_coords)
	end_node = ox.get_nearest_node(G, end_coords)

	#Load gdfs
	gdf_nodes = pd.read_pickle('nodes.pkl',protocol=4)
	gdf_edges = pd.read_pickle('edges.pkl',protocol=4)

	tree_counts = {}
	road_safety = {}
	lengths = {}
	print('22')
	#Set each edge's tree weight as the average of the tree weights of the edge's vertices
	for row in gdf_edges.itertuples():
		u = getattr(row,'u')
		v = getattr(row,'v')
		key = getattr(row, 'key')
		tree_count = getattr(row, 'trees')
		safety_score = getattr(row, 'safety')
		length = getattr(row, 'length')

		tree_counts[(u,v,key)] = tree_count
		road_safety[(u,v,key)] = safety_score
		lengths[(u,v,key)] = length

	#optimized is weighted combo of normal length, tree counts, and road safety
	optimized = {}

	#Still need to rescale lengths and safety scores
	for key in lengths.keys(): optimized[key] = w1*(lengths[key]/75) + w2*(10/max(1,tree_counts[key])) + (w3/2)*(road_safety[key]/2)
	print('333')
	nx.set_edge_attributes(G, tree_counts, 'numtrees')
	nx.set_edge_attributes(G, optimized, 'optimized')

	#These are lists of the nodes that the routes take
	shortest_route = nx.shortest_path(G, start_node, end_node, weight = 'length')
	optimized_route = nx.shortest_path(G, start_node, end_node, weight = 'optimized')
	

	#This finds the bounds of the final map to show based on the paths
	min_x, max_x, min_y, max_y = get_map_bounds(gdf_nodes, shortest_route, optimized_route)

	#These are lists of origin/destination coords of the paths that the routes take
	short_start_lat, short_start_lon, short_dest_lat, short_dest_lon = nodes_to_lats_lons(gdf_nodes, shortest_route)
	opt_start_lat, opt_start_lon, opt_dest_lat, opt_dest_lon = nodes_to_lats_lons(gdf_nodes, optimized_route)

	#Find the average lat/long to center the map
	center_x = 0.5*(max_x + min_x)
	center_y = 0.5*(max_y + min_y)

	#Move coordinates into dfs
	short_df = pd.DataFrame({'startlat':short_start_lat, 'startlon':short_start_lon, 'destlat': short_dest_lat, 'destlon':short_dest_lon})
	opt_df = pd.DataFrame({'startlat':opt_start_lat, 'startlon':opt_start_lon, 'destlat': opt_dest_lat, 'destlon':opt_dest_lon})

	print(short_df.head(5))

	st.pydeck_chart(pdk.Deck(initial_view_state=pdk.ViewState(latitude = center_y, longitude = center_x, zoom=11), layers=[pdk.Layer('LineLayer',data=short_df, getSourcePosition = '[startlon, startlat]', getTargetPosition = '[destlon, destlat]', getColor = '[200,200,200]', getWidth = '5'), pdk.Layer('LineLayer',data=opt_df, getSourcePosition = '[startlon, startlat]', getTargetPosition = '[destlon, destlat]', getColor = '[50,50,220]', getWidth = '10') ] ))

	return

latitude = 42.3
longitude = -71.05

st.header("DogGo - The best walking route for your best friend!")

st.write(
"""
Step 1: Type your starting and final destinations\n
Step 2: Indicate your preferences\n
Step 3: Push 'Go!'\n\n
"""
)

#User inputs source and destination
input1 = st.sidebar.text_input('Where are you starting off?')
input2 = st.sidebar.text_input('Where are you finishing?')

#Sliders for tree and car avoidance
w1 = st.sidebar.slider('How much of a detour could you endure? 10 = Big detour!', 0, 10, 5, key=1)
w2 = st.sidebar.slider('How much does your dog love trees? 10 = Love them!', 0, 10, 5, key=2)
w3 = st.sidebar.slider('Do you want to avoid busy roads? 10 = Avoid them!', 0, 10, 5, key=3)

submit2 = st.button('Optimize route - Go!', key=1)
if submit2:
	print(input1, input2, w1, w2, w3)
	get_coords(input1, input2, w1, w2, w3)
	print('done')
