import streamlit as st
import math
import random
import pydeck as pdk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox

st.cache()
def show_empty_map(center_y,center_x):
	st.pydeck_chart(pdk.Deck(initial_view_state=pdk.ViewState(latitude = center_y, longitude = center_x, zoom=11)))
	return

st.cache()
def get_map():
	#Load graph from graphml
	G = ox.load_graphml(filename='greater_boston')
	return G

st.cache()
def get_gdfs():
	#Load gdfs, ensure that pickle protocol is set appropriately
	gdf_nodes = pd.read_pickle('nodes.pkl')
	gdf_edges = pd.read_pickle('edges.pkl')

	return gdf_nodes, gdf_edges

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




def source_to_source(s,dist,w1,w2,w3):
	#Load graph from graphml
	G = get_map()

	if (s==''):
		s = 'Boston College'

	#Get coordinates from addresses
	start_location = ox.geo_utils.geocode(s)
	start_coords = (start_location[0], start_location[1])

	#Snap addresses to graph nodes
	start_node = ox.get_nearest_node(G, start_coords)

	#Load gdfs, ensure that pickle protocol is set appropriately
	gdf_nodes, gdf_edges = get_gdfs()

	tree_counts = {}
	road_safety = {}
	lengths = {}

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
	for key in lengths.keys():
		optimized[key] = w1*(lengths[key]/75) + w2*(10/max(1,tree_counts[key])) + (w3/2)*(road_safety[key]/2)
	
	#We must set this after user preferences are input
	nx.set_edge_attributes(G, optimized, 'optimized')

	opt_return = optimized.copy()
	nx.set_edge_attributes(G, opt_return, 'opt_return')


#Step 1: Identify contender midpoints
	#Returns 2 dicts: first of end nodes:distance and second of end node:node path
	candidate_midpoints = nx.single_source_dijkstra(G, start_node, weight='length', cutoff=0.5*dist)
	candidate_paths = candidate_midpoints[0]

	#Contender midpoint nodes
	candidates = sorted([keys for keys,indices in candidate_paths.items()])[:min(len(candidate_paths),10)]

#Step 2: Sort contender midpoints by optimized weight
	candidate_weights = []

	for i in candidates:
		candidate_weights.append(nx.shortest_path_length(G, start_node, i, weight = 'optimized'))


	#The optimized midpoint
	midpoint = candidates[candidate_weights.index(max(candidate_weights))]

	#Path of nodes
	path = candidate_midpoints[1][midpoint]

#Step 3: Set edge weights to minimize backtracking
	for node in range(-1+len(path)):
		#print(path[node], path[node+1])
		G[path[node]][path[node+1]][0]['opt_return'] += 10000
		#print(G[path[node]][path[node+1]][0])


#Step 4: Get route back
	route_back = nx.shortest_path(G, midpoint, start_node, weight = 'opt_return')

	#print('test')
	#print(path)
	#print(route_back)

	#Maybe we add the midpoint twice here
	path += route_back


#Step 5: Reset edge weights


#Step 6: 
	loop1_start_lat, loop1_start_lon, loop1_dest_lat, loop1_dest_lon = nodes_to_lats_lons(gdf_nodes, path)
	loop2_start_lat, loop2_start_lon, loop2_dest_lat, loop2_dest_lon = nodes_to_lats_lons(gdf_nodes, route_back)
	
	#This finds the bounds of the final map to show based on the paths
	min_x, max_x, min_y, max_y = get_map_bounds(gdf_nodes, path, route_back)

	#Find the average lat/long to center the map
	center_x = 0.5*(max_x + min_x)
	center_y = 0.5*(max_y + min_y)

	#Move coordinates into dfs
	loop1_df = pd.DataFrame({'startlat':loop1_start_lat, 'startlon':loop1_start_lon, 'destlat': loop1_dest_lat, 'destlon':loop1_dest_lon})
	loop2_df = pd.DataFrame({'startlat':loop2_start_lat, 'startlon':loop2_start_lon, 'destlat': loop2_dest_lat, 'destlon':loop2_dest_lon})

	st.pydeck_chart(pdk.Deck(initial_view_state=pdk.ViewState(latitude = center_y, longitude = center_x, zoom=10), layers=[pdk.Layer('LineLayer',data=loop1_df, getSourcePosition = '[startlon, startlat]', getTargetPosition = '[destlon, destlat]', getColor = '[50,50,220]', getWidth = '5'), pdk.Layer('LineLayer',data=loop2_df, getSourcePosition = '[startlon, startlat]', getTargetPosition = '[destlon, destlat]', getColor = '[50,220,50]', getWidth = '5')] ))


	return

st.cache()
def source_to_dest(s,e,w1,w2,w3):
	#Load graph from graphml
	G = get_map()

	if (s==''):
		s = 'Boston College'
	if (e==''):
		e = '280 Summer St Boston'

	#Get coordinates from addresses
	start_location = ox.geo_utils.geocode(s)
	end_location = ox.geo_utils.geocode(e)
	start_coords = (start_location[0], start_location[1])
	end_coords = (end_location[0], end_location[1])

	#Snap addresses to graph nodes
	start_node = ox.get_nearest_node(G, start_coords)
	end_node = ox.get_nearest_node(G, end_coords)

	#Load gdfs, ensure that pickle protocol is set appropriately
	gdf_nodes, gdf_edges = get_gdfs()

	tree_counts = {}
	road_safety = {}
	lengths = {}

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
	for key in lengths.keys():
		optimized[key] = w1*(lengths[key]/75) + w2*(10/max(1,tree_counts[key])) + (w3/2)*(road_safety[key]/2)
	
	#We must set this after user preferences are input
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

	st.pydeck_chart(pdk.Deck(initial_view_state=pdk.ViewState(latitude = center_y, longitude = center_x, zoom=10), layers=[pdk.Layer('LineLayer',data=short_df, getSourcePosition = '[startlon, startlat]', getTargetPosition = '[destlon, destlat]', getColor = '[200,200,200]', getWidth = '5'), pdk.Layer('LineLayer',data=opt_df, getSourcePosition = '[startlon, startlat]', getTargetPosition = '[destlon, destlat]', getColor = '[50,50,220]', getWidth = '10') ] ))

	return




st.header("DogGo - The best walk for your best friend!")

#st.write(
"""
Step 1: Type your starting and final destinations\n
Step 2: Indicate your preferences\n
Step 3: Push 'Go!'\n\n
"""
#)

#User inputs source and destination

st.sidebar.markdown('Plan your walk')
select = st.sidebar.selectbox('Returning to where you are or going somewhere else?',('Returning','Going somewhere else'))
print(select, type(select))

if select=='Returning':
	dog_df = pd.read_csv('data/dogbreeds.csv')
	dogs = dict(zip(dog_df['Name'], dog_df['Exercise-Needs']))


	input1 = st.sidebar.text_input('Where are you now?')
	input2 = ''.join(input1)

	temp = ''
	temp = st.sidebar.selectbox('What type of dog do you have?', list(dogs.keys()))
	if temp != '':
		duration = st.sidebar.number_input('How many minutes for your walk?', step=5, value=10*int(1.2*dogs[temp]))

else:
	input1 = st.sidebar.text_input('Where are you starting?')
	input2 = st.sidebar.text_input('Where do you want to end up?')

print(select, type(select))



#Sliders for tree and car avoidance
w1 = st.sidebar.slider('How much of a detour could you endure? 10 = Big detour!', 0, 10, 5, key=1)
w2 = st.sidebar.slider('How much does your dog love trees? 10 = Love them!', 0, 10, 5, key=2)
w3 = st.sidebar.slider('Do you want to avoid busy roads? 10 = Avoid them!', 0, 10, 5, key=3)

submit = st.button('Optimize route - Go!', key=1)

if not submit:
	latitude = 42.358
	longitude = -71.085
	show_empty_map(latitude, longitude)
else:
	#print(input1, input2, w1, w2, w3)

	if input1==input2 or input2 == '':
		with st.spinner('Routing...'):
			source_to_source(input1, 150*duration, w1,w2,w3)
	else:
		source_to_dest(input1, input2, w1, w2, w3)

