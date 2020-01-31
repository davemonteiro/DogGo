import streamlit as st
import math
import random
import pydeck as pdk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox

def get_dist(lat1, lon1, lat2, lon2):
	return (111073*abs(lat1-lat2)+82850*abs(lon1-lon2))


def get_node_df(location):
	start_node_df = pd.DataFrame({'lat':[location[0]], 'lon':[location[1]]})
	icon_data = {"url": "https://img.icons8.com/plasticine/100/000000/marker.png", "width": 128, "height":128, "anchorY": 128}
	start_node_df['icon_data']= None
	for i in start_node_df.index:
	     start_node_df['icon_data'][i] = icon_data

	return start_node_df

def get_text_df(text, location):
	text_df = pd.DataFrame({'lat':[location[0]], 'lon':[location[1]], 'text':text})

	return text_df

def make_iconlayer(df):
	return pdk.Layer(
	    type='IconLayer',
	    data=df,
	    get_icon='icon_data',
	    get_size=4,
	    pickable=True,
	    size_scale=15,
	    get_position='[lon, lat]'
	)

def make_textlayer(df, color_array):
	return pdk.Layer(
	    type='TextLayer',
	    data=df,
	    get_text='text',
	    get_size=4,
	    pickable=True,
	    size_scale=6,
	    getColor = color_array,
	    get_position='[lon, lat]'
	)

def make_linelayer(df, color_array):
	return pdk.Layer(
	    type='LineLayer',
	    data=df,
	    getSourcePosition = '[startlon, startlat]',
	    getTargetPosition = '[destlon, destlat]',
	    getColor = color_array,
	    getWidth = '5'
	)

@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def get_map():
	print('getmap')
	#Load graph from graphml
	G = ox.load_graphml(filename='greater_boston')
	return G

@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def get_gdfs():
	print('getgdf')
	#Load gdfs, ensure that pickle protocol is set appropriately
	gdf_nodes = pd.read_pickle('nodes.pkl')
	gdf_edges = pd.read_pickle('edges.pkl')

	return gdf_nodes, gdf_edges

def get_map_bounds(gdf_nodes, route1, route2):
	#Crop target image by finding bounding box of paths + border
	max_x = -1000
	min_x = 1000
	max_y = -1000
	min_y = 1000

	for i in (route1 + route2):
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

############################################################################

def source_to_source(G, gdf_nodes, gdf_edges, s, dist, w1, w3):
	if (s==''):
		s = '280 Summer St Boston'

	#Get coordinates from addresses
	start_location = ox.geo_utils.geocode(s)
	start_coords = (start_location[0], start_location[1])

	#Snap addresses to graph nodes
	start_node = ox.get_nearest_node(G, start_coords)

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
		temp = int(lengths[key]/75 + 200/max(1,tree_counts[key]))
		if w3:
			temp += 100*(road_safety[key])
		optimized[key] = temp
	
	#We must set this after user preferences are input
	nx.set_edge_attributes(G, optimized, 'optimized')

	opt_return = optimized.copy()
	nx.set_edge_attributes(G, opt_return, 'opt_return')

	if w1:#Find a park close to 1/2 distance
		parks = pd.read_csv('data/parks.csv')
		dists = []
		for row in parks.itertuples():
			dists.append(get_dist(getattr(row,'lat'),getattr(row,'lon'), start_location[0], start_location[1]))

		#Find a park closest to 0.5*the distance you'd like to travel

		dists = [abs(x-0.5*dist) for x in dists]

		index = dists.index(min(dists))
		midpoint_coords = (parks.iloc[index]['lat'], parks.iloc[index]['lon'])

		#Node of chosen park
		midpoint = ox.get_nearest_node(G, midpoint_coords)
		
		#Path to chosen park
		path = nx.shortest_path(G, start_node, midpoint, weight = 'optimized')
		text_layer = make_textlayer(get_text_df(parks.iloc[index]['Name'], midpoint_coords), '[0,0,0]')

	else:#The midpoint has to be calculated separately
		#Step 1: Identify candidate midpoints
		#Returns 2 dicts: first of end nodes:distance and second of end node:node path
		candidate_midpoints = nx.single_source_dijkstra(G, start_node, weight='length', cutoff=0.5*dist)
		candidate_paths = candidate_midpoints[0]

		#Contender midpoint nodes
		midpoint_nodes = [k for (k, v) in candidate_paths.items() if v >= 0.95 * max(candidate_paths.values())]

		#Step 2: Sort contender midpoints by optimized weight
		candidate_weights = []
		for i in midpoint_nodes:
			candidate_weights.append(nx.shortest_path_length(G, start_node, i, weight = 'optimized'))

		#The optimized midpoint
		midpoint = midpoint_nodes[candidate_weights.index(max(candidate_weights))]

		#Path of nodes
		path = nx.shortest_path(G, start_node, midpoint, weight = 'optimized')

		#Dummy code
		text_layer = make_textlayer(get_text_df('', start_coords), '[255,255,255]')

	#Step 3: Set edge weights to avoid backtracking
	for node in range(-1+len(path)):
		G[path[node+1]][path[node]][0]['opt_return'] += 100000

	#Step 4: Get route back
	route_back = nx.shortest_path(G, midpoint, start_node, weight = 'opt_return')

	#Step 5: Reset edge weights
	for node in range(-1+len(path)):
		G[path[node+1]][path[node]][0]['opt_return'] -= 100000

	#Step 6: Plot route
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

	#Add map marker icon at start 
	start_node_df = get_node_df(start_location)
	outbound_layer = make_linelayer(loop1_df, '[150,150,220]')
	inbound_layer = make_linelayer(loop2_df, '[220,50,50]')
	icon_layer = make_iconlayer(start_node_df)

	st.pydeck_chart(pdk.Deck(map_style="mapbox://styles/mapbox/light-v9", initial_view_state=pdk.ViewState(latitude = center_y, longitude = center_x, zoom=13), layers=[text_layer, outbound_layer, inbound_layer, icon_layer]))
	
	return

################################################################################

def source_to_dest(G, gdf_nodes, gdf_edges, duration, s, e, w2, w3):
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
		temp = int(lengths[key])
		if w2:
			temp += int(100/max(1,tree_counts[key]))
		if w3:
			temp += int(100*(road_safety[key]))
		optimized[key] = temp

	#We must set this after user preferences are input
	nx.set_edge_attributes(G, optimized, 'optimized')

	#These are lists of the nodes that the routes take
	optimized_route = nx.shortest_path(G, start_node, end_node, weight = 'optimized')
	
	#This finds the bounds of the final map to show based on the paths
	min_x, max_x, min_y, max_y = get_map_bounds(gdf_nodes, optimized_route, optimized_route)

	#These are lists of origin/destination coords of the paths that the routes take
	opt_start_lat, opt_start_lon, opt_dest_lat, opt_dest_lon = nodes_to_lats_lons(gdf_nodes, optimized_route)

	#Find the average lat/long to center the map
	center_x = 0.5*(max_x + min_x)
	center_y = 0.5*(max_y + min_y)

	#Move coordinates into dfs
	opt_df = pd.DataFrame({'startlat':opt_start_lat, 'startlon':opt_start_lon, 'destlat': opt_dest_lat, 'destlon':opt_dest_lon})

	if (w2 or w3):
		shortest_route = nx.shortest_path(G, start_node, end_node, weight = 'length')
		short_start_lat, short_start_lon, short_dest_lat, short_dest_lon = nodes_to_lats_lons(gdf_nodes, shortest_route)
		short_df = pd.DataFrame({'startlat':short_start_lat, 'startlon':short_start_lon, 'destlat': short_dest_lat, 'destlon':short_dest_lon})
		short_layer = make_linelayer(short_df, '[200,200,200]')

	start_node_df = get_node_df(start_location)
	icon_layer = make_iconlayer(start_node_df)
	optimized_layer = make_linelayer(opt_df, '[50,220,50]')

	if (w2 or w3):
		st.pydeck_chart(pdk.Deck(initial_view_state=pdk.ViewState(latitude = center_y, longitude = center_x, zoom=13), layers=[short_layer, optimized_layer, icon_layer]))

	else:
		st.pydeck_chart(pdk.Deck(map_style="mapbox://styles/mapbox/light-v9", initial_view_state=pdk.ViewState(latitude = center_y, longitude = center_x, zoom=13), layers=[optimized_layer, icon_layer]))

	
	return

########################################################################################

G = get_map()
gdf_nodes, gdf_edges = get_gdfs()

st.header("DogGo - planning the best walk for your best friend!")

#User inputs source and destination

st.sidebar.markdown('Plan your walk')
select = st.sidebar.selectbox('Returning to where you are or going somewhere else?',('Returning','Going somewhere else'))

if select=='Returning':
	dog_df = pd.read_csv('data/dogbreeds.csv')
	dogs = dict(zip(dog_df['Name'], dog_df['Exercise-Needs']))

	input1 = st.sidebar.text_input('Where are you now?')
	input2 = ''.join(input1)

	temp = ''
	temp = st.sidebar.selectbox('What type of dog do you have?', list(dogs.keys()))
	if temp != '':
		duration = st.sidebar.number_input('How many minutes for your walk?', step=5, value=10*int(1.2*dogs[temp]))

	w1 = st.sidebar.checkbox('Take me to a park!', value=False , key=1)
	w3 = st.sidebar.checkbox('Keep me away from busy streets.', value=False , key=3)

else:
	input1 = st.sidebar.text_input('Where are you starting?')
	input2 = st.sidebar.text_input('Where do you want to end up?')

	w2 = st.sidebar.checkbox('Take me via a scenic route.', value=False , key=2)
	w3 = st.sidebar.checkbox('Keep me away from busy streets.', value=False , key=3)


submit = st.button('Calculate route - Go!', key=1)

if not submit:
	latitude = 42.358
	longitude = -71.085
	st.pydeck_chart(pdk.Deck(map_style="mapbox://styles/mapbox/light-v9", initial_view_state=pdk.ViewState(latitude = latitude, longitude = longitude, zoom=11)))
else:
	if input1==input2 or input2 == '':
		with st.spinner('Routing...'):
			source_to_source(G, gdf_nodes, gdf_edges, input1, 100*duration, w1, w3)
	else:
		duration = 20
		source_to_dest(G, gdf_nodes, gdf_edges, duration, input1, input2, w2, w3)

