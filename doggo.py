import streamlit as st
import pydeck as pdk
import pandas as pd
import networkx as nx
import osmnx as ox

def get_node_df(location):
	#Inputs: location as tuple of coords (lat, lon)
	#Returns: 1-line dataframe to display an icon at that location on a map

	#Location of Map Marker icon
	icon_data = {
		"url": "https://img.icons8.com/plasticine/100/000000/marker.png", 
		"width": 128, 
		"height":128, 
		"anchorY": 128}
	
	start_node_df = pd.DataFrame({'lat':[location[0]], 'lon':[location[1]], 'icon_data': [icon_data]})

	return start_node_df

def get_text_df(text, location):
	#Inputs: text to display and location as tuple of coords (lat, lon)
	#Returns: 1-line dataframe to display text at that location on a map

	#Missing background so text is more visible*********
	text_df = pd.DataFrame({'lat':[location[0]], 'lon':[location[1]], 'text':text})
	return text_df

def make_iconlayer(df):
	#Inputs: df with [lat, lon, icon_data]
	#Returns: pydeck IconLayer

	return pdk.Layer(
	    type='IconLayer',
	    data=df,
	    get_icon='icon_data',
	    get_size=4,
	    pickable=True,
	    size_scale=15,
	    get_position='[lon, lat]')

def make_textlayer(df, color_array):
	#Inputs: df with [lat, lon, text] and font color as str([R,G,B]) - yes '[R,G,B]'
	#Returns: pydeck TextLayer

	return pdk.Layer(
	    type='TextLayer',
	    data=df,
	    get_text='text',
	    get_size=4,
	    pickable=True,
	    size_scale=6,
	    getColor = color_array,
	    get_position='[lon, lat]')

def make_linelayer(df, color_array):
	#Inputs: df with [startlat, startlon, destlat, destlon] and font color as str([R,G,B]) - yes '[R,G,B]'
	#Plots lines between each line's [startlon, startlat] and [destlon, destlat]
	#Returns: pydeck LineLayer

	return pdk.Layer(
	    type='LineLayer',
	    data=df,
	    getSourcePosition = '[startlon, startlat]',
	    getTargetPosition = '[destlon, destlat]',
	    getColor = color_array,
	    getWidth = '5')

@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def get_map():
	#Returns: map as graph from graphml
	#Cached by Streamlit

	G = ox.load_graphml(filename='greater_boston')
	return G

@st.cache(suppress_st_warning=True, allow_output_mutation=True, show_spinner=False)
def get_gdfs():
	#Returns: nodes and edges from pickle
	#Cached by Streamlit

	gdf_nodes = pd.read_pickle('data/nodes.pkl')
	gdf_edges = pd.read_pickle('data/edges.pkl')
	return gdf_nodes, gdf_edges

def get_dist(lat1, lon1, lat2, lon2):
	#Inputs: 4 integers, latitudes and longitudes from point 1 followed by point 2
	#Returns: birds-eye distance between them

	#Coefficients are distance across 1degree Lat/Long
	#Precalculated for Boston lat and lon
	
	return (111073*abs(lat1-lat2)+82850*abs(lon1-lon2))

def get_map_bounds(gdf_nodes, route1, route2):
	#Inputs: node df, and two lists of nodes along path
	#Returns: Coordinates of smallest rectangle that contains all nodes

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

def nodes_to_lats_lons(nodes, path_nodes):
	#Inputs: node df, and list of nodes along path
	#Returns: 4 lists of source and destination lats/lons for each step of that path for LineLayer
	#S-lon1,S-lat1 -> S-lon2,S-lat2; S-lon2,S-lat2 -> S-lon3,S-lat3...

	source_lats = []
	source_lons = []
	dest_lats = []
	dest_lons = []
   
	for i in range(0,len(path_nodes)-2):
		source_lats.append(nodes.loc[path_nodes[i]]['y'])
		source_lons.append(nodes.loc[path_nodes[i]]['x'])
		dest_lats.append(nodes.loc[path_nodes[i+1]]['y'])
		dest_lons.append(nodes.loc[path_nodes[i+1]]['x'])

	return (source_lats, source_lons, dest_lats, dest_lons)

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def set_walking_rate(rate):
	#Inputs: walking rate
	#Returns: walking rate (cached for streamlit)
	return int(rate)

############################################################################

def find_turnaround_point(G, start_node, dist):
	#Inputs: G, index of start_node, distance of walk
	#Returns: index of midpoint_node, list of indices of nodes along path to midpoint

	#Step 1: Identify candidate midpoints within factor1*dist
	factor1 = 0.4
	candidate_midpoints = nx.single_source_dijkstra(G, start_node, weight = 'length', cutoff = factor1*dist)
	#Returns 2 dicts: first of end nodes:distance and second of end node:node path

	#Dict of node index:distance from start node
	candidate_paths = candidate_midpoints[0]

	#Consider all nodes that are within factor2 of target length
	factor2 = 0.95
	farthest_node_considered = max(candidate_paths.values())
	midpoint_nodes = [k for (k, v) in candidate_paths.items() if v >= factor2 * farthest_node_considered]

	#Step 2: Sort contender midpoints by optimized weight
	candidate_weights = []
	for i in midpoint_nodes:
		candidate_weights.append(nx.shortest_path_length(G, start_node, i, weight = 'optimized'))

	#The optimized midpoint
	midpoint = midpoint_nodes[candidate_weights.index(max(candidate_weights))]

	#Path of nodes
	path = nx.shortest_path(G, start_node, midpoint, weight = 'optimized')

	return midpoint, path

def find_park(G, parks, start_node, start_location, dist):
	#Inputs: G, parks df, index of start_node, coords of start_node, distance of walk
	#Returns: index of midpoint_node, list of indices of nodes along path to midpoint, index of chosen park

	#Find a park close to 1/2 distance
	dists = []
	for row in parks.itertuples():
		dists.append(get_dist(getattr(row,'lat'),getattr(row,'lon'), start_location[0], start_location[1]))

	#Find a park closest to factor*the distance you'd like to travel
	#This might benefit from tinkering
	factor = 0.5
	dists = [abs(x-factor*dist) for x in dists]

	#Identify node closest to coordinates of chosen park
	index = dists.index(min(dists))
	midpoint_coords = (parks.iloc[index]['lat'], parks.iloc[index]['lon'])
	midpoint = ox.get_nearest_node(G, midpoint_coords)
	
	#Path to chosen park
	path = nx.shortest_path(G, start_node, midpoint, weight = 'optimized')
	
	return midpoint, path, index

def source_to_source(G, gdf_nodes, gdf_edges, s, dist, w1, w2):
	#Inputs: Graph, nodes, edges, source, distance to walk, w1 bool = route to park?, w2 bool = avoid busy roads
	
	#Set default source to Insight Offices
	if s == '':
		#No address, default to Insight
		st.write('Source address not found, defaulting to Insight Offices')
		s = '280 Summer St Boston'
		start_location = ox.geo_utils.geocode(s)
	else:
		try:
			start_location = ox.geo_utils.geocode(s + ' Boston')
		except:
			#No address found, default to Insight
			st.write('Source address not found, defaulting to Insight Offices')
			s = '280 Summer St Boston'
			start_location = ox.geo_utils.geocode(s)

	#Get coordinates from start address and snap to node
	start_coords = (start_location[0], start_location[1])
	start_node = ox.get_nearest_node(G, start_coords)

	#Calculate new edge weights
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
	for key in lengths.keys():
		temp = int(lengths[key])
		temp += int(200/max(1,tree_counts[key]))
		if w2:
			temp += int(100*(road_safety[key]))
		optimized[key] = temp
	
	#Generate new edge attributes - depending on user prefs
	nx.set_edge_attributes(G, optimized, 'optimized')

	if w1:
		#Get coords of park and path to it
		parks = pd.read_csv('data/parks.csv')
		midpoint, path, park_index = find_park(G, parks, start_node, start_location, dist)
		
		#Display name of park on the map
		text_layer = make_textlayer(get_text_df(parks.iloc[park_index]['Name'], (parks.iloc[park_index]['lat'], parks.iloc[park_index]['lon'])), '[0,0,0]')

	else:
		#Get coords of midpoint and path to it
		midpoint, path = find_turnaround_point(G, start_node, dist)

		#Dummy code because we are not outputting text
		#Or maybe we are - think about what to output here*******
		text_layer = make_textlayer(get_text_df('', start_coords), '[255,255,255]')

	#Step 3: Set edge weights to avoid backtracking
	for node in range(-1+len(path)):
		G[path[node+1]][path[node]][0]['optimized'] += 200

	#Step 4: Get route back
	route_back = list(reversed(nx.shortest_path(G, midpoint, start_node, weight = 'optimized')))

	#Step 5: Reset edge weights
	for node in range(-1+len(path)):
		G[path[node+1]][path[node]][0]['optimized'] -= 200

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

	st.pydeck_chart(pdk.Deck(
		map_style="mapbox://styles/mapbox/light-v9", 
		initial_view_state=pdk.ViewState(latitude = center_y, longitude = center_x, zoom = 13, max_zoom = 15, min_zoom = 12),
		layers=[text_layer, outbound_layer, inbound_layer, icon_layer]))
	return

############################################################################

def find_midpoint(G, start_node, end_node, dist):
	#Inputs: G, index of start_node, index of end_node, distance of walk
	#Returns: index of midpoint_node

	#Step 1: Identify candidate midpoints
	#Lists of nodes reachable from node within factor1*dist
	factor1 = 0.5
	start_midpoints = nx.single_source_dijkstra(G, start_node, weight='length', cutoff=factor1*dist)
	end_midpoints = nx.single_source_dijkstra(G, end_node, weight='length', cutoff=factor1*dist)
	#ssdijkstra returns 2 dicts: first of end nodes:distance and second of end node:node path

	candidate_midpoints = list(start_midpoints[0].keys())
	total_dists = [start_midpoints[0][i]+end_midpoints[0][i] for i in candidate_midpoints if i in end_midpoints[0]]
	#candidate_midpoints = [node for node in candidate_midpoints if (start_midpoints[0][node] + nx.shortest_path_length(G, node, end_node, weight = 'length')) < factor2 * dist]
	
	#Consider midpoints such that shortest path is within factor2 of target dist
	factor2 = 0.8
	midpoint_nodes = [k for (k, v) in zip(candidate_midpoints, total_dists) if (v >= factor2*dist and v <= (1/factor2)*dist)]
	
	#Step 2: Sort contender midpoints by optimized weight
	candidate_weights = []
	for i in midpoint_nodes:
		candidate_weights.append(nx.shortest_path_length(G, end_node, i, weight = 'optimized') + nx.shortest_path_length(G, start_node, i, weight = 'optimized'))
	
	#The optimized midpoint
	midpoint = candidate_midpoints[candidate_weights.index(max(candidate_weights))]
	return midpoint

def source_to_dest(G, gdf_nodes, gdf_edges, s, e, dist, pace, w2):
	#Inputs: Graph, nodes, edges, source, end, distance to walk, pace = speed, w2 bool = avoid busy roads

	if s == '':
		#No address, default to Insight
		st.write('Source address not found, defaulting to Insight Offices')
		s = '280 Summer St Boston'
		start_location = ox.geo_utils.geocode(s)
	else:
		try:
			start_location = ox.geo_utils.geocode(s + ' Boston')
		except:
			#No address found, default to Insight
			st.write('Source address not found, defaulting to Insight Offices')
			s = '280 Summer St Boston'
			start_location = ox.geo_utils.geocode(s)

	if e == '':
		#No address, default to Fenway Park
		st.write('Destination address not found, defaulting to Fenway Park')
		e = 'Fenway Park Boston'
		end_location = ox.geo_utils.geocode(e)
	else:
		try:
			end_location = ox.geo_utils.geocode(e + ' Boston')
		except:
			#No address found, default to Insight
			st.write('Destination address not found, defaulting to Fenway Park')
			e = 'Fenway Park Boston'
			end_location = ox.geo_utils.geocode(e)

	#Get coordinates from addresses
	start_coords = (start_location[0], start_location[1])
	end_coords = (end_location[0], end_location[1])

	#Snap addresses to graph nodes
	start_node = ox.get_nearest_node(G, start_coords)
	end_node = ox.get_nearest_node(G, end_coords)

	#Calculate new edge weights
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
	for key in lengths.keys():
		temp = int(lengths[key])
		temp += int(200/max(1,tree_counts[key]))
		if w2:
			temp += int(100*(road_safety[key]))
		optimized[key] = temp

	#Generate new edge attributes - depending on user prefs
	nx.set_edge_attributes(G, optimized, 'optimized')

	#We need to make sure that dist is at least the length of the shortest path
	min_dist = nx.shortest_path_length(G, start_node, end_node, weight = 'length')

	if dist < 1.2*min_dist:
		dist = min_dist
		st.write('This walk will probably take a bit longer - approximately ' + str(int(dist/pace)) + ' minutes total.')
		#We are in a rush, take the shortest path
		optimized_route = nx.shortest_path(G, start_node, end_node, weight = 'length')

	else: #dist > 1.2*min_dist
		#Try a new strategy of identifying a midpoint during a long walk.
		#Midpoint should be approx 1/2 distance to start and end nodes
		midpoint = find_midpoint(G, start_node, end_node, dist)

		#Path of nodes
		path1 = nx.shortest_path(G, start_node, midpoint, weight = 'optimized')

		#Step 3: Set edge weights to avoid backtracking
		for node in range(-1+len(path1)):
			G[path1[node+1]][path1[node]][0]['optimized'] += 200

		#Step 4: Get route back
		path2 = nx.shortest_path(G, midpoint, end_node, weight = 'optimized')

		#Step 5: Reset edge weights
		for node in range(-1+len(path1)):
			G[path1[node+1]][path1[node]][0]['optimized'] -= 200

		optimized_route = path1 + path2[1:]

	shortest_route = nx.shortest_path(G, start_node, end_node, weight = 'length')
	short_start_lat, short_start_lon, short_dest_lat, short_dest_lon = nodes_to_lats_lons(gdf_nodes, shortest_route)
	short_df = pd.DataFrame({'startlat':short_start_lat, 'startlon':short_start_lon, 'destlat': short_dest_lat, 'destlon':short_dest_lon})
	short_layer = make_linelayer(short_df, '[200,200,200]')
	
	#This finds the bounds of the final map to show based on the paths
	min_x, max_x, min_y, max_y = get_map_bounds(gdf_nodes, shortest_route, optimized_route)

	#These are lists of origin/destination coords of the paths that the routes take
	opt_start_lat, opt_start_lon, opt_dest_lat, opt_dest_lon = nodes_to_lats_lons(gdf_nodes, optimized_route)

	#Find the average lat/long to center the map
	center_x = 0.5*(max_x + min_x)
	center_y = 0.5*(max_y + min_y)

	#Move coordinates into dfs
	opt_df = pd.DataFrame({'startlat':opt_start_lat, 'startlon':opt_start_lon, 'destlat': opt_dest_lat, 'destlon':opt_dest_lon})

	start_node_df = get_node_df(start_location)
	icon_layer = make_iconlayer(start_node_df)
	optimized_layer = make_linelayer(opt_df, '[50,220,50]')

	optimized_length = sum(ox.geo_utils.get_route_edge_attributes(G, optimized_route, attribute = 'length'))
	optimized_trees = sum([int(x) for x in ox.geo_utils.get_route_edge_attributes(G, optimized_route, attribute = 'numtrees')])
	short_length = sum(ox.geo_utils.get_route_edge_attributes(G, shortest_route, attribute = 'length'))
	short_trees = sum([int(x) for x in ox.geo_utils.get_route_edge_attributes(G, shortest_route, attribute = 'numtrees')])

	st.pydeck_chart(pdk.Deck(
		initial_view_state=pdk.ViewState(latitude = center_y, longitude = center_x, zoom=13, max_zoom = 15, min_zoom = 12), 
		layers=[short_layer, optimized_layer, icon_layer]))

	st.write('The optimized path is ' + '{0:.2f}'.format(optimized_length/short_length) + 'X longer than the shortest path, but it is ' + '{0:.2f}'.format(optimized_trees/short_trees) + ' times greener.')

	return

############################################################################

G = get_map()
gdf_nodes, gdf_edges = get_gdfs()

#Main
st.header("DogGo - the best walk for your best friend!")
st.header("")
st.markdown('Plan your walk:')

input1 = st.text_input('Where will the walk begin?')
input2 = ''

select = st.selectbox(
	'Going out for a walk or traveling somewhere else?',
	('Going out for a walk','Traveling somewhere else'))

if select=='Traveling somewhere else':
	input2 = st.text_input('Where will the walk end?')

dog_df = pd.read_csv('data/dogbreeds.csv')
dogs = dict(zip(dog_df['Name'], zip(dog_df['Exercise-Needs'], dog_df['Height']) ))

temp = st.selectbox(
		'What type of dog do you have?', 
		list(dogs.keys()))

#Initial estimate for duration depends on dog breed
duration = st.number_input('How much time (minutes) do you have for this walk?', step = 5, value = 10*int(1.2*dogs[temp][0]))

#Estimate for walking speed depends on dog breed
#Vary by ~20% depending on height
walking_rate = set_walking_rate(80 - (15*(1/14)*(14-dogs[temp][1]))) #m/min

if select=='Going out for a walk':
	w1 = st.checkbox('Take me to a park!', value=False , key=1)
	w2 = st.checkbox('Keep me away from busy streets', value=False , key=3)
else:
	w1 = False
	w2 = st.checkbox('Keep me away from busy streets', value=False , key=3)

submit = st.button('Calculate route - Go!', key=1)

if not submit:
	latitude = 42.358
	longitude = -71.085
	st.pydeck_chart(pdk.Deck(
		map_style="mapbox://styles/mapbox/light-v9", 
		initial_view_state=pdk.ViewState(latitude = latitude, longitude = longitude, zoom=11)))
else:
	if select=='Going out for a walk':
		with st.spinner('Routing...'):
			source_to_source(G, gdf_nodes, gdf_edges, input1, walking_rate*duration, w1, w2)
	else:
		with st.spinner('Routing...'):
			source_to_dest(G, gdf_nodes, gdf_edges, input1, input2, walking_rate*duration, walking_rate, w2)