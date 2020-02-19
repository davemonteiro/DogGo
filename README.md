# DogGo
DogGo - finding the best walk for your best friend!

Online at https://dm-doggo.herokuapp.com

DogGo is an application that will route a Boston dog walk for you and display the route over a map of the city. DogGo uses an OSMnx graph of streets in Boston, and finds an optimum path for your dog through the city depending on your schedule and preferences.

DogGo has a few distinct use cases that each implement different variations of Dijkstra's shortest path algorithm. Edges in the graph - segments of walkable sidewalks in Boston - have mutliple attributes that contribute to the optimum path. In addition to length, greenery and road safety are considered to ensure that the user's dog spends most of his/her time near parks and trees and away from cars on busier roads.

Data from Openstreetmaps were used to generate the graph of walkable paths via the OSMnx package in Python and provide information on busier/more residential streets.

Data from Boston and Brookline Parks and Recreation databases were used to identify locations of 200+ city parks and coordinates of 200k+ public trees.

Data from Dogtime.com were used to suggest a minimum walk duration and adjust the pace of the walk, depending on the user's dog breed.