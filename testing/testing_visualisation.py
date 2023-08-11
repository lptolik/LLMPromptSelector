import folium

# Define the coordinates for each location in your route
coordinates = [
    (55.95065, -3.19101),  # The Milkman
    (55.9486, -3.1999),  # Edinburgh Castle
    (55.949, -3.19564),  # Camera Obscura and World of Illusions
    (55.9520, -3.1769),  # Oink
    (55.9533, -3.1883),  # Royal Mile
    (55.9507, -3.1746)   # Dynamic Earth
]

# Create a map object centered around Edinburgh
map = folium.Map(location=(55.9531, -3.1889), zoom_start=13)

# Add markers for each location in your route with pop-up labels showing their names
for i in range(len(coordinates)):
    marker = folium.Marker(coordinates[i], popup=f'Location {{i+1}}')
    marker.add_to(map)

# Add a line connecting all the locations in your route
folium.PolyLine(coordinates).add_to(map)

# Save the map as an HTML file named "route_map.html"
map.save("route_map.html")