import json

def tight_track_generator(
    screen_width=600,
    screen_height=600,
    offset=50,
    num_obstacles=50,
    rectangle_size=30,
    path_margin=1.5  # Multiplier for spacing around the diagonal path
):
    obstacles = []
    
    # Calculate the drawable area
    drawable_width = screen_width - 2 * offset
    drawable_height = screen_height - 2 * offset
    
    # Starting point offset from the top-left corner
    start_x = offset
    start_y = offset
    
    # Step sizes based on the number of obstacles
    step_x = drawable_width / num_obstacles
    step_y = drawable_height / num_obstacles
    
    for i in range(0, num_obstacles, 2):
        x = start_x + i * step_x
        y_center = start_y + i * step_y
        
        # Upper rectangle (above the diagonal path)
        y_upper = y_center - (path_margin * rectangle_size) / 2
        obstacles.append({
            "type": "rectangle",
            "pos": [int(x), int(y_upper)],
            "width": rectangle_size,
            "height": rectangle_size,
            "color": [128, 128, 128]
        })
        
        # Lower rectangle (below the diagonal path)
        y_lower = y_center + (path_margin * rectangle_size) / 2
        obstacles.append({
            "type": "rectangle",
            "pos": [int(x), int(y_lower)],
            "width": rectangle_size,
            "height": rectangle_size,
            "color": [128, 128, 128]
        })
    
    # Output to JSON
    track_data = {
        "screen_size": [screen_width, screen_height],
        "offset": offset,
        "obstacles": obstacles
    }
    
    with open("tight_track.json", "w") as f:
        json.dump(track_data, f, indent=4)
    
    print(f"Generated tight_track.json with {len(obstacles)} obstacles.")

# Example usage
tight_track_generator(path_margin=2.5, rectangle_size=60)