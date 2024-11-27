import json
import random

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

def generate_random_obstacles_quadrant(
    screen_width=800,
    screen_height=600,
    num_obstacles=20,
    rectangle_size_range=(30, 100),  # (min_width, max_width) and (min_height, max_height)
    circle_radius_range=(15, 50),
    obstacle_types=['rectangle', 'circle'],
    output_file='random_track.json'
):
    obstacles = []

    for _ in range(num_obstacles):
        # Random position
        x = random.randint(50, screen_width - 50)

        y = 0
        if x < screen_width // 2:
            y = random.randint(50 + (screen_height-50) // 2, screen_height - 50)
        else:
            y = random.randint(50, (screen_height-50))

        ob_type = random.choice(obstacle_types)

        # Random size
        if ob_type == 'rectangle':
            width = random.randint(rectangle_size_range[0], rectangle_size_range[1])
            height = random.randint(rectangle_size_range[0], rectangle_size_range[1])
            obstacles.append({
                "type": "rectangle",
                "pos": [x, y],
                "width": width,
                "height": height,
                "color": [128, 128, 128]
            })
        elif ob_type == 'circle':
            radius = random.randint(circle_radius_range[0], circle_radius_range[1])
            obstacles.append({
                "type": "circle",
                "pos": [x, y],
                "radius": radius,
                "color": [128, 128, 128]
            })

    # Write to JSON file
    with open(output_file, "w") as f:
        json.dump({
            "screen_size": [screen_width, screen_height],
            "obstacles": obstacles
        }, f, indent=4)

    print(f"Generated {output_file} with {len(obstacles)} obstacles.")

def generate_random_obstacles(
    screen_width=800,
    screen_height=600,
    num_obstacles=20,
    rectangle_size_range=(30, 100),  # (min_width, max_width) and (min_height, max_height)
    circle_radius_range=(15, 50),
    obstacle_types=['rectangle', 'circle'],
    output_file='random_track.json'
):
    obstacles = []

    for _ in range(num_obstacles):
        # Random position
        x = random.randint(60, screen_width - 200)

        y = random.randint(60, screen_height- 100)

        ob_type = random.choice(obstacle_types)

        # Random size
        if ob_type == 'rectangle':
            width = random.randint(rectangle_size_range[0], rectangle_size_range[1])
            height = random.randint(rectangle_size_range[0], rectangle_size_range[1])
            obstacles.append({
                "type": "rectangle",
                "pos": [x, y],
                "width": width,
                "height": height,
                "color": [128, 128, 128]
            })
        elif ob_type == 'circle':
            radius = random.randint(circle_radius_range[0], circle_radius_range[1])
            obstacles.append({
                "type": "circle",
                "pos": [x, y],
                "radius": radius,
                "color": [128, 128, 128]
            })

    # Write to JSON file
    with open(output_file, "w") as f:
        json.dump({
            "screen_size": [screen_width, screen_height],
            "obstacles": obstacles
        }, f, indent=4)

    print(f"Generated {output_file} with {len(obstacles)} obstacles.")



# Example usage
if __name__ == "__main__":
    generate_random_obstacles_quadrant(
        screen_width=800,
        screen_height=600,
        num_obstacles=30,
        rectangle_size_range=(30, 100),
        circle_radius_range=(15, 50),
        obstacle_types=['rectangle', 'circle'],
        output_file='random_track_q.json'
    )

    generate_random_obstacles(
        screen_width=800,
        screen_height=600,
        num_obstacles=20,
        rectangle_size_range=(30, 100),
        circle_radius_range=(15, 50),
        obstacle_types=['rectangle', 'circle'],
        output_file='random_track.json'
    )

    # Example usage
    tight_track_generator(path_margin=2.5, rectangle_size=60)