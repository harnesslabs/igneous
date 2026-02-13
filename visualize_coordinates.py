import argparse
import numpy as np
import pyvista as pv

def visualize_torus_angles(filename):
    print(f"Loading {filename}...")
    
    header_end = 0
    num_vertices = 0
    
    # 1. Parse header to get vertex count and find data start
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if "element vertex" in line:
                num_vertices = int(line.split()[-1])
            if line.strip() == "end_header":
                header_end = i + 1
                break
    
    if header_end == 0 or num_vertices == 0:
        print("Error: Could not parse PLY header correctly.")
        return

    print(f"  Header parsed. Reading {num_vertices} vertices...")
    
    # 2. Extract Data
    # We only read exactly 'num_vertices' rows to ignore the face data below
    try:
        data = np.loadtxt(lines[header_end : header_end + num_vertices], usecols=(0,1,2,3,4,5))
        points = data[:, 0:3]
        colors = data[:, 3:6].astype(np.uint8)
    except Exception as e:
        print(f"Error parsing vertex data: {e}")
        return

    # 3. Create PyVista Object
    point_cloud = pv.PolyData(points)
    point_cloud.point_data['RGB'] = colors

    # 4. Plotting
    plotter = pv.Plotter(window_size=[1000, 800])
    plotter.set_background('white')
    
    plotter.add_mesh(
        point_cloud, 
        scalars='RGB', 
        rgb=True, 
        point_size=5.0, 
        render_points_as_spheres=True,
        lighting=True
    )

    plotter.add_text(f"Torus Angles: {filename}", position='upper_left', color='black')
    print("Opening interactive viewer...")
    plotter.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to the .ply file")
    args = parser.parse_args()
    visualize_torus_angles(args.file)