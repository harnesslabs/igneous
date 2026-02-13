import argparse
import numpy as np
import pyvista as pv

def visualize_manual(filename, arrow_scale=0.1):
    print(f"Manual load of {filename}...")
    
    # 1. Parse ASCII PLY manually to bypass VTK quirks
    header_end = 0
    with open(filename, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.strip() == "end_header":
                header_end = i + 1
                break
    
    if header_end == 0:
        print("Error: Could not find 'end_header' in PLY file.")
        return

    print("  Header parsed. Reading data...")
    
    # Load all data into a numpy array (skipping header)
    # This assumes the columns are: x y z vx vy vz
    try:
        data = np.loadtxt(filename, skiprows=header_end)
    except Exception as e:
        print(f"Error reading data: {e}")
        return

    # Split into Points and Vectors
    points = data[:, 0:3]
    vectors = data[:, 3:6]

    print(f"  Loaded {len(points)} points.")

    # 2. Build PyVista Mesh
    mesh = pv.PolyData(points)
    mesh.point_data['Flow'] = vectors
    mesh.point_data['Magnitude'] = np.linalg.norm(vectors, axis=1)

    # 3. Filter zero-magnitude vectors (noise reduction)
    # For harmonic forms, some areas might have very low flow
    mask = mesh.point_data['Magnitude'] > 1e-6
    arrows_mesh = mesh.extract_points(mask)

    # 4. Generate Arrows
    arrows = arrows_mesh.glyph(
        orient='Flow', 
        scale='Magnitude', 
        factor=arrow_scale,
        geom=pv.Arrow()
    )

    # 5. Visualize
    plotter = pv.Plotter()
    plotter.set_background('white')
    
    # Draw faint points for context
    plotter.add_mesh(mesh, color='black', point_size=2, opacity=0.1, render_points_as_spheres=True)
    
    # Draw arrows
    plotter.add_mesh(arrows, scalars='Magnitude', cmap='coolwarm', lighting=True)
    
    plotter.add_text(f"Harmonic Form (Manual Load)", position='upper_left', color='black')
    print("Displaying...")
    plotter.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to .ply file")
    parser.add_argument("--scale", type=float, default=0.15)
    args = parser.parse_args()
    
    visualize_manual(args.file, args.scale)