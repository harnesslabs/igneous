import glob
import pyvista as pv
import time

# Get all frame files sorted
files = sorted(glob.glob("frame_*.obj"))

plotter = pv.Plotter()
plotter.open_gif("curvature_flow.gif")

for filename in files:
    mesh = pv.read(filename)
    plotter.clear()
    # Color by the vertex colors baked into the OBJ
    plotter.add_mesh(mesh, rgb=True, show_scalar_bar=False)
    plotter.add_text(f"Simulation: {filename}", position='upper_left')
    
    # Keep camera fixed?
    if filename == files[0]:
        plotter.camera_position = 'xy'
        plotter.reset_camera()
        
    plotter.write_frame()

plotter.close()