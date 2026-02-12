import argparse
import glob
import numpy as np
import pyvista as pv

parser = argparse.ArgumentParser(description="Visualize curvature flow simulation.")
parser.add_argument("-i", "--input-dir", default="output", help="directory containing frame_*.obj files (default: output)")
parser.add_argument("-o", "--output", default="curvature_flow.gif", help="output gif filename (default: curvature_flow.gif)")
parser.add_argument("-n", type=int, default=None, help="max number of frames to include (default: all)")

def read_obj_with_colors(filename):
    """Read OBJ with 'v x y z r g b' vertex color extension."""
    verts = []
    colors = []
    faces = []
    with open(filename) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == 'v' and len(parts) >= 7:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                colors.append([float(parts[4]), float(parts[5]), float(parts[6])])
            elif parts[0] == 'f':
                idx = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                faces.append(idx)
    verts = np.array(verts, dtype=np.float32)
    colors = (np.array(colors, dtype=np.float32) * 255).astype(np.uint8)
    face_arr = np.hstack([[len(f)] + f for f in faces])
    mesh = pv.PolyData(verts, face_arr)
    mesh.point_data['RGB'] = colors
    return mesh

def main(args):
    files = sorted(glob.glob(f"{args.input_dir}/frame_*.obj"))[:args.n]

    plotter = pv.Plotter()
    plotter.open_gif(args.output)

    for filename in files:
        mesh = read_obj_with_colors(filename)
        plotter.clear()
        plotter.add_mesh(mesh, scalars='RGB', rgb=True, show_scalar_bar=False)
        plotter.add_text(f"Simulation: {filename}", position='upper_left')

        if filename == files[0]:
            plotter.camera_position = 'xy'
            plotter.reset_camera()

        plotter.write_frame()

    plotter.close()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
