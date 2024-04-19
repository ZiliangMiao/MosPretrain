import torch
import numpy as np

mos_colormap = {
        0: (255/255, 255/255, 255/255),  # unknown: white
        1: (25/255, 80/255, 25/255),    # static: green
        2: (255/255, 20/255, 20/255)     # moving: red
}

check_pcd_file = "/home/user/Projects/vis_tmp_dir/sekitti_ego_mask_check.pcd"

def save_mos_sample(points, labels):
    # check input data type, convert to numpy
    if torch.is_tensor(points):
        points = points.cpu().detach().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().detach().numpy()
    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    vfunc = np.vectorize(mos_colormap.get)
    colors = np.array(vfunc(labels)).T
    pts.colors = open3d.utility.Vector3dVector(colors)
    open3d.io.write_point_cloud(check_pcd_file, pts)

def render_mos_pcd():
    import open3d
    vis = open3d.visualization.Visualizer()

    # draw origin
    axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(axis_pcd)

    # draw points
    pcd = open3d.io.read_point_cloud(check_pcd_file)
    vis.add_geometry(pcd)

    # view settings
    # vis.get_render_option().point_size = 3.0
    # vis.get_render_option().background_color = np.zeros(3)
    #
    # view_ctrl = vis.get_view_control()
    # view_ctrl.set_front((0.75263429526187886, -0.13358133681379755, 0.64474618575893383))
    # view_ctrl.set_lookat((16.206845402638745, -3.8676194858766819, 15.365323753623207))
    # view_ctrl.set_up((-0.64932205862151104, 0.011806106960120792, 0.76042190922274799))
    # view_ctrl.set_zoom((0.059999999999999998))

    # run vis
    vis.run()

if __name__ == '__main__':
    pcd = open3d.io.read_point_cloud(check_pcd_file)
    open3d.visualization.draw_geometries([pcd])

    # render_mos_pcd()
