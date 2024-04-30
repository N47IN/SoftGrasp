import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import math
import subprocess


def mesh2PointCloud(mesh):
    n_pts = 10000 
    pcd = mesh.sample_points_uniformly(n_pts)
    return pcd

class Affordance():
    def __init__(self,input):
        self.mesh = o3d.io.read_triangle_mesh(input)
        self.pcd = mesh2PointCloud(self.mesh)
        o3d.io.write_point_cloud("output.ply", self.pcd)
        o3d.io.write_triangle_mesh("output.off", self.mesh)
        
    def get_segments(self):
        segments = []
        cpp_object_file = "/home/navin/SoftGrasp/Affordance-Modelling-CGAL/Surface_mesh_segmentation/new1"
        subprocess.run(cpp_object_file)
        for i in range(10):
            filename = "Segment_" + str(i) + ".off"
            segment_mesh = o3d.io.read_triangle_mesh(filename)
            if segment_mesh.is_empty():
                return segments
            pcd = mesh2PointCloud(segment_mesh)
            o3d.visualization.draw_geometries([pcd])
            segments.append(pcd)
        return segments
    
    def get_planes(self):
        planes = []
        cpp_object_file = "/home/navin/SoftGrasp/Affordance-Modelling-CGAL/planar_seg/new1"
        subprocess.run(cpp_object_file)
        point_cloud = o3d.io.read_point_cloud("/home/navin/SoftGrasp/Affordance-Modelling-CGAL/planar_seg/planes_point_set_3.ply")
        o3d.visualization.draw_geometries([point_cloud])
        colors = np.asarray(point_cloud.colors)
        unique_colors = np.unique(colors, axis=0)
        color_point_map = {}
        for color in unique_colors:
        
            color_indices = np.where(np.all(colors == color, axis=1))[0]
            color_points = point_cloud.select_by_index(color_indices)
            color_points_np = color_points
            color_point_map[tuple(color)] = color_points_np
            planes.append(color_points_np)
        return planes

mesh = "/home/navin/SoftGrasp/Affordance-Modelling-CGAL/Surface_mesh_segmentation/Mug.STL"
affordancey = Affordance(mesh)
segments = affordancey.get_segments()
planes = affordancey.get_planes()




        
        



