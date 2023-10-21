%time
import open3d as o3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.optimize import minimize
import cupy as cp



class logger():
    def __init__(self,):
      
        self.candidate1 = []
        self.candidate2 = []
        self.candidate3 = []
        self.err = []
        self.handle_force = []
        self.sec_force = []
        self.ter_force = []
        self.temp = 10
        self.min = []
    
    

    def log(self,sol,id,err,pcd):

        ns = np.asarray([pcd.normals[id[0]],pcd.normals[id[1]],pcd.normals[id[2]]])
        rank = np.linalg.matrix_rank(ns) 

        if err < self.temp:
            self.temp = err
            self.min = id
        self.candidate1.append(id[0])
        self.candidate2.append(id[1])
        self.candidate3.append(id[2])
        self.handle_force.append([sol[2]])
        self.sec_force.append([sol[5]]) 
        self.ter_force.append([sol[8]])
        self.err.append(err)
          

    def save_file(self):
       df = pd.DataFrame({"Pt1" : self.candidate1, "Pt2" : self.candidate2,"Pt3" : self.candidate3, "F1" : self.handle_force, "F2": self.sec_force, "F3": self.ter_force, "Error" :self.err})
       df.to_csv("realtime_diagnostics.csv", index = False)
       return(self.min)
       

    def cost_visualizer(self):
        k = []
        for i in range(len(self.err)):
            k.append(i)
        plt.plot( k ,self.err,label=' Cost comparison')
        plt.xlabel('Candidate Number')
        plt.ylabel('Final cost')
        plt.title('cost analysis')
        plt.show()
class KdTree:
    def __init__(self, pcd):
        self.pcd = pcd
        self.kd_tree = o3d.geometry.KDTreeFlann(self.pcd)
        self.radius_sweep = 80
        self.selected_points = []
        self.checked_points = []   

    def get_points(self, center_point):

        [k, idx, _] = self.kd_tree.search_radius_vector_3d(
            center_point, self.radius_sweep)
        return idx

    def search(self):
     for k in range(len(self.pcd.points)):
        min = 12
        handle_id = k
        query_point = self.pcd.points[handle_id]
        idx = self.get_points(query_point)
        idx = list(idx)
        idx.remove(handle_id)
        force_optimization = Optimization(handle_id,idx,self.pcd)
        force_optimization.transformation()
        # So reqd_combination will have the required points and normals


class Optimization:

    def __init__(self, handle_id, idx, pcd):
        self.handle_id = handle_id
        self.idx = idx
        self.pcd = pcd
        self.max_force = 10
        self.f_ext = cp.array([0, 0, 10, 0, 0, 0])
        self.Bci = cp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.G = None
        self.mew = 0.7
        self.fc = cp.array([0, 0, 2, 0, 0, 2, 0, 0, 2])
        self.min = 15
        self.solved_combination = None
        self.solution = None

    def choose(self):
        unique_combinations = list(combinations(self.idx, 2))
        return unique_combinations

    def transformation(self):
        unique_combinations = list(self.choose())
        new_combinations = [[item[0], item[1], self.handle_id] for item in unique_combinations]

        for i in range(len(new_combinations)):
            self.G = None
            self.idt = []
            for j in range(3):
                id = new_combinations[i][j]
                self.idt.append(id)
                normal = self.pcd.normals[id]
                point = self.pcd.points[id]
                normal = -normal
                x_axis_angle = cp.arctan2(cp.linalg.norm(cp.cross(normal, cp.array([1, 0, 0])), cp.dot(normal, cp.array([1, 0, 0]))))
                y_axis_angle = cp.arctan2(cp.linalg.norm(cp.cross(normal, cp.array([0, 1, 0])), cp.dot(normal, cp.array([0, 1, 0]))))
                z_axis_angle = cp.arctan2(cp.linalg.norm(cp.cross(normal, cp.array([0, 0, 1])), cp.dot(normal, cp.array([0, 0, 1]))))

                R_alpha = cp.array([[cp.cos(z_axis_angle), -cp.sin(z_axis_angle), 0],
                                   [cp.sin(z_axis_angle), cp.cos(z_axis_angle), 0],
                                   [0, 0, 1]])

                R_beta = cp.array([[cp.cos(y_axis_angle), 0, cp.sin(y_axis_angle)],
                                  [0, 1, 0],
                                  [-cp.sin(y_axis_angle), 0, cp.cos(y_axis_angle)]])

                R_gamma = cp.array([[1, 0, 0],
                                   [0, cp.cos(x_axis_angle), -cp.sin(x_axis_angle)],
                                   [0, cp.sin(x_axis_angle), cp.cos(x_axis_angle)]])
                R = cp.dot(R_alpha, cp.dot(R_beta, R_gamma))

                H = cp.array([[0, -point[2], point[1]], [point[2], 0, -point[0]], [-point[1], point[0], 0]])

                cross_G = cp.dot(H, R)
                zeros_matrix = cp.zeros((3, 3))
                G_i = cp.vstack((cp.hstack((R, zeros_matrix)), cp.hstack((cp.cross(cross_G, R), R))))
                F_oi = cp.dot(G_i, self.Bci)
                if self.G is None:
                    self.G = F_oi
                else:
                    self.G = cp.hstack((self.G, F_oi))
            self.solve()

    def objective_function(self, fc):
        return cp.linalg.norm(cp.dot(self.G, fc) + self.f_ext)

    def constraint_4(self, fc):
        return self.mew * fc[2] - cp.sqrt(fc[0]**2 + fc[1]**2)

    def constraint_5(self, fc):
        return self.mew * fc[5] - cp.sqrt(fc[3]**2 + fc[4]**2)

    def constraint_6(self, fc):
        return self.mew * fc[8] - cp.sqrt(fc[6]**2 + fc[7]**2)

    def solve(self):
        con4 = {'type': 'ineq', 'fun': self.constraint_4}
        con5 = {'type': 'ineq', 'fun': self.constraint_5}
        con6 = {'type': 'ineq', 'fun': self.constraint_6}
        b = (0, 10)
        bnds = [b, b, b, b, b, b, b, b, b]
        cons = [con4, con5, con6]
        sol = minimize(self.objective_function, self.fc, method='SLSQP', bounds=bnds, constraints=cons)
        err = self.objective_function(sol.x)
        solution = list(sol.x)
        if self.objective_function(sol.x) < 10:
            log1.log(solution, self.idt, err, self.pcd)

       
def visualize(mesh):

    points = np.asarray(mesh.points)
    x_centroid = np.mean(points[:, 0])
    y_centroid = np.mean(points[:, 1])
    z_centroid = np.mean(points[:, 2])
    mesh_coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=5, origin=[x_centroid, y_centroid, z_centroid])
    o3d.visualization.draw_geometries(
        [mesh_coord_frame, mesh], point_show_normal=True)


def mesh2PointCloud(mesh):
    n_pts = 50
    pcd = mesh.sample_points_uniformly(n_pts)
    return pcd


def force_visualizer(mesh, points, normals,center_point):
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    visualizer.add_geometry(mesh)

    points1 = np.array([[center_point[0],center_point[1],-75], [center_point[0],center_point[1],75]])
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points1)
 
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    visualizer.add_geometry(pcd)

    scene = o3d.geometry.PointCloud()
    points2 = np.array([[100, 200, 300], [400, 500, 600], [700, 800, 900]])
    scene.points = o3d.utility.Vector3dVector(points2)
    axes_line_set = o3d.geometry.LineSet()
    axes_line_set.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    axes_line_set.lines = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])
    colors1 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    axes_line_set.colors = o3d.utility.Vector3dVector(colors1)
    visualizer.add_geometry(axes_line_set)

    lines = [[0, 1]]
    colors = [[1, 0, 0]]  # Red color
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points1)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    visualizer.add_geometry(line_set)
    visualizer.run()
    visualizer.destroy_window()

def visualizer(mesh):
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    visualizer.add_geometry(mesh)
    visualizer.run()
    visualizer.destroy_window()


def main():

    mesh_path = "/content/mesh.stl"
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    pcd = mesh2PointCloud(mesh)
    obj = KdTree(pcd)
    center_point = np.mean(np.asarray(pcd.points), axis=0)
    pcd_df = pd.DataFrame(np.concatenate((np.asarray(pcd.points), np.asarray(pcd.normals)), axis=1),
                          columns=["x", "y", "z", "norm-x", "norm-y", "norm-z"]
                          )

    
    reqd_combination = obj.search()
    min = log1.save_file()
    log1.cost_visualizer()
    print( "Optimum force at fingers is ",pcd.normals[min[0]],pcd.normals[min[1]],pcd.normals[min[2]])
    ns = np.asarray([pcd.normals[min[0]],pcd.normals[min[1]],pcd.normals[min[2]]])
    
    pts = np.asarray([pcd.points[min[0]],pcd.points[min[1]],pcd.points[min[2]]])
    force_visualizer(mesh,pts,ns,center_point)
    visualizer(mesh)

log1 = logger()

if __name__ == "__main__":
    main()
