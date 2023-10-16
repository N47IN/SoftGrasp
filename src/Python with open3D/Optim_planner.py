
import open3d as o3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.optimize import minimize



class logger():
    def __init__(self):
      
        self.candidates = []
        self.err = []
        self.handle_force = []
        self.sec_force = []
        self.ter_force = []
    
    def log(self,sol,id,err):
        self.candidates.append([id[0],id[1],id[2]])
        self.handle_force.append[sol.x[0]]
        self.sec_force.append[sol.x[1]] 
        self.ter_force.append[sol.x[2]]
        self.err.append(err)  

    def save_file(self):
       df = pd.DataFrame({"Candidates" : np.array(self.candidates), "F1" : np.array(self.handle_force), "F2": np.array(self.sec_force), "f3": np.array(self.ter_force), "Error" :np.array(self.err)})
       df.to_csv("diagnostics.csv", index = False)

    def cost_visualizer(self):
        k = []
        for i in range(len(self.err)):
            k.append(i)
        plt.plot( k ,self.err,label=' Cost comparison')
        plt.xlabel('Candidate Number')
        plt.ylabel('Final cost')
        plt.title('cost analysis')


log1 = logger()
class KdTree:
    def __init__(self, pcd):
        self.pcd = pcd
        self.kd_tree = o3d.geometry.KDTreeFlann(self.pcd)
        self.radius_sweep = 20
        self.selected_points = []
        self.checked_points = []   

    def get_points(self, center_point):

        [k, idx, _] = self.kd_tree.search_radius_vector_3d(
            center_point, self.radius_sweep)
        return idx

    def search(self):
        handle_id = 5
        query_point = self.pcd.points[handle_id]
        idx = self.get_points(query_point)
        idx = list(idx)
        idx.remove(handle_id)
        force_optimization = Optimization(handle_id,idx,self.pcd)
        force_optimization.transformation()
        min_error_combination = force_optimization.solved_combination
        min_error = force_optimization.min
        
        if min > min_error:
                reqd_combination = min_error_combination
                min = min_error
                solution = force_optimization.solution
        # So reqd_combination will have the required points and normals

        print(f"Error={min}")
        print(
            f"Normal forces to be applied at the contacts {solution[2]} {solution[5]} {solution[8]}")
        print(
            f"Friction forces in these points are {solution[0]} {solution[1]} {solution[3]} {solution[4]} {solution[6]} {solution[7]}")
        return reqd_combination


class Optimization:

    def __init__(self,handle_id, idx, pcd):
        self.handle_id = handle_id
        self.idx = idx
        self.pcd = pcd
        self.max_force = 10
        self.f_ext = np.asarray([0, 0, 10, 0, 0, 0])
        # For point contact with friction
        self.Bci = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1], [
            0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.G = None
        self.mew = 1  # assuming it is high friction surface for testing
        self.fc = [0, 0, 2, 0, 0, 2, 0, 0, 2]
        self.min = 15  # Taking 15 since max error will be 10
        self.solved_combination = None
        self.solution = None

    def choose(self):

        unique_combinations = np.asarray(
            list(combinations(self.idx), 2))
        return unique_combinations

    def transformation(self):
        self.id = []
        unique_combinations = self.choose()
        new_combinations = [(unique_combinations[0], unique_combinations[1], self.handle_id) for item in unique_combinations]
        
        for i in len(new_combinations):
            for j in range(3):
                self.id.append(new_combinations[i][j]) 
                normal = self.pcd.normals[new_combinations[i][j]]
                point = self.pcd.points[new_combinations[i][j]]
                # This gives us orientation of normal vector with x,y and z axis
                normal = -normal
                x_axis_angle = np.arctan2(np.linalg.norm(np.cross(
                    normal, np.asarray([1, 0, 0]))), np.dot(normal, np.asarray([1, 0, 0])))
                y_axis_angle = np.arctan2(np.linalg.norm(np.cross(
                    normal, np.asarray([0, 1, 0]))), np.dot(normal, np.asarray([0, 1, 0])))
                z_axis_angle = np.arctan2(np.linalg.norm(np.cross(
                    normal, np.asarray([0, 0, 1]))), np.dot(normal, np.asarray([0, 0, 1])))

                R_alpha = np.array([[np.cos(z_axis_angle), -np.sin(z_axis_angle), 0],
                                    [np.sin(z_axis_angle), np.cos(
                                        z_axis_angle), 0],
                                    [0, 0, 1]])

                R_beta = np.array([[np.cos(y_axis_angle), 0, np.sin(y_axis_angle)],
                                   [0, 1, 0],
                                   [-np.sin(y_axis_angle), 0, np.cos(y_axis_angle)]])

                R_gamma = np.array([[1, 0, 0],
                                    [0, np.cos(x_axis_angle), -
                                     np.sin(x_axis_angle)],
                                    [0, np.sin(x_axis_angle), np.cos(x_axis_angle)]])
                R = np.dot(R_alpha, np.dot(R_beta, R_gamma))

                H = np.matrix(
                    [[0, -point[2], point[1]], [point[2], 0, -point[0]], [-point[1], point[0], 0]])

                cross_G = np.dot(H, R)
                zeros_matrix = np.zeros((3, 3))
                G_i = np.vstack((np.hstack((R, zeros_matrix)),
                                np.hstack((np.cross(cross_G, R), R))))
                F_oi = np.dot(G_i, self.Bci)
                if self.G is None:
                    self.G = F_oi
                else:
                    self.G = np.hstack((self.G, F_oi))
            # I have to optimize the points from here
            self.solve()

    def objective_function(self, fc):
        return np.linalg.norm(np.dot(self.G, fc)+self.f_ext)

    def constraint_4(self, fc):
        return self.mew*fc[2]-np.sqrt(fc[0]**2+fc[1]**2)

    def constraint_5(self, fc):
        return self.mew*fc[5]-np.sqrt(fc[3]**2+fc[4]**2)

    def constraint_6(self, fc):
        return self.mew*fc[8]-np.sqrt(fc[6]**2+fc[7]**2)

    def solve(self):
        con4 = {'type': 'ineq', 'fun': self.constraint_4}
        con5 = {'type': 'ineq', 'fun': self.constraint_5}
        con6 = {'type': 'ineq', 'fun': self.constraint_6}
        b = (0, 10)
        bnds = [b, b, b, b, b, b, b, b, b]
        cons = [con4, con5, con6]
        sol = minimize(self.objective_function, self.fc,
                       method='SLSQP', bounds=bnds, constraints=cons)
        err = self.objective_function(sol.x)
        
        log1.log(sol,self.id,err)

        if self.objective_function(sol.x) < 8:
            print(
                f"Normal forces to be applied at the contacts {sol.x[2]} {sol.x[5]} {sol.x[8]} and corresponding error = {self.objective_function(sol.x)}")
            print(
                f"Friction forces in these points are {sol.x[0]} {sol.x[1]} {sol.x[3]} {sol.x[4]} {sol.x[6]} {sol.x[7]}")
        if self.objective_function(sol.x) < self.min:
            self.min = self.objective_function(sol.x)
            
            self.solution = sol.x


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
    n_pts = 100
    pcd = mesh.sample_points_uniformly(n_pts)
    return pcd


def force_visualizer(mesh, points, normals):
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()
    visualizer.add_geometry(mesh)  # Display the STL mesh

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    visualizer.add_geometry(pcd)
    visualizer.run()
    visualizer.destroy_window()


def main():
    log = logger()
    mesh_path = "cuboid.stl"
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    pcd = mesh2PointCloud(mesh)
    pcd_df = pd.DataFrame(np.concatenate((np.asarray(pcd.points), np.asarray(pcd.normals)), axis=1),
                          columns=["x", "y", "z", "norm-x", "norm-y", "norm-z"]
                          )

    obj = KdTree(pcd)
    reqd_combination = obj.search()
  
    points = []
    normals = []
    for point, normal in reqd_combination:
        points.append(point)
        normals.append(normal)

    force_visualizer(mesh, np.asarray(points), np.asarray(normals))
    log1.save_file()


if __name__ == "__main__":
    main()
