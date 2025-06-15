import open3d as o3d
import numpy as np

pointcloud_path = "/Users/yazansharkia/Desktop/Topology3D/mesh_generation/input/fused_clean_up.ply"
pcd = o3d.io.read_point_cloud(pointcloud_path)
 
# === Remove statistical outliers ===
cl, ind = pcd.remove_statistical_outlier(nb_neighbors=15, std_ratio=1)
cleaned_pcd = pcd.select_by_index(ind)
# cleaned_pcd = cleaned_pcd.voxel_down_sample(voxel_size=0.01)
o3d.visualization.draw_geometries([cleaned_pcd], window_name="Outliers Removed")

# === Estimate normals (needed for Poisson) ===
cleaned_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
    radius=0.3, max_nn=17))

# # Orient normals consistently
# cleaned_pcd.orient_normals_consistent_tangent_plane(k=30)

# === Run Poisson surface reconstruction ===
print("Running Poisson reconstruction...")
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    cleaned_pcd, depth=9, linear_fit=False)
o3d.visualization.draw_geometries([mesh], window_name="Raw Poisson Mesh")

# === Remove low-density vertices to clean mesh ===
densities_np = np.asarray(densities)
threshold = np.quantile(densities_np, 0.05)
vertices_to_keep = densities_np > threshold
mesh = mesh.select_by_index(np.where(vertices_to_keep)[0])

# === (Optional) Apply Laplacian smoothing ===
# mesh = mesh.filter_smooth_laplacian(number_of_iterations=2)
o3d.visualization.draw_geometries([mesh], window_name="Smoothed Mesh", mesh_show_back_face=True) # Enable backface rendering when visualizing

# === Save final output ===
success = o3d.io.write_triangle_mesh("/Users/yazansharkia/Desktop/Topology3D/mesh_generation/output/foot_mesh_smoothed.ply", mesh)
if not success:
    print("❌ Failed to save the mesh! Check the output directory and permissions.")
else:
    print("✅ Mesh saved as 'foot_mesh_smoothed.ply'")
