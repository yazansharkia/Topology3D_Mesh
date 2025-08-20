# # #!/usr/bin/env python3
# # import argparse, os, sys, json
# # import numpy as np
# # import open3d as o3d
# # import matplotlib.pyplot as plt
# # import matplotlib.cm as cm

# # try:
# #     import alphashape
# #     from shapely.geometry import Polygon
# #     HAVE_ALPHA = True
# # except Exception:
# #     from scipy.spatial import ConvexHull
# #     HAVE_ALPHA = False

# # def ensure_dir(d): os.makedirs(d, exist_ok=True)

# # def load_calibration(path):
# #     with open(path, "r") as f:
# #         data = json.load(f)
# #     R = np.array(data["R"], dtype=float)
# #     centroid = np.array(data["centroid"], dtype=float)
# #     scale = float(data["scale_mm_per_unit"])
# #     return centroid, R, scale

# # def clean_point_cloud(pcd, voxel_mm=1.5, nb_neighbors=40, std_ratio=1.8):
# #     out = pcd
# #     if voxel_mm and voxel_mm > 0:
# #         out = out.voxel_down_sample(voxel_size=float(voxel_mm))
# #     out, _ = out.remove_statistical_outlier(nb_neighbors=int(nb_neighbors),
# #                                             std_ratio=float(std_ratio))
# #     return out

# # def transform_and_scale(points, centroid, R, scale_mm):
# #     # into paper PCA frame, then mm
# #     return ((points - centroid) @ R) * scale_mm

# # def slice_cloud(points_xyz_mm, step_mm=0.5, thickness_mm=1.5, axis=0):
# #     vals = points_xyz_mm[:, axis]
# #     vmin, vmax = vals.min(), vals.max()
# #     print (vmin, vmax)

# #     centers = np.arange(vmin, vmax + step_mm, step_mm)
# #     half = thickness_mm / 2.0
# #     out = []
# #     print (len(centers))
# #     for c in centers:
# #         m = (vals >= c - half) & (vals <= c + half)
# #         pts = points_xyz_mm[m]
# #         if pts.shape[0] >= 100:
# #             out.append((pts, c))
# #     return out

# # def boundary_metrics(slice_pts_mm, axis=0, alpha_param=None):
# #     # Project to plane ⟂ axis
# #     if axis == 0:      # slicing along X -> project to (Y,Z); width across Y
# #         P = slice_pts_mm[:, [1,2]]
# #         width = slice_pts_mm[:,1].max() - slice_pts_mm[:,1].min()
# #     elif axis == 1:
# #         P = slice_pts_mm[:, [0,2]]; width = slice_pts_mm[:,0].max() - slice_pts_mm[:,0].min()
# #     else:
# #         P = slice_pts_mm[:, [0,1]]; width = slice_pts_mm[:,0].max() - slice_pts_mm[:,0].min()

# #     if HAVE_ALPHA and len(P) >= 4:
# #         if alpha_param is None:
# #             # quick heuristic: median nn distance
# #             try:
# #                 from sklearn.neighbors import NearestNeighbors
# #                 k = min(8, len(P)-1)
# #                 nbrs = NearestNeighbors(n_neighbors=k).fit(P)
# #                 dists, _ = nbrs.kneighbors(P)
# #                 med = float(np.median(dists[:,1:]))
# #                 alpha_param = max(2.0*med, 1e-6)
# #             except Exception:
# #                 alpha_param = 1.0
# #         ashape = alphashape.alphashape(P, alpha_param)
# #         if ashape.is_empty:
# #             perimeter, area, boundary_xy = _convex_metrics(P)
# #         else:
# #             geom = max(list(ashape.geoms), key=lambda g: g.area) if ashape.geom_type=="MultiPolygon" else ashape
# #             perimeter = float(geom.length)
# #             area = float(geom.area)
# #             boundary_xy = np.array(geom.exterior.coords)
# #     else:
# #         perimeter, area, boundary_xy = _convex_metrics(P)

# #     return perimeter, width, area, boundary_xy

# # def _convex_metrics(P):
# #     if len(P) < 3:
# #         return 0.0, 0.0, None
# #     hull = ConvexHull(P)
# #     poly = Polygon(P[hull.vertices])
# #     return float(poly.length), float(poly.area), np.array(list(poly.exterior.coords))

# # def detect_ball_index(xs, perims, widths, smooth_win=7):
# #     def smooth(y, k):
# #         if len(y) < k: return y
# #         pad = k//2
# #         ypad = np.pad(y, (pad,pad), mode='edge')
# #         return np.convolve(ypad, np.ones(k)/k, mode='valid')
# #     ps = smooth(perims, smooth_win)
# #     ws = smooth(widths, smooth_win)
# #     n = len(xs)
# #     end = max(int(0.4*n), 3)
# #     idx, best = 0, -1.0
# #     for i in range(1, end-1):
# #         if ps[i] > ps[i-1] and ps[i] > ps[i+1]:
# #             score = ps[i] + 0.5*ws[i]
# #             if score > best:
# #                 best, idx = score, i
# #     return idx

# # def plot_curves(xs_mm, perim_mm, width_mm, out_png, ball_i):
# #     plt.figure(figsize=(8,5))
# #     plt.plot(xs_mm, perim_mm, label="Perimeter (mm)")
# #     plt.plot(xs_mm, width_mm, label="Width (mm)")
# #     if 0 <= ball_i < len(xs_mm):
# #         plt.axvline(xs_mm[ball_i], ls="--", alpha=0.7, label="Ball slice")
# #     plt.xlabel("Along length (mm)"); plt.ylabel("Value (mm)")
# #     plt.legend(); plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

# # def plot_slice_outline(boundary_xy, out_png, title):
# #     if boundary_xy is None or len(boundary_xy) < 3: return
# #     plt.figure(figsize=(5,5))
# #     plt.plot(boundary_xy[:,0], boundary_xy[:,1], '-')
# #     plt.gca().set_aspect('equal', 'box')
# #     plt.title(title); plt.xlabel("Axis-1 in slice plane"); plt.ylabel("Axis-2 in slice plane")
# #     plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()




# # def make_axes_from_R(origin, R, scale=50.0):
# #     # Columns of R are PC1, PC2, PC3 in world frame
# #     o = origin.reshape(3)
# #     x = o + R[:,0]*scale
# #     y = o + R[:,1]*scale
# #     z = o + R[:,2]*scale

# #     pts = np.vstack([o,x,o,y,o,z])
# #     lines = [[0,1],[2,3],[4,5]]
# #     colors = [[1,0,0],[0,1,0],[0,0,1]]  # X=red, Y=green, Z=blue

# #     ls = o3d.geometry.LineSet()
# #     ls.points = o3d.utility.Vector3dVector(pts)
# #     ls.lines  = o3d.utility.Vector2iVector(lines)
# #     ls.colors = o3d.utility.Vector3dVector(colors)
# #     return ls



# # def main():
# #     ap = argparse.ArgumentParser()
# #     ap.add_argument("--input", required=True, help="Raw foot point cloud (.ply/.pcd/.xyz) — same coords as A4 step")
# #     ap.add_argument("--calib", required=True, help="a4_calibration.txt (JSON) with centroid, R, scale_mm_per_unit")
# #     ap.add_argument("--out_dir", required=True)
# #     ap.add_argument("--voxel_mm", type=float, default=1.5)
# #     ap.add_argument("--nb_neighbors", type=int, default=40)
# #     ap.add_argument("--std_ratio", type=float, default=1.8)
# #     ap.add_argument("--slice_step_mm", type=float, default=0.5)
# #     ap.add_argument("--slice_thickness_mm", type=float, default=1.5)
# #     ap.add_argument("--alpha", type=float, default=None)
# #     ap.add_argument("--show_slices_on_model", action="store_true", help="Color and visualize point cloud by slice bands")
# #     ap.add_argument("--export_xyz_csv", action="store_true", help="Export aligned XYZ with slice metadata to CSV")
# #     ap.add_argument("--xyz_csv_path", type=str, default=None, help="Optional path for XYZ CSV; defaults to out_dir/03_points_aligned_mm.csv")
# #     args = ap.parse_args()

# #     ensure_dir(args.out_dir)

# #     # 1) Load & clean foot
# #     print("Step 1: Original Point Cloud")
# #     pcd = o3d.io.read_point_cloud(args.input)
# #     o3d.visualization.draw_geometries([pcd])

# #     if len(pcd.points) == 0:
# #         print("Empty point cloud.", file=sys.stderr); sys.exit(2)

# #     print("Step 2: Cleaned Point Cloud")
# #     # === Remove statistical outliers ===
# #     cl, ind = pcd.remove_statistical_outlier(nb_neighbors=15, std_ratio=1)
# #     cleaned_pcd = pcd.select_by_index(ind)
# #     print(f"After cleaning: {len(cleaned_pcd.points)} points remain.")
# #     o3d.visualization.draw_geometries([cleaned_pcd])

# #     # pcd = clean_point_cloud(pcd, args.voxel_mm, args.nb_neighbors, args.std_ratio)

# #     o3d.io.write_point_cloud(os.path.join(args.out_dir, "01_clean_preview.ply"), pcd)

# #     print("Step 3: Calibration")
# #     # 2) Load calibration (centroid, R, scale) from A4 step
# #     centroid, R, scale = load_calibration(args.calib)
# #     print("--------------------------------")
# #     print (centroid, R, scale)

# #     # 3) Transform foot into A4 PCA frame & scale to mm
# #     P = np.asarray(pcd.points)
# #     Pm = transform_and_scale(P, centroid, R, scale)
# #     pcd_mm = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pm))
# #     o3d.io.write_point_cloud(os.path.join(args.out_dir, "02_foot_aligned_mm.ply"), pcd_mm)

# # # this is new code 

# #     # Optional: export aligned XYZ (with slice metadata) to CSV for Excel review
# #     if args.export_xyz_csv:
# #         xvals = Pm[:, 0]
# #         vmin, vmax = xvals.min(), xvals.max()
# #         centers = np.arange(vmin, vmax + args.slice_step_mm, args.slice_step_mm)
# #         if centers.size == 0:
# #             centers = np.array([vmin])
# #         half = args.slice_thickness_mm / 2.0
# #         # nearest slice index per point and whether it lies within the band
# #         slice_idx = np.clip(np.round((xvals - vmin) / max(args.slice_step_mm, 1e-9)).astype(int), 0, len(centers) - 1)
# #         slice_center_mm = centers[slice_idx]
# #         in_slice_band = (np.abs(xvals - slice_center_mm) <= half).astype(int)
# #         csv_path_pts = args.xyz_csv_path or os.path.join(args.out_dir, "03_points_aligned_mm.csv")
# #         ensure_dir(os.path.dirname(csv_path_pts))
# #         with open(csv_path_pts, "w", newline="") as f:
# #             import csv as _csv
# #             w = _csv.writer(f)
# #             w.writerow(["point_index", "x_mm", "y_mm", "z_mm", "slice_index", "slice_center_mm", "in_slice_band"])
# #             for i, (x, y, z, si, sc, ins) in enumerate(zip(Pm[:, 0], Pm[:, 1], Pm[:, 2], slice_idx, slice_center_mm, in_slice_band)):
# #                 w.writerow([i, float(x), float(y), float(z), int(si), float(sc), int(ins)])
# # #  ends here
# #     # After you have Pm (aligned/scaled), visualize:
# #     pcd_vis = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pm))
# #     origin = np.zeros(3)
# #     axes = make_axes_from_R(origin, np.eye(3), scale=0.50)  # if already in A4 PCA frame (X length)
# #     # Or, if you want to recompute PCA on the foot for sanity:
# #     # mu,Rpca = compute_pca_axes(Pm)
# #     # axes = make_axes_from_R(mu, Rpca, scale=50.0)

# # #  this is new code 
# #     # Add a colored visualization of slice bands directly on the model if requested
# #     if args.show_slices_on_model:
# #         xvals = Pm[:, 0]
# #         vmin, vmax = xvals.min(), xvals.max()
# #         centers = np.arange(vmin, vmax + args.slice_step_mm, args.slice_step_mm)
# #         if centers.size == 0:
# #             centers = np.array([vmin])
# #         half = args.slice_thickness_mm / 2.0
# #         base_color = np.array([0.82, 0.82, 0.82])
# #         colors = np.tile(base_color, (Pm.shape[0], 1))
# #         cmap = cm.get_cmap("rainbow", len(centers))
# #         for i, c in enumerate(centers):
# #             mask = (xvals >= c - half) & (xvals <= c + half)
# #             if np.any(mask):
# #                 colors[mask] = np.array(cmap(i)[:3])
# #         pcd_col = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pm))
# #         pcd_col.colors = o3d.utility.Vector3dVector(colors)
# #         o3d.io.write_point_cloud(os.path.join(args.out_dir, "03_colored_slices_points.ply"), pcd_col)
# #         o3d.visualization.draw_geometries([pcd_col, axes])
# # #  ends here

# #     # Visualize your aligned/scaled points with axes
# #     # pcd_vis = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pm))
# #     o3d.visualization.draw_geometries([pcd_vis, axes])

# #     print("Step 4: Slicing")
# #     mins = Pm.min(axis=0); maxs = Pm.max(axis=0); ext = maxs - mins
# #     print("Scale (mm/unit):", scale)
# #     print(f"Extents (mm)  X:{ext[0]:.2f}  Y:{ext[1]:.2f}  Z:{ext[2]:.2f}")
# #         # 4) Slice every 0.5 mm along PC1 (X in this frame)
# #     slices = slice_cloud(Pm, step_mm=args.slice_step_mm, thickness_mm=args.slice_thickness_mm, axis=0)

# #     xs, perims, widths, areas = [], [], [], []
# #     outlines = {}
# #     for si, (pts, xc) in enumerate(slices):
# #         per, wid, ar, bxy = boundary_metrics(pts, axis=0, alpha_param=args.alpha)
# #         xs.append(xc); perims.append(per); widths.append(wid); areas.append(ar)
# #         if si % max(1, len(slices)//20) == 0 and bxy is not None:
# #             outlines[si] = bxy

# #     xs = np.array(xs); perims = np.array(perims); widths = np.array(widths); areas = np.array(areas)

# #     # 5) Detect ball
# #     ball_i = detect_ball_index(xs, perims, widths)

# #     # 6) Save CSV
# #     import csv
# #     csv_path = os.path.join(args.out_dir, "slices.csv")
# #     with open(csv_path, "w", newline="") as f:
# #         w = csv.writer(f)
# #         w.writerow(["x_mm","perimeter_mm","width_mm","area_mm2","npts"])
# #         for i in range(len(xs)):
# #             w.writerow([xs[i], perims[i], widths[i], areas[i], 0])

# #     # 7) Plots
# #     plot_curves(xs, perims, widths, os.path.join(args.out_dir, "05_girth_width_vs_X.png"), ball_i)

# #     # 8) Save outlines
# #     for si, bxy in outlines.items():
# #         plot_slice_outline(bxy, os.path.join(args.out_dir, f"04_slice_outline_{si}.png"),
# #                            f"Slice outline (YZ) at x={xs[si]:.1f} mm")
# #     if 0 <= ball_i < len(slices):
# #         per, wid, ar, bxy = boundary_metrics(slices[ball_i][0], axis=0, alpha_param=args.alpha)
# #         plot_slice_outline(bxy, os.path.join(args.out_dir, "06_ball_slice_outline.png"),
# #                            f"BALL slice outline at x={xs[ball_i]:.1f} mm\nPerim={perims[ball_i]:.1f} mm")

# #     # 9) Print result
# #     if 0 <= ball_i < len(xs):
# #         print(f"Ball slice at x = {xs[ball_i]:.2f} mm")
# #         print(f"Ball girth ≈ {perims[ball_i]:.2f} mm | Ball width ≈ {widths[ball_i]:.2f} mm")
# #     else:
# #         print("Ball not detected confidently. Check 05_girth_width_vs_X.png")



# # if __name__ == "__main__":
# #     main()
# #
# # code for regression from chatgpt: 
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from mpl_toolkits.mplot3d import Axes3D
# # from sklearn.linear_model import LinearRegression
# # from sklearn.preprocessing import PolynomialFeatures

# # # === Step 1: Load the data ===
# # data_str = '''-2.25238513947,2.74996256828,3.8438873291
# # -2.23491549492,2.74215102196,3.80290031433
# # -2.21566081047,2.72386407852,3.76206493378
# # -2.20410656929,2.70459580421,3.73236656189
# # -2.20499825478,2.70060539246,3.72757697105
# # -2.20187711716,2.69421195984,3.71903419495
# # -2.19925928116,2.68492770195,3.71384334564
# # -2.19189977646,2.67840766907,3.69955587387
# # -2.18563485146,2.66384363174,3.6857843399
# # -2.17848825455,2.64946651459,3.6631500721
# # -2.17110395432,2.63090324402,3.64872264862
# # -2.16112875938,2.62222886086,3.63206338882
# # -2.16112875938,2.62222886086,3.63206338882
# # -2.15803742409,2.60637712479,3.61951589584
# # -2.15361404419,2.58956193924,3.60943841934
# # -2.15175914764,2.58702421188,3.60529446602
# # -2.14438056946,2.58212661743,3.58882856369
# # -2.13857698441,2.5691242218,3.57575750351
# # -2.13305521011,2.56301474571,3.56967234612
# # -2.14022374153,2.53973913193,3.58483719826
# # -2.13887929916,2.5268945694,3.57919836044
# # -2.1365237236,2.51288151741,3.57740402222
# # -2.13121676445,2.50268793106,3.56435489655
# # -2.12999892235,2.49202847481,3.56005573273
# # -2.12842845917,2.4825155735,3.55264186859
# # -2.12409567833,2.47191357613,3.54553389549
# # -2.12040781975,2.47636342049,3.52410507202
# # -2.11511564255,2.4751598835,3.51181030273
# # -2.1147274971,2.46237587929,3.51208734512
# # -2.11768007278,2.42666220665,3.52478647232
# # -2.12025761604,2.36959648132,3.53596115112
# # -2.12430715561,2.34903001785,3.53886413574
# # -2.11725711823,2.32033634186,3.52662682533
# # -2.11882638931,2.29548621178,3.53010606766
# # -2.11144018173,2.25548553467,3.52350997925
# # -2.12900447845,2.22851920128,3.5362663269
# # -2.14177274704,2.20972728729,3.56015181541
# # -2.14380383492,2.18781685829,3.55813121796
# # -2.15467166901,2.16464734077,3.57044267654
# # -2.17306423187,2.16064786911,3.60872983932
# # -2.16987466812,2.13906860352,3.60319757462
# # -2.17821884155,2.12316894531,3.61932468414
# # -2.18323540688,2.1010556221,3.62833094597
# # -2.19918608665,2.09945011139,3.66640520096
# # -2.20523643494,2.07441949844,3.66389083862
# # -2.21055793762,2.0621676445,3.67612051964
# # -2.2109234333,2.0425028801,3.67270040512
# # -2.21497344971,2.02492856979,3.68134951591
# # -2.21667051315,2.01553964615,3.69145536423
# # -2.21500205994,2.00785970688,3.69865608215'''

# # # Convert to NumPy array
# # points = np.array([list(map(float, line.split(','))) for line in data_str.strip().splitlines()])

# # # === Step 2: Fit a regression line along the curvature ===
# # # We'll use polynomial regression (degree=3) on the arc-length parameterization

# # # Arc-length parameter
# # s = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
# # s = np.insert(s, 0, 0)  # include first point as 0

# # # Polynomial regression
# # degree = 3
# # poly = PolynomialFeatures(degree)
# # S_poly = poly.fit_transform(s.reshape(-1, 1))
# # model = LinearRegression().fit(S_poly, points)

# # # Predict curve
# # s_dense = np.linspace(s.min(), s.max(), 300)
# # S_dense_poly = poly.transform(s_dense.reshape(-1, 1))
# # curve = model.predict(S_dense_poly)

# # # === Step 3: Plot the original points and the regression curve ===
# # fig = plt.figure(figsize=(10, 6))
# # ax = fig.add_subplot(111, projection='3d')
# # ax.plot(points[:, 0], points[:, 1], points[:, 2], 'o', label='Original Points', alpha=0.5)
# # ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], '-', label='Regression Curve', color='red')
# # ax.set_title('3D Polynomial Regression Curve (Degree 3)')
# # ax.legend()
# # plt.tight_layout()

# # plt.show()



# #!/usr/bin/env python3
# import argparse, os, sys, json
# import numpy as np
# import open3d as o3d
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.pipeline import Pipeline

# try:
#     import alphashape
#     from shapely.geometry import Polygon
#     HAVE_ALPHA = True
# except Exception:
#     from scipy.spatial import ConvexHull
#     HAVE_ALPHA = False

# def ensure_dir(d): os.makedirs(d, exist_ok=True)

# def load_calibration(path):
#     with open(path, "r") as f:
#         data = json.load(f)
#     R = np.array(data["R"], dtype=float)
#     centroid = np.array(data["centroid"], dtype=float)
#     scale = float(data["scale_mm_per_unit"])
#     return centroid, R, scale

# def clean_point_cloud_gentle(pcd, nb_neighbors=15, std_ratio=1.0):
#     """Gentle outlier removal"""
#     print(f"Cleaning point cloud: {len(pcd.points)} points initially")
#     cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
#     cleaned_pcd = pcd.select_by_index(ind)
#     print(f"After gentle cleaning: {len(cleaned_pcd.points)} points remain")
#     return cleaned_pcd

# def transform_point_cloud(points, centroid, R, scale_mm):
#     """Transform: center -> rotate -> scale"""
#     print(f"Transforming {len(points)} points")
#     print(f"Centroid: {centroid}")
#     print(f"Scale: {scale_mm} mm/unit")
#     print(f"Rotation matrix shape: {R.shape}")
    
#     # Step 1: Center
#     centered = points - centroid
#     print(f"Centering: min={centered.min(axis=0)}, max={centered.max(axis=0)}")
    
#     # Step 2: Rotate
#     rotated = centered @ R
#     print(f"Rotating: min={rotated.min(axis=0)}, max={rotated.max(axis=0)}")
    
#     # Step 3: Scale
#     scaled = rotated * scale_mm
#     print(f"Scaling: min={scaled.min(axis=0)}, max={scaled.max(axis=0)}")
    
#     return scaled

# def create_non_overlapping_slices(points_xyz_mm, slice_width_mm=0.5, slice_thickness_mm=0.25, debug_out_dir=None, debug_max_plots=5, overlapping_slices=False):
#     """Create non-overlapping slices along X-axis"""
#     print(f"\nCreating slices with width={slice_width_mm}mm, thickness={slice_thickness_mm}mm")
    
#     # Sort points by X value
#     x_sorted_indices = np.argsort(points_xyz_mm[:, 0])
#     points_sorted = points_xyz_mm[x_sorted_indices]
    
#     print(f"X range: {points_sorted[0, 0]:.3f} to {points_sorted[-1, 0]:.3f} mm")
    
#     slices = []
#     used_points = set()
#     x_min = points_sorted[0, 0]
#     x_max = points_sorted[-1, 0]
    
#     # Create slice centers
#     slice_centers = np.arange(x_min + slice_width_mm/2, x_max - slice_width_mm/2, slice_width_mm)
#     print(f"Creating {len(slice_centers)} slices")

#     # Track how many debug plots we have saved
#     num_debug_plots = 0
#     modul_of_debug_plots = max(1, len(slice_centers) // max(1, debug_max_plots))
#     print (modul_of_debug_plots)
#     print (len(slice_centers))
#     for i, center_x in enumerate(slice_centers):
#         # Find points within this slice (X ± thickness/2)
#         half_thickness = slice_thickness_mm / 2.0
#         in_slice_mask = (points_sorted[:, 0] >= center_x - half_thickness) & \
#                        (points_sorted[:, 0] <= center_x + half_thickness)
        
#         # Get slice points (only unused points)
#         slice_point_indices = np.where(in_slice_mask)[0]
#         unused_in_slice = [idx for idx in slice_point_indices if idx not in used_points]
        
#         if len(unused_in_slice) >= 10:  # Minimum points for regression
#             slice_points = points_sorted[unused_in_slice]
#             # Map sorted indices back to original indices for downstream usage (e.g., coloring)
#             orig_indices = x_sorted_indices[unused_in_slice]
#             slices.append({
#                 'center_x': center_x,
#                 'points': slice_points,
#                 'point_indices': orig_indices.tolist()
#             })
#             # Mark these points as used (track positions within the sorted array)
#             if overlapping_slices:
#                 used_points.update(unused_in_slice)
#             print(f"Slice {i}: center_x={center_x:.3f}, {len(unused_in_slice)} points")

#             # Optional debug plot of selected points only (YZ scatter)
#             if debug_out_dir is not None and num_debug_plots < debug_max_plots and i % modul_of_debug_plots == 0:
#                 save_debug_slice_plot(slice_points, center_x, debug_out_dir, num_debug_plots)
#                 num_debug_plots += 1
    
#     print(f"Total slices created: {len(slices)}")
#     return slices

# def fit_regression_to_slice(slice_data, degree=2):
#     """Fit polynomial regression to YZ points in a slice"""
#     points = slice_data['points']
#     center_x = slice_data['center_x']
    
#     if len(points) < degree + 1:
#         print(f"Warning: Not enough points ({len(points)}) for degree {degree} regression")
#         return None
    
#     # Extract YZ coordinates
#     YZ = points[:, [1, 2]]  # Y and Z coordinates
    
#     # Fit polynomial regression: Z = f(Y)
#     y_coords = YZ[:, 0].reshape(-1, 1)  # Y coordinates
#     z_coords = YZ[:, 1]  # Z coordinates
    
#     # Create polynomial features
#     poly_features = PolynomialFeatures(degree=degree, include_bias=False)
#     y_poly = poly_features.fit_transform(y_coords)
    
#     # Fit linear regression
#     reg = LinearRegression()
#     reg.fit(y_poly, z_coords)
    
#     # Store regression info
#     regression_info = {
#         'center_x': center_x,
#         'coefficients': reg.coef_,
#         'intercept': reg.intercept_,
#         'degree': degree,
#         'n_points': len(points),
#         'y_range': [YZ[:, 0].min(), YZ[:, 0].max()],
#         'z_range': [YZ[:, 1].min(), YZ[:, 1].max()]
#     }
    
#     print(f"Regression at x={center_x:.3f}: {len(points)} points, degree={degree}")
#     print(f"  Y range: {regression_info['y_range'][0]:.3f} to {regression_info['y_range'][1]:.3f}")
#     print(f"  Z range: {regression_info['z_range'][0]:.3f} to {regression_info['z_range'][1]:.3f}")
    
#     return regression_info

# def visualize_slices_3d(points_xyz_mm, slices, out_dir):
#     """Visualize slices in 3D with different colors"""
#     print("\nCreating 3D slice visualization...")
    
#     # Create colored point cloud
#     colors = np.ones((len(points_xyz_mm), 3)) * 0.7  # Default gray
    
#     # Color each slice differently
#     cmap = cm.get_cmap('rainbow', len(slices))
#     for i, slice_data in enumerate(slices):
#         slice_indices = slice_data['point_indices']
#         color = np.array(cmap(i)[:3])
#         colors[slice_indices] = color
    
#     # Create point cloud
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points_xyz_mm)
#     pcd.colors = o3d.utility.Vector3dVector(colors)
    
#     # Save colored point cloud
#     colored_ply_path = os.path.join(out_dir, "03_slices_colored_3d.ply")
#     o3d.io.write_point_cloud(colored_ply_path, pcd)
#     print(f"Saved colored slice visualization: {colored_ply_path}")
    
#     # Show in viewer
#     o3d.visualization.draw_geometries([pcd])
    
#     return pcd

# def visualize_regressions_2d(slices, regressions, out_dir):
#     """Visualize regression fits for each slice"""
#     print("\nCreating 2D regression visualizations...")
    
#     n_slices = len(slices)
#     fig, axes = plt.subplots(2, (n_slices + 1) // 2, figsize=(15, 8))
#     if n_slices == 1:
#         axes = [axes]
#     else:
#         axes = axes.flatten()
    
#     for i, (slice_data, regression) in enumerate(zip(slices, regressions)):
#         if regression is None:
#             continue
            
#         ax = axes[i]
#         points = slice_data['points']
#         YZ = points[:, [1, 2]]
        
#         # Plot original points
#         ax.scatter(YZ[:, 0], YZ[:, 1], alpha=0.6, s=10, label='Points')
        
#         # Plot regression line
#         y_min, y_max = regression['y_range']
#         y_line = np.linspace(y_min, y_max, 100).reshape(-1, 1)
        
#         # Apply polynomial transformation
#         poly_features = PolynomialFeatures(degree=regression['degree'], include_bias=False)
#         y_poly = poly_features.fit_transform(y_line)
        
#         # Predict Z values
#         z_line = y_poly @ regression['coefficients'] + regression['intercept']
        
#         ax.plot(y_line.flatten(), z_line, 'r-', linewidth=2, label='Regression')
        
#         ax.set_xlabel('Y (mm)')
#         ax.set_ylabel('Z (mm)')
#         ax.set_title(f'Slice at X={slice_data["center_x"]:.1f}mm\n{regression["n_points"]} points')
#         ax.legend()
#         ax.grid(True, alpha=0.3)
    
#     # Hide unused subplots
#     for i in range(len(slices), len(axes)):
#         axes[i].set_visible(False)
    
#     plt.tight_layout()
#     regression_plot_path = os.path.join(out_dir, "04_regression_fits.png")
#     plt.savefig(regression_plot_path, dpi=150, bbox_inches='tight')
#     plt.close()
#     print(f"Saved regression visualization: {regression_plot_path}")

# def visualize_regressions_3d_stacked(slices, regressions, out_dir):
#     """Visualize all regressions stacked together in 3D space"""
#     print("\nCreating 3D stacked regression visualization...")
    
#     fig = plt.figure(figsize=(12, 10))
#     ax = fig.add_subplot(111, projection='3d')
    
#     # Color map for different slices
#     cmap = cm.get_cmap('viridis', len(slices))
    
#     # Collect all points and lines for PLY export
#     all_points = []
#     all_colors = []
#     all_lines = []
#     point_idx = 0
    
#     for i, (slice_data, regression) in enumerate(zip(slices, regressions)):
#         if regression is None:
#             continue
            
#         center_x = slice_data['center_x']
#         color = cmap(i)
        
#         # Get Y range for this slice
#         y_min, y_max = regression['y_range']
#         y_line = np.linspace(y_min, y_max, 50).reshape(-1, 1)
        
#         # Apply polynomial transformation
#         poly_features = PolynomialFeatures(degree=regression['degree'], include_bias=False)
#         y_poly = poly_features.fit_transform(y_line)
        
#         # Predict Z values
#         z_line = y_poly @ regression['coefficients'] + regression['intercept']
        
#         # Create X coordinates (all same value for this slice)
#         x_line = np.full_like(y_line, center_x)
        
#         # Plot the regression curve in 3D
#         ax.plot(x_line.flatten(), y_line.flatten(), z_line, 
#                 color=color, linewidth=2, alpha=0.8,
#                 label=f'X={center_x:.1f}mm')
        
#         # Add a small scatter of original points for reference
#         points = slice_data['points']
#         ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
#                   c=[color], s=5, alpha=0.6)
        
#         # Collect points and lines for PLY export
#         curve_points = np.column_stack([x_line.flatten(), y_line.flatten(), z_line.flatten()])
#         all_points.extend(curve_points)
        
#         # Add colors for each point (repeat the same color for all points in this curve)
#         curve_color = np.array(color[:3])  # RGB values
#         all_colors.extend([curve_color] * len(curve_points))
        
#         # Add line segments for this curve
#         for j in range(len(curve_points) - 1):
#             all_lines.append([point_idx + j, point_idx + j + 1])
        
#         point_idx += len(curve_points)
    
#     # Customize the plot
#     ax.set_xlabel('X (Length, mm)')
#     ax.set_ylabel('Y (Width, mm)')
#     ax.set_zlabel('Z (Height, mm)')
#     ax.set_title('3D Stacked Regression Curves\nFoot Cross-Sections Along Length')
    
#     # Add legend (show only some slices to avoid clutter)
#     legend_elements = []
#     step = max(1, len(slices) // 10)  # Show ~10 legend entries
#     for i in range(0, len(slices), step):
#         if regressions[i] is not None:
#             color = cmap(i)
#             legend_elements.append(plt.Line2D([0], [0], color=color, 
#                                             label=f'X={slices[i]["center_x"]:.1f}mm'))
#     ax.legend(handles=legend_elements, loc='upper right')
    
#     # Set equal aspect ratio for better visualization
#     ax.set_box_aspect([1, 1, 1])
    
#     plt.tight_layout()
#     regression_3d_path = os.path.join(out_dir, "05_regressions_3d_stacked.png")
#     plt.savefig(regression_3d_path, dpi=150, bbox_inches='tight')
#     plt.close()
#     print(f"Saved 3D stacked regression visualization: {regression_3d_path}")
    
#     # Save as PLY file
#     if all_points:
#         all_points = np.array(all_points)
#         all_colors = np.array(all_colors)
#         all_lines = np.array(all_lines)
        
#         # Create PLY file
#         ply_path = os.path.join(out_dir, "06_regressions_3d_stacked.ply")
#         save_regression_curves_as_ply(all_points, all_colors, all_lines, ply_path)
#         print(f"Saved 3D regression curves as PLY: {ply_path}")
    
#     # Also create an interactive 3D plot using matplotlib
#     print("Creating interactive 3D plot...")
#     fig_interactive = plt.figure(figsize=(12, 10))
#     ax_interactive = fig_interactive.add_subplot(111, projection='3d')
    
#     # Same plotting logic but for interactive version
#     for i, (slice_data, regression) in enumerate(zip(slices, regressions)):
#         if regression is None:
#             continue
            
#         center_x = slice_data['center_x']
#         color = cmap(i)
        
#         y_min, y_max = regression['y_range']
#         y_line = np.linspace(y_min, y_max, 50).reshape(-1, 1)
        
#         poly_features = PolynomialFeatures(degree=regression['degree'], include_bias=False)
#         y_poly = poly_features.fit_transform(y_line)
#         z_line = y_poly @ regression['coefficients'] + regression['intercept']
#         x_line = np.full_like(y_line, center_x)
        
#         ax_interactive.plot(x_line.flatten(), y_line.flatten(), z_line, 
#                            color=color, linewidth=2, alpha=0.8)
    
#     ax_interactive.set_xlabel('X (Length, mm)')
#     ax_interactive.set_ylabel('Y (Width, mm)')
#     ax_interactive.set_zlabel('Z (Height, mm)')
#     ax_interactive.set_title('Interactive 3D: Foot Cross-Sections\n(Rotate with mouse)')
#     ax_interactive.set_box_aspect([1, 1, 1])
    
#     plt.tight_layout()
#     print("Showing interactive 3D plot (close window to continue)...")
#     plt.show()
    
#     return fig_interactive

# def save_regression_curves_as_ply(points, colors, lines, ply_path):
#     """Save regression curves as a colored PLY file"""
#     ensure_dir(os.path.dirname(ply_path))
    
#     with open(ply_path, 'w') as f:
#         # PLY header
#         f.write("ply\n")
#         f.write("format ascii 1.0\n")
#         f.write(f"element vertex {len(points)}\n")
#         f.write("property float x\n")
#         f.write("property float y\n")
#         f.write("property float z\n")
#         f.write("property uchar red\n")
#         f.write("property uchar green\n")
#         f.write("property uchar blue\n")
#         f.write(f"element edge {len(lines)}\n")
#         f.write("property int vertex1\n")
#         f.write("property int vertex2\n")
#         f.write("end_header\n")
        
#         # Write vertices with colors
#         for i, (point, color) in enumerate(zip(points, colors)):
#             r, g, b = (color * 255).astype(int)
#             f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {r} {g} {b}\n")
        
#         # Write edges
#         for line in lines:
#             f.write(f"{line[0]} {line[1]}\n")
    
#     print(f"PLY file saved with {len(points)} vertices and {len(lines)} edges")

# # Debug helper: save a YZ scatter plot for a slice's selected points only
# def save_debug_slice_plot(slice_points, center_x, out_dir, slice_idx):
#     ensure_dir(out_dir)
#     yz = slice_points[:, [1, 2]]
#     plt.figure(figsize=(4.5, 4.5))
#     plt.scatter(yz[:, 0], yz[:, 1], s=8, alpha=0.8)
#     plt.gca().set_aspect('equal', 'box')
#     plt.xlabel('Y (mm)')
#     plt.ylabel('Z (mm)')
#     plt.title(f'Selected points at X={center_x:.2f} mm (n={len(yz)})')
#     plt.grid(True, alpha=0.3)
#     out_path = os.path.join(out_dir, f"debug_slice_{slice_idx+1}_x_{center_x:.1f}.png")
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=160)
#     plt.close()
#     print(f"Saved debug slice plot: {out_path}")

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--input", required=True, help="Raw foot point cloud (.ply/.pcd/.xyz)")
#     ap.add_argument("--calib", required=True, help="a4_calibration.txt (JSON) with centroid, R, scale_mm_per_unit")
#     ap.add_argument("--out_dir", required=True)
#     ap.add_argument("--slice_width_mm", type=float, default=0.5, help="Width between slice centers")
#     ap.add_argument("--slice_thickness_mm", type=float, default=0.25, help="Thickness of each slice")
#     ap.add_argument("--regression_degree", type=int, default=2, help="Polynomial degree for regression")
#     ap.add_argument("--export_xyz_csv", action="store_true", help="Export transformed XYZ to CSV")
#     ap.add_argument("--overlapping_slices", action="store_true", help="Use overlapping slices")
#     args = ap.parse_args()

#     ensure_dir(args.out_dir)
    
#     print("=" * 60)
#     print("FOOT BALL GIRTH ANALYSIS - NEW APPROACH")
#     print("=" * 60)

#     # Step 1: Load point cloud
#     print("\nSTEP 1: Loading point cloud")
#     pcd = o3d.io.read_point_cloud(args.input)
#     print(f"Loaded point cloud: {len(pcd.points)} points")
    
#     if len(pcd.points) == 0:
#         print("ERROR: Empty point cloud!", file=sys.stderr)
#         sys.exit(1)
    
#     # Visualize original
#     print("Showing original point cloud...")
#     o3d.visualization.draw_geometries([pcd])

#     # Step 2: Gentle cleaning
#     print("\nSTEP 2: Gentle outlier removal")
#     cleaned_pcd = clean_point_cloud_gentle(pcd)
    
#     # Visualize cleaned
#     print("Showing cleaned point cloud...")
#     o3d.visualization.draw_geometries([cleaned_pcd])
    
#     # Save cleaned
#     cleaned_ply_path = os.path.join(args.out_dir, "01_cleaned.ply")
#     o3d.io.write_point_cloud(cleaned_ply_path, cleaned_pcd)
#     print(f"Saved cleaned point cloud: {cleaned_ply_path}")

#     # Step 3: Load calibration
#     print("\nSTEP 3: Loading calibration")
#     centroid, R, scale = load_calibration(args.calib)
#     print(f"Calibration loaded successfully")

#     # Step 4: Transform points
#     print("\nSTEP 4: Transforming point cloud")
#     points = np.asarray(cleaned_pcd.points)
#     points_transformed = transform_point_cloud(points, centroid, R, scale)
    
#     # Sanity check
#     print("\nSanity check - Transformed point ranges:")
#     for i, axis in enumerate(['X', 'Y', 'Z']):
#         min_val, max_val = points_transformed[:, i].min(), points_transformed[:, i].max()
#         print(f"  {axis}: {min_val:.3f} to {max_val:.3f} mm")
    
#     # Create transformed point cloud
#     pcd_transformed = o3d.geometry.PointCloud()
#     pcd_transformed.points = o3d.utility.Vector3dVector(points_transformed)
    
#     # Visualize transformed
#     print("Showing transformed point cloud...")
#     o3d.visualization.draw_geometries([pcd_transformed])
    
#     # Save transformed
#     transformed_ply_path = os.path.join(args.out_dir, "02_transformed.ply")
#     o3d.io.write_point_cloud(transformed_ply_path, pcd_transformed)
#     print(f"Saved transformed point cloud: {transformed_ply_path}")

#     # Step 5: Create non-overlapping slices
#     print("\nSTEP 5: Creating non-overlapping slices")
#     slices = create_non_overlapping_slices(points_transformed, 
#                                          args.slice_width_mm, 
#                                          args.slice_thickness_mm,
#                                          debug_out_dir=args.out_dir,
#                                          debug_max_plots=5,
#                                          overlapping_slices=args.overlapping_slices)
    
#     if len(slices) == 0:
#         print("ERROR: No slices created!", file=sys.stderr)
#         sys.exit(1)

#     # Step 6: Fit regressions to each slice
#     print("\nSTEP 6: Fitting regressions to slices")
#     regressions = []
#     for i, slice_data in enumerate(slices):
#         print(f"\nProcessing slice {i+1}/{len(slices)}")
#         regression = fit_regression_to_slice(slice_data, args.regression_degree)
#         regressions.append(regression)
    
#     # Count successful regressions
#     successful_regressions = sum(1 for r in regressions if r is not None)
#     print(f"\nRegression summary: {successful_regressions}/{len(slices)} successful fits")

#     # Step 7: Visualize slices in 3D
#     print("\nSTEP 7: 3D slice visualization")
#     visualize_slices_3d(points_transformed, slices, args.out_dir)

#     # Step 8: Visualize regressions in 2D
#     print("\nSTEP 8: 2D regression visualization")
#     visualize_regressions_2d(slices, regressions, args.out_dir)

#     # Step 9: Visualize regressions in 3D stacked
#     print("\nSTEP 9: 3D stacked regression visualization")
#     visualize_regressions_3d_stacked(slices, regressions, args.out_dir)

#     # Step 10: Export data if requested
#     if args.export_xyz_csv:
#         print("\nSTEP 10: Exporting XYZ data to CSV")
#         csv_path = os.path.join(args.out_dir, "05_transformed_xyz.csv")
#         with open(csv_path, "w", newline="") as f:
#             import csv
#             w = csv.writer(f)
#             w.writerow(["point_index", "x_mm", "y_mm", "z_mm"])
#             for i, (x, y, z) in enumerate(points_transformed):
#                 w.writerow([i, float(x), float(y), float(z)])
#         print(f"Saved XYZ data: {csv_path}")

#     # Step 11: Save regression data
#     print("\nSTEP 11: Saving regression data")
#     regression_data = []
#     for slice_data, regression in zip(slices, regressions):
#         if regression is not None:
#             data_row = {
#                 'slice_index': len(regression_data),
#                 'center_x_mm': regression['center_x'],
#                 'n_points': regression['n_points'],
#                 'degree': regression['degree'],
#                 'coefficients': regression['coefficients'].tolist(),
#                 'intercept': regression['intercept'],
#                 'y_min': regression['y_range'][0],
#                 'y_max': regression['y_range'][1],
#                 'z_min': regression['z_range'][0],
#                 'z_max': regression['z_range'][1]
#             }
#             regression_data.append(data_row)
    
#     # Save regression data as JSON
#     regression_json_path = os.path.join(args.out_dir, "06_regression_data.json")
#     with open(regression_json_path, "w") as f:
#         json.dump(regression_data, f, indent=2)
#     print(f"Saved regression data: {regression_json_path}")
    
#     # Save regression data as CSV
#     regression_csv_path = os.path.join(args.out_dir, "07_regression_data.csv")
#     with open(regression_csv_path, "w", newline="") as f:
#         import csv
#         w = csv.writer(f)
#         w.writerow(["slice_index", "center_x_mm", "n_points", "degree", "intercept", 
#                    "coef_1", "coef_2", "coef_3", "y_min", "y_max", "z_min", "z_max"])
#         for data in regression_data:
#             coefs = data['coefficients']
#             # Pad coefficients to 3 elements
#             coefs_padded = coefs + [0.0] * (3 - len(coefs))
#             w.writerow([data['slice_index'], data['center_x_mm'], data['n_points'], 
#                        data['degree'], data['intercept']] + coefs_padded + 
#                       [data['y_min'], data['y_max'], data['z_min'], data['z_max']])
#     print(f"Saved regression CSV: {regression_csv_path}")

#     print("\n" + "=" * 60)
#     print("ANALYSIS COMPLETE!")
#     print(f"Results saved in: {args.out_dir}")
#     print(f"Total slices processed: {len(slices)}")
#     print(f"Successful regressions: {successful_regressions}")
#     print("=" * 60)

# if __name__ == "__main__":
#     main()





    

# # Existing imports (ensure these are included)
# import open3d as o3d
# import argparse  # Added for command-line arguments
# import sys
# import os
# import json
# import numpy as np
# from scipy.spatial import ConvexHull
# try:
#     import alphashape
#     from shapely.geometry import Polygon
#     HAVE_ALPHA = True
# except ImportError:
#     HAVE_ALPHA = False

# # Your existing ensure_dir function
# def ensure_dir(d): os.makedirs(d, exist_ok=True)

# def points_between_planes(points, normal, d_center, thickness):
#     """
#     Find points between two parallel planes with given normal, centered at 'd_center', with thickness.
#     Returns: Boolean mask for points in the slab.
#     """
#     half_thickness = thickness / 2.0
#     # d_center = d_center  # Plane equation: normal · x + d = 0
#     d1 = d_center - half_thickness    # Back plane
#     d2 = d_center + half_thickness    # Front plane
#     f1 = np.dot(points, normal)  # Distance to back plane
#     # f2 = np.dot(points, normal) + d2  # Distance to front plane

#     print(points[1508987])
#     return (f1 < d1) & (f1 >d2) | (f1 < d2) & (f1 >d1) # Points between planes
#     # Calculate distance from each point to the center plane
#     # For plane equation: normal · x = d_center
#     # distances = np.dot(points, normal) 
#     # print(distances)
#     # print(distances <= half_thickness)
#     # # Points are between the two planes if |distance| <= half_thickness
#     # in_slab = np.abs(distances) <= half_thickness
    
#     # return in_slab

# def plane_equation_parameters_extraction(selected_points): 
#     p1 = np.array(selected_points[0])
#     p2 = np.array(selected_points[1])
#     p3 = np.array(selected_points[2])
#     print(p1)
#     print(p2)
#     print(p3)

#     # Create two vectors in the plane
#     v1 = p2 - p1
#     v2 = p3 - p1
#     normal_vector = np.cross(v1, v2)
#     a_x, b_y, c_z = normal_vector
#     d1 = np.dot(normal_vector, p1)
#     return a_x, b_y, c_z, d1
# # Semi-manual measurement function
# def semi_manual_measure(pcd_mm, slice_thickness_mm=10, out_dir='.'):
#     """
#     Semi-manual measurement: Shoemaker selects points to define a plane (e.g., ball),
#     then auto-computes width and girth from the slice.
#     """
#     print("Opening visualizer: Select 3 points to define a measurement plane (e.g., ball area).")
#     # vis = o3d.visualization.VisualizerWithVertexSelection()  
#     vis = o3d.visualization.VisualizerWithEditing()
#     # vis.create_window()
#     # vis.add_geometry(pcd_mm)
#     # vis.run() # This displays the window and allows user interacti
#     # vis.destroy_window()
#     # print("")
#     # picked_points = vis.get_picked_points()
    
#     vis.create_window("Select Points for Plane (e.g., Ball Area)")
#     vis.add_geometry(pcd_mm)
#     vis.run()  # Shoemaker selects points
#     vis.destroy_window()
    
#     picked_indices = vis.get_picked_points()

#     points = np.asarray(pcd_mm.points)
#     if len(picked_indices) < 3:
#         print("Need at least 3 points. Using default slice at midpoint.")
#         normal = np.array([1.0, 0.0, 0.0])  # X-axis normal
#         center_x = np.mean(points[:, 0])
#         d_1 = center_x  # For X-axis slicing
#     else:
#         selected_points = points[picked_indices]
#         a_x, b_y, c_z, d_1 = plane_equation_parameters_extraction(selected_points)
#         print(f"Selected points: {selected_points}")
#         print(f"Plane equation: {a_x:.3f}x + {b_y:.3f}y + {c_z:.3f}z = {d_1:.3f}")
        
#         # Create proper normal vector array
#         normal = np.array([a_x, b_y, c_z])
#         # Normalize the normal vector
#         # normal = normal / np.linalg.norm(normal)
        
#         # Calculate center_x for visualization (project center of selected points onto X-axis)
#         # center = np.mean(selected_points, axis=0)
#         # center_x = center[0]
    
#     # Find points in slab using corrected plane logic
#     slice_mask = points_between_planes(points, normal, d_1, slice_thickness_mm)
#     slice_pts = points[slice_mask]
#     print(f"Slice contains {len(slice_pts)} points")
    
#     if len(slice_pts) < 10:
#         print("Too few points in slice. Try selecting again.")
#         return None, None
    
#     # Compute width (Y-range)
#     width = slice_pts[:, 1].max() - slice_pts[:, 1].min()
    
#     # Compute girth (perimeter via alphashape or convex hull)
#     P = slice_pts[:, [1, 2]]  # YZ projection
#     if HAVE_ALPHA and len(P) >= 4:
#         alpha_param = 2.0  # Tune based on testing
#         ashape = alphashape.alphashape(P, alpha_param)
#         if not ashape.is_empty:
#             geom = ashape if ashape.geom_type != "MultiPolygon" else max(ashape.geoms, key=lambda g: g.area)
#             girth = geom.length
#         else:
#             girth = _convex_perimeter(P)
#     else:
#         girth = _convex_perimeter(P)
    
#     # Visualize slice
#     slice_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(slice_pts))
#     slice_pcd.paint_uniform_color([1, 1, 1])  # Red color for slice
    
#     # Create two bounding planes using normal vector and uniform point arrays
#     print("Creating bounding planes using normal vector...")
    
#     # Get foot bounding box to size the planes
#     bbox = pcd_mm.get_axis_aligned_bounding_box()
#     bbox_extent = bbox.get_extent()
    
#     # Create uniform grid for plane visualization
#     grid_size = max(bbox_extent[1], bbox_extent[2]) * 1.2  # Make planes larger than foot
#     grid_step = 1.0  # 2mm spacing between grid points
    
#     # Create uniform Y,Z grid
#     y_range = np.arange(-grid_size/2, grid_size/2, grid_step)
#     z_range = np.arange(-grid_size/2, grid_size/2, grid_step)
#     Y, Z = np.meshgrid(y_range, z_range)
    
#     # For each plane, solve for X using the plane equation: normal · [x,y,z] = d
#     # normal[0]*x + normal[1]*y + normal[2]*z = d
#     # So: x = (d - normal[1]*y - normal[2]*z) / normal[0]
    
#     def create_plane_points(normal, d_value, Y_grid, Z_grid):
#         """Create points that satisfy the plane equation normal · [x,y,z] = d"""
#         # Avoid division by zero - handle case where normal[0] is very small
#         # if abs(normal[0]) > 1e-6:
#             # Solve for X: x = (d - normal[1]*y - normal[2]*z) / normal[0]
#         X = (d_value - normal[1]*Y_grid - normal[2]*Z_grid) / normal[0]
#         # elif abs(normal[1]) > 1e-6:
#         #     # Solve for Y: y = (d - normal[0]*x - normal[2]*z) / normal[1]
#         #     X = np.full_like(Y_grid, 0)  # Place at x=0
#         #     Y_grid = (d_value - normal[0]*X - normal[2]*Z_grid) / normal[1]
#         # else:
#         #     # Solve for Z: z = (d - normal[0]*x - normal[1]*y) / normal[2]
#         #     X = np.full_like(Y_grid, 0)  # Place at x=0
#         #     Z_grid = (d_value - normal[0]*X - normal[1]*Y_grid) / normal[2]
        
#         # Stack into 3D points
#         plane_points = np.column_stack([X.flatten(), Y_grid.flatten(), Z_grid.flatten()])
#         return plane_points
    
#     # Create the two bounding planes
#     half_thickness = slice_thickness_mm / 2.0
#     d1 = d_1 - half_thickness  # Back plane
#     d2 = d_1 + half_thickness  # Front plane
    
#     # Generate plane points
#     back_plane_points = create_plane_points(normal, d1, Y, Z)
#     front_plane_points = create_plane_points(normal, d2, Y, Z)
    
#     # Create point clouds for the planes
#     back_plane_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(back_plane_points))
#     front_plane_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(front_plane_points))
    
#     # Color the planes
#     back_plane_pcd.paint_uniform_color([0, 1, 0])    # Green for back plane
#     front_plane_pcd.paint_uniform_color([0, 0, 1])   # Blue for front plane
    
#     # Show slice with original point cloud and bounding planes
#     print("Showing selected slice (white points) with bounding planes (green=back, blue=front)...")
#     o3d.visualization.draw_geometries([pcd_mm, slice_pcd, back_plane_pcd, front_plane_pcd], 
#                                      window_name=f"Slice at X={1:.2f} mm with Bounding Planes")
    
#     # Save measurements
#     ensure_dir(out_dir)
#     csv_path = os.path.join(out_dir, "measurements.csv")
#     with open(csv_path, "a") as f:
#         import csv
#         writer = csv.writer(f)
#         if f.tell() == 0:  # Write header if file is empty
#             writer.writerow(["Slice Center (mm)", "Width (mm)", "Girth/Perimeter (mm)"])
#         writer.writerow([center_x, width, girth])
    
#     print(f"Measurements at X={center_x:.2f} mm: Width = {width:.2f} mm, Girth = {girth:.2f} mm")
#     return width, girth

# def _convex_perimeter(P):
#     if len(P) < 3:
#         return 0.0
#     hull = ConvexHull(P)
#     poly = Polygon(P[hull.vertices])
#     return poly.length

# def load_calibration(path):
#     """Load calibration data from JSON file"""
#     with open(path, "r") as f:
#         data = json.load(f)
#     R = np.array(data["R"], dtype=float)
#     centroid = np.array(data["centroid"], dtype=float)
#     scale = float(data["scale_mm_per_unit"])
#     return centroid, R, scale

# # Update your main() function
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--input", required=True, help="Raw foot point cloud (.ply/.pcd/.xyz)")
#     ap.add_argument("--calib", required=True, help="a4_calibration.txt (JSON)")
#     ap.add_argument("--out_dir", required=True)
#     ap.add_argument("--slice_thickness_mm", type=float, default=1.5, help="Slice thickness for measurements")
#     args = ap.parse_args()

#     ensure_dir(args.out_dir)
    
#     # Load and process point cloud (your existing code)
#     print("Loading point cloud...")
#     pcd = o3d.io.read_point_cloud(args.input)
#     if len(pcd.points) == 0:
#         print("ERROR: Empty point cloud!", file=sys.stderr)
#         sys.exit(1)
    
#     # Clean (using your gentle cleaning)
#     print("Cleaning point cloud...")
#     cl, ind = pcd.remove_statistical_outlier(nb_neighbors=15, std_ratio=1.0)
#     pcd = pcd.select_by_index(ind)
#     o3d.io.write_point_cloud(os.path.join(args.out_dir, "01_cleaned.ply"), pcd)
    
#     # Load calibration and transform
#     print("Transforming point cloud...")
#     centroid, R, scale = load_calibration(args.calib)
#     points = np.asarray(pcd.points)
#     points_mm = ((points - centroid) @ R) * scale
#     pcd_mm = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_mm))
#     o3d.io.write_point_cloud(os.path.join(args.out_dir, "02_transformed.ply"), pcd_mm)
    
#     # Run semi-manual measurement
#     print("Starting semi-manual measurement...")
#     width, girth = semi_manual_measure(pcd_mm, args.slice_thickness_mm, args.out_dir)
#     if width is not None:
#         print(f"Final Measurements: Width = {width:.2f} mm, Girth = {girth:.2f} mm")
#     else:
#         print("Measurement failed. Check point cloud or try different selection.")

# if __name__ == "__main__":
#     main()

import open3d as o3d
import argparse
import sys
import os
import json
import numpy as np
from scipy.spatial import ConvexHull
try:
    import alphashape
    from shapely.geometry import Polygon
    HAVE_ALPHA = True
except ImportError:
    HAVE_ALPHA = False

def ensure_dir(d): os.makedirs(d, exist_ok=True)

def points_between_planes(points, normal, d_center, thickness):
    """
    Find points between two parallel planes with given normal and center plane distance d_center.
    Returns: Boolean mask for points in the slab.
    """
    half_thickness = thickness / 2.0
    normal = normal / np.linalg.norm(normal)  # Normalize
    distances = np.dot(points, normal) - d_center
    return np.abs(distances) <= half_thickness

def plane_equation_parameters_extraction(selected_points):
    """
    Compute plane equation ax + by + cz = d from three points.
    Returns: a, b, c (normal), d (constant).
    """
    p1 = np.array(selected_points[0])
    p2 = np.array(selected_points[1])
    p3 = np.array(selected_points[2])
    print(f"Selected points: p1={p1}, p2={p2}, p3={p3}")
    v1 = p2 - p1
    v2 = p3 - p1
    normal_vector = np.cross(v1, v2)
    if np.linalg.norm(normal_vector) < 1e-6:
        print("Error: Selected points are collinear, invalid plane.")
        return None, None, None, None
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    a_x, b_y, c_z = normal_vector
    d = np.dot(normal_vector, p1)
    return a_x, b_y, c_z, d

def create_plane_mesh(normal, d_value, bbox_extent, grid_size_factor=1.2):
    """
    Create a triangle mesh for a plane defined by normal · x = d_value.
    Returns: Open3D TriangleMesh.
    """
    normal = normal / np.linalg.norm(normal)
    grid_size = max(bbox_extent) * grid_size_factor
    # Generate a square grid in a plane perpendicular to normal
    # Find two vectors orthogonal to normal
    if abs(normal[0]) > 1e-6 or abs(normal[1]) > 1e-6:
        u = np.cross(normal, [0, 0, 1])  # Orthogonal to normal and Z-axis
    else:
        u = np.cross(normal, [1, 0, 0])
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)
    
    # Create grid points
    steps = np.linspace(-grid_size/2, grid_size/2, 20)  # 20x20 grid
    U, V = np.meshgrid(steps, steps)
    points = d_value * normal + U.flatten()[:, None] * u + V.flatten()[:, None] * v
    
    # Create triangle mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    triangles = []
    n = len(steps)
    for i in range(n-1):
        for j in range(n-1):
            idx = i * n + j
            triangles.append([idx, idx+1, idx+n])
            triangles.append([idx+1, idx+n, idx+n+1])
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    return mesh

def semi_manual_measure(pcd_mm, slice_thickness_mm=20.0, out_dir='.'):
    """
    Semi-manual measurement: Select points to define a plane, compute width and girth.
    """
    print("Opening visualizer: Select 3 points to define a measurement plane (e.g., ball area).")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window("Select Points for Plane (e.g., Ball Area)")
    vis.add_geometry(pcd_mm)
    vis.run()
    vis.destroy_window()
    
    points = np.asarray(pcd_mm.points)
    print(f"Point cloud size: {len(points)} points")
    print(f"Point cloud bounds: X=[{points[:,0].min():.2f}, {points[:,0].max():.2f}], "
          f"Y=[{points[:,1].min():.2f}, {points[:,1].max():.2f}], "
          f"Z=[{points[:,2].min():.2f}, {points[:,2].max():.2f}] mm")
    
    picked_indices = vis.get_picked_points()
    print(f"Picked {len(picked_indices)} points")
    
    if len(picked_indices) < 3:
        print("Need at least 3 points. Using default X-axis slice at midpoint.")
        normal = np.array([1.0, 0.0, 0.0])
        d_center = np.mean(points[:, 0])
        center_x = d_center
    else:
        selected_points = points[picked_indices]
        a_x, b_y, c_z, d_center = plane_equation_parameters_extraction(selected_points)
        if a_x is None:
            print("Falling back to X-axis slice due to invalid plane.")
            normal = np.array([1.0, 0.0, 0.0])
            d_center = np.mean(points[:, 0])
            center_x = d_center
        else:
            normal = np.array([a_x, b_y, c_z])
            center_x = d_center / normal[0] if abs(normal[0]) > 1e-6 else np.mean(points[:, 0])
            print(f"Plane equation: {a_x:.3f}x + {b_y:.3f}y + {c_z:.3f}z = {d_center:.3f}")
    
    slice_mask = points_between_planes(points, normal, d_center, slice_thickness_mm)
    print(f"Slice mask: {slice_mask[:10]}... (Total True: {np.sum(slice_mask)})")
    slice_pts = points[slice_mask]
    print(f"Slice contains {len(slice_pts)} points")
    
    if len(slice_pts) < 10:
        print("Too few points in slice. Try increasing slice_thickness_mm or checking calibration.")
        return None, None
    
    # Compute width (Y-range)
    width = slice_pts[:, 1].max() - slice_pts[:, 1].min()
    
    # Compute girth (YZ projection)
    P = slice_pts[:, [1, 2]]
    if HAVE_ALPHA and len(P) >= 4:
        alpha_param = 2.0
        ashape = alphashape.alphashape(P, alpha_param)
        print(f"Alphashape computed, empty: {ashape.is_empty}")
        if not ashape.is_empty:
            geom = ashape if ashape.geom_type != "MultiPolygon" else max(ashape.geoms, key=lambda g: g.area)
            girth = geom.length
        else:
            girth = _convex_perimeter(P)
    else:
        girth = _convex_perimeter(P)
    
    # Visualize slice and planes as meshes
    slice_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(slice_pts))
    slice_pcd.paint_uniform_color([1, 0, 1])  # Magenta
    pcd_mm.colors = o3d.utility.Vector3dVector(np.ones_like(points) * [0.7, 0.7, 0.7])
    
    bbox = pcd_mm.get_axis_aligned_bounding_box()
    bbox_extent = bbox.get_extent()
    half_thickness = slice_thickness_mm / 2.0
    d1 = d_center - half_thickness
    d2 = d_center + half_thickness
    print(f"Back plane d1: {d1:.2f}, Front plane d2: {d2:.2f} (Separation: {d2-d1:.2f} mm)")
    
    back_plane_mesh = create_plane_mesh(normal, d1, bbox_extent)
    front_plane_mesh = create_plane_mesh(normal, d2, bbox_extent)
    back_plane_mesh.paint_uniform_color([0, 1, 0])  # Green
    front_plane_mesh.paint_uniform_color([0, 0, 1])  # Blue
    
    print("Showing slice (magenta), back plane (green), front plane (blue)...")
    o3d.visualization.draw_geometries([pcd_mm, slice_pcd, back_plane_mesh, front_plane_mesh],
                                     window_name=f"Slice at d={d_center:.2f}")
    
    # Save measurements
    ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, "measurements.csv")
    with open(csv_path, "a") as f:
        import csv
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["Slice Center (mm)", "Width (mm)", "Girth/Perimeter (mm)"])
        writer.writerow([center_x, width, girth])
    
    print(f"Measurements: Width = {width:.2f} mm, Girth = {girth:.2f} mm")
    return width, girth

def _convex_perimeter(P):
    if len(P) < 3:
        return 0.0
    hull = ConvexHull(P)
    poly = Polygon(P[hull.vertices])
    return poly.length

def load_calibration(path):
    with open(path, "r") as f:
        data = json.load(f)
    R = np.array(data["R"], dtype=float)
    centroid = np.array(data["centroid"], dtype=float)
    scale = float(data["scale_mm_per_unit"])
    return centroid, R, scale

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Raw foot point cloud (.ply/.pcd/.xyz)")
    ap.add_argument("--calib", required=True, help="a4_calibration.txt (JSON)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--slice_thickness_mm", type=float, default=20.0)
    args = ap.parse_args()
    
    ensure_dir(args.out_dir)
    print("Loading point cloud...")
    pcd = o3d.io.read_point_cloud(args.input)
    if len(pcd.points) == 0:
        print("ERROR: Empty point cloud!", file=sys.stderr)
        sys.exit(1)
    
    print("Cleaning point cloud...")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=15, std_ratio=1.0)
    pcd = pcd.select_by_index(ind)
    o3d.io.write_point_cloud(os.path.join(args.out_dir, "01_cleaned.ply"), pcd)
    
    print("Transforming point cloud...")
    centroid, R, scale = load_calibration(args.calib)
    points = np.asarray(pcd.points)
    points_mm = ((points - centroid) @ R) * scale
    pcd_mm = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_mm))
    o3d.io.write_point_cloud(os.path.join(args.out_dir, "02_transformed.ply"), pcd_mm)
    
    print("Starting semi-manual measurement...")
    width, girth = semi_manual_measure(pcd_mm, args.slice_thickness_mm, args.out_dir)
    if width is not None:
        print(f"Final Measurements: Width = {width:.2f} mm, Girth = {girth:.2f} mm")
    else:
        print("Measurement failed. Check point cloud or try different selection.")

if __name__ == "__main__":
    main()