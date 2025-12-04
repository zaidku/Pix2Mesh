

from flask import Flask, render_template, request, jsonify, send_file
import os
import cv2
import numpy as np
from rembg import remove
from PIL import Image
import open3d as o3d
from pathlib import Path
import shutil
import subprocess
import json
from skimage import measure
from skimage import measure

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max file size

# Create necessary directories
for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_images():
    """Handle image uploads and process them"""
    try:
        # Clear previous uploads
        for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]:
            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.makedirs(folder)
        
        files = request.files.getlist('images')
        
        if len(files) < 4:
            return jsonify({'error': 'Please upload at least 4 images'}), 400
        
        if len(files) > 20:
            return jsonify({'error': 'Maximum 20 images allowed'}), 400
        
        image_paths = []
        for i, file in enumerate(files):
            if file and allowed_file(file.filename):
                filename = f'image_{i}.png'
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                image_paths.append(filepath)
        
        return jsonify({
            'success': True,
            'message': f'{len(image_paths)} images uploaded successfully',
            'count': len(image_paths)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process', methods=['POST'])
def process_images():
    """Remove backgrounds and generate 3D mesh"""
    try:
        data = request.get_json()
        output_format = data.get('format', 'stl')  # 'stl' or 'ply'
        method = data.get('method', 'voxel')  # 'voxel' or 'colmap'
        
        print(f"Starting processing with format: {output_format}, method: {method}")
        
        if method == 'colmap':
            # COLMAP-based reconstruction (no background removal needed)
            print("Using COLMAP Structure-from-Motion pipeline...")
            mesh = process_with_colmap(output_format)
        elif method == 'ct_stack':
            # CT/Medical scan stack reconstruction
            print("Using CT/Medical scan stack reconstruction...")
            mesh = process_ct_stack(output_format)
        else:
            # Original voxel-based method
            # Step 1: Remove backgrounds
            print("Step 1: Removing backgrounds...")
            processed_images = remove_backgrounds()
            
            if not processed_images:
                print("ERROR: No processed images")
                return jsonify({'error': 'Failed to process images'}), 500
            
            print(f"Processed {len(processed_images)} images")
            
            # Step 2: Generate point cloud from images
            print("Step 2: Generating voxel grid...")
            voxel_grid = generate_voxel_grid(processed_images)
            
            # Step 3: Create mesh using Marching Cubes
            print("Step 3: Creating mesh with Marching Cubes...")
            mesh = create_mesh_from_voxels(voxel_grid)
            print(f"Created mesh with {len(mesh.vertices)} vertices")
            
            # Step 4: Optimize mesh using C++ (optional)
            print("Step 4: Optimizing mesh...")
            mesh = optimize_mesh(mesh)
        
        # Step 5: Export mesh
        print("Step 5: Exporting mesh...")
        output_file = export_mesh(mesh, output_format)
        print(f"Exported to: {output_file}")
        
        if not os.path.exists(output_file):
            print(f"ERROR: File not created at {output_file}")
            return jsonify({'error': 'Failed to create output file'}), 500
        
        return jsonify({
            'success': True,
            'message': '3D model generated successfully',
            'filename': os.path.basename(output_file),
            'preview_available': output_format == 'ply'
        })
    
    except Exception as e:
        print(f"ERROR in process_images: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Download the generated 3D model"""
    filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True, download_name=filename)
    return jsonify({'error': 'File not found'}), 404

@app.route('/preview/<filename>')
def preview_file(filename):
    """Serve model file for 3D preview"""
    filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    if os.path.exists(filepath):
        return send_file(filepath, mimetype='application/octet-stream')
    return jsonify({'error': 'File not found'}), 404

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def remove_backgrounds():
    """Remove background from all uploaded images"""
    processed_images = []
    upload_folder = app.config['UPLOAD_FOLDER']
    
    for filename in sorted(os.listdir(upload_folder)):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            input_path = os.path.join(upload_folder, filename)
            output_path = os.path.join(upload_folder, f'nobg_{filename}')
            
            # Remove background
            with open(input_path, 'rb') as input_file:
                input_data = input_file.read()
                output_data = remove(input_data)
            
            with open(output_path, 'wb') as output_file:
                output_file.write(output_data)
            
            processed_images.append(output_path)
    
    return processed_images

def generate_voxel_grid(image_paths):
    """Generate 3D voxel grid from multiple images using visual hull/space carving"""
    
    # Load all images and extract silhouettes
    images = []
    silhouettes = []
    
    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
        if img.shape[2] == 4:  # Has alpha channel
            alpha = img[:, :, 3]
            img_rgb = img[:, :, :3]
        else:
            alpha = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
            img_rgb = img
        
        # Create binary silhouette
        silhouette = (alpha > 128).astype(np.uint8)
        images.append(img_rgb)
        silhouettes.append(silhouette)
    
    # Get dimensions
    num_images = len(image_paths)
    h, w = silhouettes[0].shape
    
    # Dilate silhouettes slightly to avoid over-carving thin parts
    kernel = np.ones((3, 3), np.uint8)
    silhouettes = [cv2.dilate(sil, kernel, iterations=2) for sil in silhouettes]
    
    # Define voxel grid parameters
    voxel_resolution = 256  # Higher resolution for smoother surfaces
    grid_size = max(h, w)
    
    # Calculate angles for each view (assuming orthogonal views: 0°, 90°, 180°, 270°)
    angles = [idx * (360.0 / num_images) for idx in range(num_images)]
    
    print(f"Creating voxel grid: {voxel_resolution}x{voxel_resolution}x{voxel_resolution}")
    
    # Initialize 3D voxel grid (all voxels start as occupied)
    voxels = np.ones((voxel_resolution, voxel_resolution, voxel_resolution), dtype=bool)
    color_grid = np.zeros((voxel_resolution, voxel_resolution, voxel_resolution, 3), dtype=np.float32)
    color_count = np.zeros((voxel_resolution, voxel_resolution, voxel_resolution), dtype=np.int32)
    
    # Space carving: remove voxels that don't project onto silhouettes
    for view_idx, (silhouette, img_rgb, angle) in enumerate(zip(silhouettes, images, angles)):
        angle_rad = np.radians(angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Check each voxel
        for i in range(voxel_resolution):
            for j in range(voxel_resolution):
                for k in range(voxel_resolution):
                    if not voxels[i, j, k]:
                        continue
                    
                    # Convert voxel coordinates to world space (-1 to 1)
                    x = (i / voxel_resolution - 0.5) * 2
                    y = (j / voxel_resolution - 0.5) * 2
                    z = (k / voxel_resolution - 0.5) * 2
                    
                    # Rotate point based on view angle (around Y-axis)
                    x_rot = x * cos_a + z * sin_a
                    z_rot = -x * sin_a + z * cos_a
                    
                    # Project to 2D image coordinates
                    px = int((x_rot + 1) * 0.5 * w)
                    py = int((1 - (y + 1) * 0.5) * h)
                    
                    # Check if projection is within image bounds
                    if 0 <= px < w and 0 <= py < h:
                        # If silhouette is empty at this point, carve away voxel
                        if silhouette[py, px] == 0:
                            voxels[i, j, k] = False
                        else:
                            # Accumulate color from this view
                            color_grid[i, j, k] += img_rgb[py, px][::-1]  # BGR to RGB
                            color_count[i, j, k] += 1
                    else:
                        # Outside image bounds, carve away
                        voxels[i, j, k] = False
    
    print(f"Voxels remaining after carving: {np.sum(voxels)}")
    
    # Average colors
    mask = color_count > 0
    color_grid[mask] = color_grid[mask] / color_count[mask, np.newaxis]
    
    return {
        'voxels': voxels.astype(np.float32),
        'colors': color_grid / 255.0,
        'resolution': voxel_resolution
    }


def create_mesh_from_voxels(voxel_data):
    """Create mesh from voxel grid using Marching Cubes algorithm"""
    voxels = voxel_data['voxels']
    colors = voxel_data['colors']
    resolution = voxel_data['resolution']
    
    # Apply Marching Cubes algorithm
    print("Running Marching Cubes algorithm...")
    vertices, faces, normals, _ = measure.marching_cubes(voxels, level=0.3, spacing=(2.0/resolution, 2.0/resolution, 2.0/resolution))
    
    # Center the mesh
    vertices -= np.array([1.0, 1.0, 1.0])
    vertices *= 100  # Scale to reasonable size
    
    # Sample colors from voxel grid for vertices
    vertex_colors = np.zeros((len(vertices), 3))
    for i, (x, y, z) in enumerate(vertices):
        # Convert back to voxel coordinates
        vx = int((x / 100 + 1) * 0.5 * resolution)
        vy = int((y / 100 + 1) * 0.5 * resolution)
        vz = int((z / 100 + 1) * 0.5 * resolution)
        
        # Clamp to valid range
        vx = np.clip(vx, 0, resolution - 1)
        vy = np.clip(vy, 0, resolution - 1)
        vz = np.clip(vz, 0, resolution - 1)
        
        vertex_colors[i] = colors[vx, vy, vz]
    
    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    
    # Compute normals for better visualization
    mesh.compute_vertex_normals()
    
    # Clean the mesh
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    
    # Apply Laplacian smoothing to reduce blocky appearance
    mesh = mesh.filter_smooth_laplacian(number_of_iterations=5, lambda_filter=0.5)
    
    # Recompute normals after smoothing
    mesh.compute_vertex_normals()
    
    print(f"Marching Cubes generated {len(vertices)} vertices and {len(faces)} faces")
    
    return mesh


def old_pointcloud_method():
    """Old method kept for reference"""
    # Convert voxels to point cloud
    points = []
    colors = []
    
    for i in range(voxel_resolution):
        for j in range(voxel_resolution):
            for k in range(voxel_resolution):
                if voxels[i, j, k]:
                    # Convert to world coordinates
                    x = (i / voxel_resolution - 0.5) * 200  # Scale to reasonable size
                    y = (j / voxel_resolution - 0.5) * 200
                    z = (k / voxel_resolution - 0.5) * 200
                    
                    points.append([x, y, z])
                    
                    # Average color from all views
                    if color_count[i, j, k] > 0:
                        avg_color = color_grid[i, j, k] / color_count[i, j, k] / 255.0
                        colors.append(avg_color)
                    else:
                        colors.append([0.5, 0.5, 0.5])
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    
    # Optional: remove outliers
    if len(points) > 100:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    print(f"Final point cloud has {len(pcd.points)} points")
    
    return pcd

def create_mesh_from_pointcloud(pcd):
    """Create mesh from point cloud using Poisson reconstruction"""
    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=15, max_nn=40)
    )
    
    # Orient normals
    pcd.orient_normals_consistent_tangent_plane(40)
    
    # Poisson surface reconstruction with higher depth for more detail
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=10, width=0, scale=1.2, linear_fit=False
    )
    
    # Remove low-density vertices (less aggressive to keep connections)
    vertices_to_remove = densities < np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    # Clean the mesh
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    
    return mesh

def optimize_mesh(mesh):
    """Optimize mesh, optionally using C++ module"""
    try:
        # Try to use C++ optimization if available
        temp_mesh_path = os.path.join(app.config['OUTPUT_FOLDER'], 'temp_mesh.ply')
        optimized_mesh_path = os.path.join(app.config['OUTPUT_FOLDER'], 'temp_optimized.ply')
        
        o3d.io.write_triangle_mesh(temp_mesh_path, mesh)
        
        # Check if C++ optimizer exists
        cpp_optimizer = './mesh_optimizer.exe' if os.name == 'nt' else './mesh_optimizer'
        
        if os.path.exists(cpp_optimizer):
            result = subprocess.run(
                [cpp_optimizer, temp_mesh_path, optimized_mesh_path],
                capture_output=True,
                timeout=30
            )
            
            if result.returncode == 0 and os.path.exists(optimized_mesh_path):
                mesh = o3d.io.read_triangle_mesh(optimized_mesh_path)
                os.remove(temp_mesh_path)
                os.remove(optimized_mesh_path)
        
    except Exception as e:
        print(f"C++ optimization failed or not available: {e}")
    
    # Python-based simplification
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=50000)
    mesh.compute_vertex_normals()
    
    return mesh

def process_with_colmap(output_format='ply'):
    """Process images using OpenCV-based Structure-from-Motion (COLMAP alternative)"""
    print("Using OpenCV-based SfM reconstruction...")
    
    upload_folder = app.config['UPLOAD_FOLDER']
    output_folder = app.config['OUTPUT_FOLDER']
    
    # Load images
    image_files = sorted([f for f in os.listdir(upload_folder) 
                         if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
    
    if len(image_files) < 2:
        raise Exception("Need at least 2 images for SfM reconstruction")
    
    images = []
    for fname in image_files:
        img = cv2.imread(os.path.join(upload_folder, fname))
        if img is not None:
            images.append(img)
    
    print(f"Loaded {len(images)} images")
    
    # Step 1: Feature detection and matching
    print("Step 1: Detecting and matching features...")
    points_3d, colors_3d = reconstruct_with_opencv_sfm(images)
    
    if len(points_3d) < 100:
        raise Exception("Not enough 3D points reconstructed. Try photos with more overlap.")
    
    print(f"Reconstructed {len(points_3d)} 3D points")
    
    # Step 2: Create point cloud
    print("Step 2: Creating point cloud...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors_3d)
    
    # Clean point cloud
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(30)
    
    # Step 3: Poisson surface reconstruction
    print("Step 3: Poisson surface reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9, width=0, scale=1.1, linear_fit=False
    )
    
    # Remove low-density vertices
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    # Clean mesh
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    
    # Simplify if too large
    if len(mesh.triangles) > 100000:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=100000)
    
    mesh.compute_vertex_normals()
    
    print(f"Final mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    
    return mesh

def reconstruct_with_opencv_sfm(images):
    """Perform Structure-from-Motion using OpenCV SIFT + RANSAC"""
    
    # Initialize SIFT detector with more features
    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.03, edgeThreshold=15)
    
    # Detect features in all images
    print("  - Detecting SIFT features...")
    keypoints_list = []
    descriptors_list = []
    
    for i, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply histogram equalization for better feature detection
        gray = cv2.equalizeHist(gray)
        kp, desc = sift.detectAndCompute(gray, None)
        keypoints_list.append(kp)
        descriptors_list.append(desc)
        print(f"    Image {i+1}: {len(kp)} keypoints")
    
    # Match features between ALL image pairs (not just consecutive)
    print("  - Matching features...")
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    
    all_points_3d = []
    all_colors_3d = []
    
    # Try all possible image pairs for better reconstruction
    num_images = len(images)
    pairs_processed = 0
    
    for i in range(num_images):
        for j in range(i + 1, min(i + 3, num_images)):  # Match with next 2 images
            img1 = images[i]
            img2 = images[j]
            kp1 = keypoints_list[i]
            kp2 = keypoints_list[j]
            desc1 = descriptors_list[i]
            desc2 = descriptors_list[j]
            
            if desc1 is None or desc2 is None:
                continue
            
            # Match descriptors
            matches = bf.knnMatch(desc1, desc2, k=2)
            
            # Apply Lowe's ratio test with more lenient threshold
            good_matches = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < 0.75 * n.distance:  # More lenient
                        good_matches.append(m)
            
            print(f"    Pair {i+1}-{j+1}: {len(good_matches)} good matches")
            
            if len(good_matches) < 8:
                print(f"    Warning: Not enough matches for pair {i+1}-{j+1}")
                continue
            
            # Get matched keypoint coordinates
            pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
            
            # Estimate Essential matrix using RANSAC with more lenient threshold
            h, w = img1.shape[:2]
            focal = max(w, h) * 1.2  # Approximate focal length
            pp = (w / 2, h / 2)  # Principal point
            
            K = np.array([[focal, 0, pp[0]],
                          [0, focal, pp[1]],
                          [0, 0, 1]], dtype=np.float32)
            
            E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=3.0)
            
            if E is None:
                print(f"    Warning: Could not compute Essential matrix for pair {i+1}-{j+1}")
                continue
            
            # Recover pose (R, t)
            _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
            
            # Triangulate points
            pts1_inliers = pts1[mask_pose.ravel() == 1]
            pts2_inliers = pts2[mask_pose.ravel() == 1]
            
            if len(pts1_inliers) < 4:
                print(f"    Warning: Not enough inliers for pair {i+1}-{j+1}")
                continue
            
            # Projection matrices
            P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
            P2 = K @ np.hstack([R, t])
            
            # Triangulate
            points_4d = cv2.triangulatePoints(P1, P2, pts1_inliers.T, pts2_inliers.T)
            points_3d = (points_4d[:3] / points_4d[3]).T
            
            # Filter points (remove points too far from camera or behind camera)
            depth_mask = points_3d[:, 2] > 0  # Points in front of camera
            valid_mask = depth_mask & (np.abs(points_3d[:, 2]) < 200)  # Not too far
            points_3d = points_3d[valid_mask]
            pts1_inliers = pts1_inliers[valid_mask]
            
            if len(points_3d) == 0:
                continue
            
            # Get colors from first image
            colors = []
            for pt in pts1_inliers:
                x, y = int(pt[0]), int(pt[1])
                if 0 <= x < w and 0 <= y < h:
                    color = img1[y, x][::-1] / 255.0  # BGR to RGB, normalize
                    colors.append(color)
                else:
                    colors.append([0.5, 0.5, 0.5])
            
            colors = np.array(colors)
            
            all_points_3d.append(points_3d)
            all_colors_3d.append(colors)
            pairs_processed += 1
            
            print(f"    -> Triangulated {len(points_3d)} 3D points")
            print(f"    -> Triangulated {len(points_3d)} 3D points")
    
    # Combine all 3D points
    if len(all_points_3d) == 0:
        raise Exception(f"Failed to triangulate any points from {num_images} images. Try: 1) Photos with more texture/detail, 2) Photos with better overlap (60-80%), 3) Photos from different angles.")
    
    print(f"  - Successfully processed {pairs_processed} image pairs")
    
    points_3d = np.vstack(all_points_3d)
    colors_3d = np.vstack(all_colors_3d)
    
    # Center and scale the point cloud
    centroid = np.mean(points_3d, axis=0)
    points_3d -= centroid
    
    max_dist = np.max(np.linalg.norm(points_3d, axis=1))
    if max_dist > 0:
        points_3d /= max_dist
        points_3d *= 100  # Scale to reasonable size
    
    return points_3d, colors_3d

def process_ct_stack(output_format='ply'):
    """Process CT/MRI scan stack (TIFF slices) into 3D mesh"""
    print("Processing CT/Medical scan stack...")
    
    upload_folder = app.config['UPLOAD_FOLDER']
    
    # Load all TIFF slices
    slice_files = sorted([f for f in os.listdir(upload_folder) 
                         if f.endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))])
    
    if len(slice_files) < 2:
        raise Exception("Need at least 2 slices for CT stack reconstruction")
    
    print(f"Loading {len(slice_files)} slices...")
    slices = []
    for fname in slice_files:
        img = cv2.imread(os.path.join(upload_folder, fname), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            slices.append(img)
    
    # Stack into 3D volume
    volume = np.stack(slices, axis=0)
    print(f"Volume shape: {volume.shape}")
    
    # Normalize to 0-1 range
    volume = volume.astype(np.float32) / 255.0
    
    # Optional: Try inverting if the scan has inverted intensities
    # (some CTs have air as bright, object as dark)
    mean_val = np.mean(volume)
    if mean_val > 0.5:
        print("Inverting volume (detected bright background)")
        volume = 1.0 - volume
    
    # Apply Otsu's thresholding for automatic segmentation
    from skimage import filters
    threshold = filters.threshold_otsu(volume)
    print(f"Otsu threshold: {threshold:.3f}")
    
    # Try multiple thresholds to find the best one
    # Sometimes Otsu picks up noise/background instead of the object
    thresholds_to_try = [
        threshold,
        threshold * 1.2,  # Slightly higher
        threshold * 0.8,  # Slightly lower
        np.percentile(volume, 60),  # 60th percentile
        np.percentile(volume, 70)   # 70th percentile
    ]
    
    print(f"Trying thresholds: {[f'{t:.3f}' for t in thresholds_to_try]}")
    
    # Use the threshold that gives a reasonable amount of voxels
    # (not too few, not too many - typically 5-30% of volume)
    best_threshold = threshold
    best_score = float('inf')
    
    for t in thresholds_to_try:
        binary_test = volume > t
        voxel_ratio = np.sum(binary_test) / binary_test.size
        # Prefer 10-20% of voxels as object
        score = abs(voxel_ratio - 0.15)
        print(f"  Threshold {t:.3f}: {voxel_ratio*100:.1f}% voxels, score: {score:.3f}")
        if score < best_score and 0.05 < voxel_ratio < 0.50:
            best_score = score
            best_threshold = t
    
    print(f"Using threshold: {best_threshold:.3f}")
    
    # Create binary mask
    binary_volume = volume > best_threshold
    
    # Morphological operations to clean up
    from scipy import ndimage
    print("Cleaning up binary volume...")
    
    # Fill holes in each slice
    for i in range(binary_volume.shape[0]):
        binary_volume[i] = ndimage.binary_fill_holes(binary_volume[i])
    
    # Remove small objects (noise)
    binary_volume = ndimage.binary_opening(binary_volume, iterations=2)
    
    # Close gaps
    binary_volume = ndimage.binary_closing(binary_volume, iterations=2)
    
    # Fill 3D holes
    binary_volume = ndimage.binary_fill_holes(binary_volume)
    
    print(f"Binary volume: {np.sum(binary_volume)} voxels")
    
    # Apply Marching Cubes
    print("Running Marching Cubes on CT volume...")
    verts, faces, normals, values = measure.marching_cubes(
        binary_volume, 
        level=0.5,
        spacing=(1.0, 1.0, 1.0)  # Adjust based on your CT spacing
    )
    
    print(f"Generated {len(verts)} vertices and {len(faces)} faces")
    
    # Create Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    
    # Center and scale
    mesh.translate(-mesh.get_center())
    scale = 100.0 / np.max(mesh.get_max_bound() - mesh.get_min_bound())
    mesh.scale(scale, center=mesh.get_center())
    
    # Smooth the mesh
    mesh = mesh.filter_smooth_laplacian(number_of_iterations=5)
    mesh.compute_vertex_normals()
    
    # Simplify if needed
    if len(faces) > 100000:
        print(f"Simplifying mesh from {len(faces)} to 100000 triangles...")
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=100000)
    
    mesh.compute_vertex_normals()
    
    print(f"Final mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    
    return mesh

def export_mesh(mesh, format='stl'):
    """Export mesh to file"""
    output_folder = app.config['OUTPUT_FOLDER']
    
    if format == 'ply':
        output_file = os.path.join(output_folder, 'model.ply')
        o3d.io.write_triangle_mesh(output_file, mesh, write_vertex_colors=True)
    else:  # stl
        output_file = os.path.join(output_folder, 'model.stl')
        o3d.io.write_triangle_mesh(output_file, mesh)
    
    return output_file

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
