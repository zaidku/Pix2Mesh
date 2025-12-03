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
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

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
        
        print(f"Starting processing with format: {output_format}")
        
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
            'filename': os.path.basename(output_file)
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

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def remove_backgrounds():
    """Remove background from all uploaded images"""
    processed_images = []
    upload_folder = app.config['UPLOAD_FOLDER']
    
    for filename in sorted(os.listdir(upload_folder)):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
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
    voxel_resolution = 180  # Increased for better detail
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
    vertices, faces, normals, _ = measure.marching_cubes(voxels, level=0.5, spacing=(2.0/resolution, 2.0/resolution, 2.0/resolution))
    
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
