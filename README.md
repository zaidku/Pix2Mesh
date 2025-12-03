# Pix2Mesh

A professional-grade Flask web application for generating 3D models from multi-view 2D images. The system employs visual hull reconstruction with Marching Cubes algorithm to create high-quality 3D meshes from photographs taken at multiple angles.

## Overview

Pix2Mesh converts a series of 2D photographs into watertight 3D mesh models suitable for 3D printing, computer graphics, and virtual reality applications. The system implements AI-powered background removal, space carving voxel generation, and isosurface extraction to produce geometrically accurate 3D reconstructions.

## Key Features

- **Modern Web Interface** - Responsive dark-themed UI with intuitive drag-and-drop functionality
- **Multi-View Input Processing** - Handles 4 to 10+ images from different viewing angles
- **AI Background Removal** - Automatic segmentation using state-of-the-art neural networks
- **Visual Hull Reconstruction** - Space carving algorithm for volumetric reconstruction
- **Marching Cubes Meshing** - Direct voxel-to-mesh conversion with smooth surface generation
- **C++ Optimization Module** - Optional high-performance mesh post-processing
- **Multiple Export Formats** - STL for manufacturing, PLY for textured visualization

## System Requirements

### Required Software
- Python 3.8 or higher
- pip package manager

### Optional Components
- MinGW-w64 GCC compiler (for C++ mesh optimizer)
- CUDA-enabled GPU (for accelerated background removal)

## Installation

### Standard Installation

1. Clone the repository:
```bash
git clone https://github.com/zaidku/Pix2Mesh.git
cd Pix2Mesh
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Launch the application:
```bash
python app.py
```

### Optional C++ Optimizer

For enhanced mesh processing performance, compile the C++ optimization module:

```bash
compile.bat
```

Note: Requires MinGW-w64 GCC compiler. The application functions fully without this optimization.

## Usage Guide

### Starting the Server

Execute the Flask application:
```bash
python app.py
```

Access the web interface at: `http://localhost:5000`

### 3D Reconstruction Workflow

1. **Image Acquisition**
   - Capture 4-10 photographs of the target object
   - Maintain consistent distance and lighting
   - Cover 360-degree viewing angles when possible

2. **Image Upload**
   - Drag images into the upload zone or use file selector
   - Preview thumbnails confirm successful upload
   - Minimum 4 images required, 6-8 recommended

3. **Format Selection**
   - STL: Standard tessellation format for 3D printing and CAD
   - PLY: Stanford polygon format with RGB vertex colors

4. **Processing**
   - Click "Generate 3D Model" to initiate reconstruction
   - Processing time: 30-90 seconds depending on resolution and image count
   - Status updates display progress through each pipeline stage

5. **Model Export**
   - Download button appears upon completion
   - Files saved to output directory with timestamp
   - Compatible with Blender, MeshLab, Cura, PrusaSlicer, and other 3D software

## Technical Architecture

### Directory Structure

```
Pix2Mesh/
├── app.py                    # Flask application core
├── mesh_optimizer.cpp        # C++ mesh processing module
├── compile.bat              # Windows compilation script
├── requirements.txt         # Python package dependencies
├── templates/
│   └── index.html          # Main application interface
├── static/
│   ├── style.css           # Application styling
│   └── script.js           # Client-side logic
├── uploads/                # Temporary image storage
└── output/                 # Generated mesh output
```

### Processing Pipeline

The reconstruction system employs a multi-stage pipeline:

1. **Image Preprocessing**
   - Background removal using U²-Net neural network architecture
   - Silhouette extraction and binary mask generation
   - Morphological dilation for connectivity preservation

2. **Volumetric Reconstruction**
   - 180×180×180 voxel grid initialization
   - Space carving with orthographic projection
   - Multi-view consistency check across camera angles

3. **Isosurface Extraction**
   - Marching Cubes algorithm at threshold 0.5
   - Vertex color sampling from voxel grid
   - Normal vector computation for lighting

4. **Mesh Optimization**
   - Duplicate vertex removal
   - Non-manifold edge correction
   - Laplacian smoothing (C++ module)

5. **Export**
   - STL binary format with triangle normals
   - PLY ASCII format with RGB vertex attributes

## Optimization Guidelines

### Image Capture Best Practices

**Angular Coverage**
- Minimum: 4 images at 90-degree intervals
- Recommended: 8 images at 45-degree intervals
- Optimal: 12+ images with overlapping fields of view

**Lighting Conditions**
- Use diffuse lighting to minimize shadows
- Maintain consistent illumination across all captures
- Avoid direct flash or harsh directional lighting
- Indoor studio lighting or overcast outdoor conditions preferred

**Camera Settings**
- Fixed focal length across all shots
- Consistent exposure settings
- Manual focus to prevent variation
- Image resolution: 1920×1080 or higher recommended

**Subject Positioning**
- Center object in frame with minimal background
- Maintain uniform distance (within 5% variation)
- Ensure complete object visibility in each view
- Rotate camera around stationary object, not vice versa

### Quality Optimization

**Voxel Resolution**
- Current default: 180³ voxels
- Increase for complex geometry: edit `VOXEL_RESOLUTION` in app.py
- Higher resolution increases processing time quadratically

**Background Removal**
- High-contrast subjects produce cleaner silhouettes
- Neutral backgrounds (gray, white, green screen) optimal
- Avoid shadows cast on background surfaces

**Mesh Quality**
- More input images reduce reconstruction artifacts
- Silhouette dilation helps preserve thin features
- Marching Cubes level threshold adjustable in source code

## Output Formats

### STL (STereoLithography)

**Specifications:**
- Binary format encoding
- Triangle mesh representation
- Surface normal vectors included
- Industry standard for additive manufacturing

**Use Cases:**
- 3D printing (FDM, SLA, SLS)
- CNC machining toolpath generation
- CAD software import
- Structural analysis and simulation

**Compatible Software:**
- Ultimaker Cura, PrusaSlicer, Simplify3D
- Autodesk Fusion 360, SolidWorks
- Blender, Meshmixer, MeshLab

### PLY (Polygon File Format)

**Specifications:**
- ASCII or binary encoding
- Vertex position and color data
- Face connectivity information
- Extensible property support

**Use Cases:**
- Textured 3D visualization
- Computer graphics rendering
- Virtual reality environments
- Research and academic applications

**Compatible Software:**
- MeshLab, CloudCompare, Blender
- MATLAB, Python (Open3D, PyVista)
- Unity, Unreal Engine

## Troubleshooting

### Performance Issues

**Symptom:** Processing exceeds 2 minutes
- Reduce input image resolution to 1920×1080
- Decrease voxel resolution in configuration
- Limit image count to 6-8 photos
- Compile C++ optimizer for 2-3x speedup

**Symptom:** High memory consumption
- Lower `VOXEL_RESOLUTION` constant
- Process images in smaller batches
- Close unnecessary applications

### Quality Issues

**Symptom:** Disconnected mesh components
- Increase silhouette dilation iterations
- Add more images from intermediate angles
- Verify all images properly segmented

**Symptom:** Hollow or incomplete geometry
- Ensure 360-degree angular coverage
- Check for consistent object visibility
- Increase voxel resolution for complex shapes

**Symptom:** Noisy or rough surfaces
- Compile and enable C++ Laplacian smoothing
- Improve lighting consistency across images
- Use higher resolution source photographs

### Background Removal Issues

**Symptom:** Object partially removed
- Increase subject-background contrast
- Use solid-color background when possible
- Manually edit problematic images

**Symptom:** Background artifacts remain
- Simplify background composition
- Ensure adequate lighting on subject
- Position object away from background

### Application Errors

**Symptom:** Module import failures
- Verify all packages installed: `pip list`
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`
- Check Python version compatibility (3.8+)

**Symptom:** CUDA/GPU warnings
- Normal behavior if CUDA unavailable
- Processing continues on CPU
- Install CUDA toolkit for GPU acceleration (optional)

## Technology Stack

### Core Dependencies

**Backend Framework**
- Flask 3.0.0 - WSGI web application framework
- Werkzeug - HTTP utilities and routing

**Computer Vision & 3D Processing**
- OpenCV 4.8.1 - Image processing and manipulation
- Open3D 0.18.0 - 3D data structures and algorithms
- NumPy 1.24.3 - Numerical computation arrays
- scikit-image 0.21.0 - Marching Cubes implementation

**AI & Machine Learning**
- rembg 2.0.50 - U²-Net background removal
- onnxruntime - Neural network inference engine

**Frontend Technologies**
- HTML5 - Document structure
- CSS3 - Dark theme styling
- Vanilla JavaScript - Client-side interactions

### Algorithm Implementation

**Visual Hull Reconstruction**
- Space carving with orthographic projection
- Multi-view silhouette consistency
- Voxel grid occupancy computation

**Marching Cubes**
- Lorensen-Cline isosurface extraction
- Adaptive threshold for surface definition
- Vertex interpolation along voxel edges

**Mesh Processing**
- Duplicate vertex welding
- Non-manifold geometry repair
- Laplacian smoothing with boundary preservation

## Performance Characteristics

**Processing Time** (Intel i5/i7 CPU, 8GB RAM)
- 4 images (1920×1080): 30-45 seconds
- 8 images (1920×1080): 60-90 seconds
- With C++ optimizer: 40-60% reduction

**Memory Requirements**
- Base application: ~500 MB
- Per image processing: ~100-200 MB
- Voxel grid (180³): ~50 MB

**Scalability**
- Linear scaling with image count
- Cubic scaling with voxel resolution
- GPU acceleration available for background removal

## Future Development

Potential enhancements for advanced reconstruction capabilities:

**Advanced Reconstruction Methods**
- Structure from Motion (SfM) with COLMAP integration
- Multi-View Stereo (MVS) dense reconstruction
- Neural Radiance Fields (NeRF) for view synthesis

**AI-Based Approaches**
- PIFuHD for high-resolution implicit function reconstruction
- ICON for clothed human digitization
- Deep learning-based mesh refinement

**Feature Additions**
- Real-time preview of reconstruction progress
- Automatic camera calibration
- Texture mapping and UV unwrapping
- Batch processing mode for multiple objects

## Contributing

Contributions are welcome for bug fixes, performance improvements, and feature additions.

**Development Guidelines**
- Follow PEP 8 style for Python code
- Document all functions with docstrings
- Include test cases for new features
- Update README with significant changes

**Submission Process**
1. Fork the repository
2. Create feature branch
3. Commit changes with clear messages
4. Submit pull request with description

## License

MIT License - Free for personal, educational, and commercial use.

## Citation

If you use this software in academic research, please cite:

```
@software{pix2mesh2024,
  title={Pix2Mesh: Multi-View 3D Reconstruction System},
  author={Kuba, Zaid},
  year={2024},
  url={https://github.com/zaidku/Pix2Mesh}
}
```

## Support

For issues, questions, or feature requests, please open an issue on GitHub.

## Acknowledgments

This project utilizes algorithms and techniques from computer vision research:
- Visual hull reconstruction concept from Laurentini (1994)
- Marching Cubes algorithm from Lorensen and Cline (1987)
- U²-Net architecture for background removal from Qin et al. (2020)
