# Pix2Mesh

A Flask web application for generating 3D models from multi-view 2D images. The system supports three reconstruction methods: voxel-based carving for controlled captures, OpenCV Structure-from-Motion for photogrammetry, and CT/MRI stack processing for medical volumetric data.

**Status**: Work in progress. Active development with daily improvements and refinements.

## Overview

Pix2Mesh converts image data into 3D mesh models suitable for 3D printing, visualization, and analysis. The application adapts to different data types: photographs, CT scans, or MRI slices.

## Key Features

- **Modern Web Interface** - Responsive dark-themed UI with drag-and-drop functionality
- **Three Reconstruction Methods**:
  - **Voxel Carving** - Visual hull reconstruction for turntable captures
  - **Structure-from-Motion** - OpenCV-based photogrammetry with SIFT feature matching
  - **CT/MRI Stack** - Medical imaging reconstruction from volumetric slice data
- **AI Background Removal** - Automatic segmentation for photograph-based reconstruction
- **Feature Matching** - SIFT keypoint detection with RANSAC pose estimation
- **Marching Cubes** - Surface extraction from voxel grids and volumetric data
- **Poisson Reconstruction** - Surface meshing from sparse point clouds
- **Multiple Export Formats** - STL for manufacturing, PLY for textured models

## Reconstruction Methods

### Voxel Carving
- Best for: Turntable captures with evenly-spaced angles
- Processing time: Seconds
- Requires: 4-20 images from rotational views
- Quality: Good for simple objects with clear silhouettes

### Structure-from-Motion (OpenCV)
- Best for: Casual photos with overlap
- Processing time: Seconds to minutes
- Requires: Photos with 60-80% overlap of the same object
- Quality: Depends on image quality and overlap
- Note: Uses OpenCV SIFT implementation, no external COLMAP required

### CT/MRI Stack
- Best for: Medical scan TIFF slices
- Processing time: Seconds
- Requires: Stack of CT or MRI slices in TIFF format
- Quality: Direct volume processing with automatic segmentation
- Status: Experimental, threshold tuning in progress

## System Requirements

### Required Software
- Python 3.8-3.12 (3.11 recommended)
### Required Software
- Python 3.11 (recommended for compatibility)
- pip package manager

### Optional Components
- MinGW-w64 GCC compiler (for C++ mesh optimizer)
- CUDA-enabled GPU (for accelerated processing)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/zaidku/Pix2Mesh.git
cd Pix2Mesh
```

2. Create virtual environment (recommended):
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Launch the application:
```bash
# Windows
$env:NUMBA_DISABLE_JIT="1"
python app.py

# Linux/Mac
NUMBA_DISABLE_JIT=1 python app.py
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
   - For photos: Capture 6-12 images with 60-80% overlap
   - For CT/MRI: Export scan slices as TIFF files
   - Maintain consistent lighting for photographs

2. **Image Upload**
   - Drag images into the upload zone or use file selector
   - Preview thumbnails confirm successful upload

3. **Method Selection**
   - **Voxel Carving**: For turntable captures
   - **Structure-from-Motion**: For overlapping photos of the same object
   - **CT/MRI Stack**: For medical scan TIFF slices

4. **Format Selection**
   - STL: For 3D printing and CAD applications
   - PLY: For visualization with vertex colors

5. **Processing**
   - Click "Generate 3D Model" to initiate reconstruction
   - Status updates display progress through pipeline stages

6. **Model Export**
   - Download button appears upon completion
   - Compatible with Blender, MeshLab, Cura, PrusaSlicer

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
