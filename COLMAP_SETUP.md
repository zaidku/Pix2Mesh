# COLMAP Setup Instructions

## What is COLMAP?

COLMAP is a state-of-the-art Structure-from-Motion (SfM) and Multi-View Stereo (MVS) pipeline that reconstructs 3D models from arbitrary photo collections.

## Benefits over Voxel Carving

- **Automatic camera pose estimation** - works with any viewpoints
- **Feature matching** - uses SIFT/ORB to find correspondences
- **Dense reconstruction** - creates highly detailed point clouds
- **Better accuracy** - photogrammetry-grade results
- **No special capture required** - works with casual photos

## Installation

### Windows

1. **Download COLMAP**
   - Go to https://github.com/colmap/colmap/releases
   - Download the latest Windows build (e.g., `COLMAP-3.9-windows-cuda.zip`)

2. **Extract and Add to PATH**
   ```powershell
   # Extract to a folder, e.g., C:\Program Files\COLMAP
   # Add to System PATH:
   $env:Path += ";C:\Program Files\COLMAP"
   # Or add permanently via System Properties > Environment Variables
   ```

3. **Verify Installation**
   ```powershell
   colmap -h
   ```

### Linux (Ubuntu/Debian)

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install \
    git \
    cmake \
    build-essential \
    libboost-all-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev

# Install COLMAP from package
sudo apt-get install colmap

# OR build from source
git clone https://github.com/colmap/colmap.git
cd colmap
mkdir build
cd build
cmake ..
make -j8
sudo make install
```

### macOS

```bash
# Using Homebrew
brew install colmap
```

## Using COLMAP in Pix2Mesh

1. Start the Flask server:
   ```powershell
   Set-Location "C:\Zaid\source\Pix2Mesh\Pix2Mesh-main"
   $env:NUMBA_DISABLE_JIT="1"
   C:\Zaid\source\Pix2Mesh\.venv\Scripts\python.exe app.py
   ```

2. Open http://127.0.0.1:5000

3. Upload your images (4-20 photos from different angles)

4. Select **COLMAP (Advanced)** reconstruction method

5. Click **Generate 3D Model**

## Performance Notes

- **Voxel Carving**: Fast (seconds), good for turntable captures
- **COLMAP**: Slower (minutes), but much better quality for arbitrary photos

COLMAP processing time depends on:
- Number of images
- Image resolution
- Feature count
- CPU/GPU performance

Typical processing time for 10 images (4K resolution):
- Feature extraction: 1-2 minutes
- Matching: 1-3 minutes
- Reconstruction: 2-5 minutes
- Dense stereo: 5-10 minutes
- **Total: 10-20 minutes**

## Troubleshooting

### "colmap: command not found"
- COLMAP is not installed or not in PATH
- Check installation and verify with `colmap -h`

### "Feature extraction failed"
- Images may be too similar or lack distinct features
- Try photos with more texture/detail

### "No reconstruction found"
- Not enough matching features between images
- Take more photos with better overlap (60-80%)
- Ensure good lighting and focus

### Out of Memory
- Reduce image resolution
- Use fewer images
- Increase system RAM or use smaller batch sizes

## Recommended Photo Capture

For best COLMAP results:
1. **Overlap**: 60-80% overlap between consecutive photos
2. **Coverage**: Capture from all angles around the object
3. **Lighting**: Consistent, diffused lighting
4. **Focus**: Sharp, in-focus images
5. **Quantity**: 20-50 images for simple objects, 100+ for complex scenes
6. **Movement**: Either move around stationary object OR rotate object with fixed camera
