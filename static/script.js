// State
let selectedFiles = [];
let uploadedCount = 0;
let generatedFilename = '';
let cameraStream = null;
let capturedPhotos = [];

// Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const imagePreview = document.getElementById('imagePreview');
const uploadBtn = document.getElementById('uploadBtn');
const clearBtn = document.getElementById('clearBtn');
const processBtn = document.getElementById('processBtn');
const downloadBtn = document.getElementById('downloadBtn');
const restartBtn = document.getElementById('restartBtn');

const uploadSection = document.getElementById('uploadSection');
const processingSection = document.getElementById('processingSection');
const progressSection = document.getElementById('progressSection');
const downloadSection = document.getElementById('downloadSection');
const errorMessage = document.getElementById('errorMessage');
const errorText = document.getElementById('errorText');
const progressText = document.getElementById('progressText');

// Camera elements
const openCameraBtn = document.getElementById('openCameraBtn');
const cameraModal = document.getElementById('cameraModal');
const closeCameraBtn = document.getElementById('closeCameraBtn');
const cameraVideo = document.getElementById('cameraVideo');
const cameraCanvas = document.getElementById('cameraCanvas');
const captureBtn = document.getElementById('captureBtn');
const finishCaptureBtn = document.getElementById('finishCaptureBtn');
const captureCount = document.getElementById('captureCount');
const cameraPreviewGrid = document.getElementById('cameraPreviewGrid');

// Drop zone handlers
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    handleFiles(e.dataTransfer.files);
});

fileInput.addEventListener('change', (e) => {
    handleFiles(e.target.files);
});

// File handling
function handleFiles(files) {
    const imageFiles = Array.from(files).filter(file => {
        const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/tiff', 'image/tif'];
        const validExtensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff'];
        const fileName = file.name.toLowerCase();
        const hasValidExtension = validExtensions.some(ext => fileName.endsWith(ext));
        return validTypes.includes(file.type) || hasValidExtension;
    });
    
    if (imageFiles.length === 0) {
        showError('Please select valid image files (PNG, JPG, JPEG, TIF, TIFF)');
        return;
    }
    
    const totalFiles = selectedFiles.length + imageFiles.length;
    if (totalFiles > 20) {
        showError(`Maximum 20 images allowed. You can add ${20 - selectedFiles.length} more images.`);
        return;
    }
    
    selectedFiles = [...selectedFiles, ...imageFiles];
    updatePreview();
    updateButtons();
    hideError();
}

function updatePreview() {
    imagePreview.innerHTML = '';
    
    selectedFiles.forEach((file, index) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const div = document.createElement('div');
            div.className = 'preview-item';
            div.innerHTML = `
                <img src="${e.target.result}" alt="Preview ${index + 1}">
                <button class="remove-btn" onclick="removeImage(${index})">×</button>
            `;
            imagePreview.appendChild(div);
        };
        reader.readAsDataURL(file);
    });
}

function removeImage(index) {
    selectedFiles.splice(index, 1);
    updatePreview();
    updateButtons();
}

function updateButtons() {
    const hasFiles = selectedFiles.length > 0;
    const hasEnoughFiles = selectedFiles.length >= 4;
    const tooManyFiles = selectedFiles.length > 20;
    
    uploadBtn.disabled = !hasEnoughFiles || tooManyFiles;
    clearBtn.disabled = !hasFiles;
    
    if (tooManyFiles) {
        showError(`Maximum 20 images allowed (currently ${selectedFiles.length})`);
    } else if (hasFiles && !hasEnoughFiles) {
        showError(`Please add at least 4 images (currently ${selectedFiles.length})`);
    } else {
        hideError();
    }
}

// Clear all
clearBtn.addEventListener('click', () => {
    selectedFiles = [];
    imagePreview.innerHTML = '';
    fileInput.value = '';
    updateButtons();
    hideError();
});

// Upload images
uploadBtn.addEventListener('click', async () => {
    if (selectedFiles.length < 4) {
        showError('Please add at least 4 images');
        return;
    }
    
    const formData = new FormData();
    selectedFiles.forEach(file => {
        formData.append('images', file);
    });
    
    uploadBtn.disabled = true;
    uploadBtn.textContent = 'Uploading...';
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Upload failed');
        }
        
        uploadedCount = data.count;
        showProcessingSection();
        
    } catch (error) {
        showError(error.message);
        uploadBtn.disabled = false;
        uploadBtn.textContent = 'Upload Images';
    }
});

// Process images
processBtn.addEventListener('click', async () => {
    const format = document.querySelector('input[name="format"]:checked').value;
    const method = document.querySelector('input[name="method"]:checked').value;
    
    processingSection.classList.add('hidden');
    progressSection.classList.remove('hidden');
    
    if (method === 'colmap') {
        progressText.textContent = 'COLMAP: Extracting features...';
    } else if (method === 'ct_stack') {
        progressText.textContent = 'CT Stack: Loading slices...';
    } else {
        progressText.textContent = 'Removing backgrounds...';
    }
    
    try {
        await new Promise(resolve => setTimeout(resolve, 500));
        
        if (method === 'colmap') {
            progressText.textContent = 'COLMAP: Matching features...';
            await new Promise(resolve => setTimeout(resolve, 500));
            progressText.textContent = 'COLMAP: Sparse reconstruction...';
            await new Promise(resolve => setTimeout(resolve, 500));
            progressText.textContent = 'COLMAP: Dense reconstruction...';
        } else if (method === 'ct_stack') {
            progressText.textContent = 'CT Stack: Stacking volume...';
            await new Promise(resolve => setTimeout(resolve, 500));
            progressText.textContent = 'CT Stack: Segmenting...';
            await new Promise(resolve => setTimeout(resolve, 500));
            progressText.textContent = 'CT Stack: Extracting surface...';
        } else {
            progressText.textContent = 'Generating 3D point cloud...';
            await new Promise(resolve => setTimeout(resolve, 500));
            progressText.textContent = 'Creating mesh...';
            await new Promise(resolve => setTimeout(resolve, 500));
            progressText.textContent = 'Optimizing mesh...';
        }
        
        const response = await fetch('/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ format, method })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Processing failed');
        }
        
        generatedFilename = data.filename;
        
        // Show preview if PLY format
        if (data.preview_available) {
            await load3DPreview(data.filename);
        }
        
        showDownloadSection();
        
    } catch (error) {
        showError(error.message);
        progressSection.classList.add('hidden');
        processingSection.classList.remove('hidden');
    }
});

// Download model
downloadBtn.addEventListener('click', () => {
    if (generatedFilename) {
        window.location.href = `/download/${generatedFilename}`;
    }
});

// Restart
restartBtn.addEventListener('click', () => {
    selectedFiles = [];
    uploadedCount = 0;
    generatedFilename = '';
    fileInput.value = '';
    imagePreview.innerHTML = '';
    
    downloadSection.classList.add('hidden');
    progressSection.classList.add('hidden');
    processingSection.classList.add('hidden');
    uploadSection.classList.remove('hidden');
    
    updateButtons();
    hideError();
    
    uploadBtn.textContent = 'Upload Images';
    uploadBtn.disabled = true;
});

// UI Helper functions
function showProcessingSection() {
    uploadSection.classList.add('hidden');
    processingSection.classList.remove('hidden');
}

function showDownloadSection() {
    progressSection.classList.add('hidden');
    downloadSection.classList.remove('hidden');
}

function showError(message) {
    errorText.textContent = message;
    errorMessage.classList.remove('hidden');
}

function hideError() {
    errorMessage.classList.add('hidden');
}

// Initial state
updateButtons();

// 3D Preview using Three.js
async function load3DPreview(filename) {
    const previewContainer = document.getElementById('modelPreview');
    if (!previewContainer) return;
    
    // Clear previous preview
    previewContainer.innerHTML = '';
    
    try {
        // Dynamically load Three.js
        if (typeof THREE === 'undefined') {
            await loadScript('https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js');
            await loadScript('https://cdn.jsdelivr.net/npm/three@0.160.0/examples/js/loaders/PLYLoader.js');
            await loadScript('https://cdn.jsdelivr.net/npm/three@0.160.0/examples/js/controls/OrbitControls.js');
        }
        
        // Setup scene
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0a0e17);
        
        const camera = new THREE.PerspectiveCamera(75, previewContainer.clientWidth / previewContainer.clientHeight, 0.1, 1000);
        camera.position.z = 2;
        
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(previewContainer.clientWidth, previewContainer.clientHeight);
        previewContainer.appendChild(renderer.domElement);
        
        // Add lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        scene.add(ambientLight);
        
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(1, 1, 1);
        scene.add(directionalLight);
        
        // Add orbit controls
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        
        // Load PLY model
        const loader = new THREE.PLYLoader();
        loader.load(`/preview/${filename}`, (geometry) => {
            geometry.computeVertexNormals();
            
            const material = new THREE.MeshStandardMaterial({
                vertexColors: true,
                flatShading: false
            });
            
            const mesh = new THREE.Mesh(geometry, material);
            
            // Center and scale the model
            geometry.computeBoundingBox();
            const center = new THREE.Vector3();
            geometry.boundingBox.getCenter(center);
            mesh.position.sub(center);
            
            const size = new THREE.Vector3();
            geometry.boundingBox.getSize(size);
            const maxDim = Math.max(size.x, size.y, size.z);
            mesh.scale.multiplyScalar(1.5 / maxDim);
            
            scene.add(mesh);
            
            // Add rotation animation
            function animate() {
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }
            animate();
        });
        
        // Handle window resize
        window.addEventListener('resize', () => {
            if (previewContainer.clientWidth > 0) {
                camera.aspect = previewContainer.clientWidth / previewContainer.clientHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(previewContainer.clientWidth, previewContainer.clientHeight);
            }
        });
        
    } catch (error) {
        console.error('Error loading 3D preview:', error);
        previewContainer.innerHTML = '<p style="color: #94a3b8; text-align: center; padding: 20px;">Preview unavailable for STL format</p>';
    }
}

// Camera functionality
openCameraBtn.addEventListener('click', async () => {
    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                facingMode: 'environment',
                width: { ideal: 1920 },
                height: { ideal: 1080 }
            } 
        });
        cameraVideo.srcObject = cameraStream;
        cameraModal.classList.remove('hidden');
        capturedPhotos = [];
        updateCameraPreview();
    } catch (error) {
        showError('Unable to access camera. Please check permissions.');
        console.error('Camera error:', error);
    }
});

closeCameraBtn.addEventListener('click', () => {
    stopCamera();
});

captureBtn.addEventListener('click', () => {
    const canvas = cameraCanvas;
    const video = cameraVideo;
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    
    canvas.toBlob((blob) => {
        const file = new File([blob], `capture_${Date.now()}.jpg`, { type: 'image/jpeg' });
        capturedPhotos.push(file);
        updateCameraPreview();
        
        if (capturedPhotos.length >= 4) {
            finishCaptureBtn.disabled = false;
        }
    }, 'image/jpeg', 0.95);
});

finishCaptureBtn.addEventListener('click', () => {
    if (capturedPhotos.length === 0) return;
    
    const totalFiles = selectedFiles.length + capturedPhotos.length;
    if (totalFiles > 20) {
        showError(`Maximum 20 images allowed. You can add ${20 - selectedFiles.length} more images.`);
        return;
    }
    
    selectedFiles = [...selectedFiles, ...capturedPhotos];
    updatePreview();
    updateButtons();
    stopCamera();
    hideError();
});

function updateCameraPreview() {
    captureCount.textContent = capturedPhotos.length;
    cameraPreviewGrid.innerHTML = '';
    
    capturedPhotos.forEach((file, index) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            const div = document.createElement('div');
            div.className = 'camera-preview-item';
            div.innerHTML = `
                <img src="${e.target.result}" alt="Captured ${index + 1}">
                <button class="remove-preview" onclick="removeCapturedPhoto(${index})">×</button>
            `;
            cameraPreviewGrid.appendChild(div);
        };
        reader.readAsDataURL(file);
    });
}

function removeCapturedPhoto(index) {
    capturedPhotos.splice(index, 1);
    updateCameraPreview();
    
    if (capturedPhotos.length < 4) {
        finishCaptureBtn.disabled = true;
    }
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    cameraModal.classList.add('hidden');
    cameraVideo.srcObject = null;
    capturedPhotos = [];
    updateCameraPreview();
}

function loadScript(src) {
    return new Promise((resolve, reject) => {
        const script = document.createElement('script');
        script.src = src;
        script.onload = resolve;
        script.onerror = reject;
        document.head.appendChild(script);
    });
}
