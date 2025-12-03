// State
let selectedFiles = [];
let uploadedCount = 0;
let generatedFilename = '';

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
    const imageFiles = Array.from(files).filter(file => 
        file.type.startsWith('image/')
    );
    
    if (imageFiles.length === 0) {
        showError('Please select valid image files');
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
    
    uploadBtn.disabled = !hasEnoughFiles;
    clearBtn.disabled = !hasFiles;
    
    if (hasFiles && !hasEnoughFiles) {
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
    
    processingSection.classList.add('hidden');
    progressSection.classList.remove('hidden');
    progressText.textContent = 'Removing backgrounds...';
    
    try {
        await new Promise(resolve => setTimeout(resolve, 500));
        progressText.textContent = 'Generating 3D point cloud...';
        
        await new Promise(resolve => setTimeout(resolve, 500));
        progressText.textContent = 'Creating mesh...';
        
        await new Promise(resolve => setTimeout(resolve, 500));
        progressText.textContent = 'Optimizing mesh...';
        
        const response = await fetch('/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ format })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Processing failed');
        }
        
        generatedFilename = data.filename;
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
