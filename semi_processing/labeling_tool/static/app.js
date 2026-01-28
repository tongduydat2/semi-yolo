/**
 * YOLO Labeling Tool - Frontend Application
 * 
 * Features:
 * - Load classes from server (from YAML config)
 * - Right sidebar for label selection
 * - Label popup after drawing (when not in single label mode)
 * - Crosshair lines in draw mode
 * - Zoom with scroll wheel
 * - Pan when zoomed
 * 
 * Keyboard Shortcuts:
 * - Q: Toggle draw mode
 * - E: Save labels
 * - A/↑: Previous image
 * - D/↓: Next image
 * - R/Delete: Delete selected box
 * - F: Edit box class
 * - K: Toggle single label mode
 * - L: Toggle auto-save
 * - Z: Confirm image
 * - X: Unconfirm image
 * - 1-9: Select class
 */

// ============================================================================
// State Management
// ============================================================================

const state = {
    // Config
    classes: [],
    
    // Images
    images: [],
    currentIndex: -1,
    currentImage: null,
    confirmedImages: new Set(),
    
    // Annotations
    boxes: [],
    selectedBox: -1,
    history: [],
    
    // Drawing
    isDrawing: false,
    drawMode: true,
    startX: 0,
    startY: 0,
    currentX: 0,
    currentY: 0,
    currentClass: 0,
    
    // Pending box (waiting for label selection)
    pendingBox: null,
    
    // Resizing box
    isResizing: false,
    resizeHandle: null,  // 'tl', 'tr', 'bl', 'br', 't', 'b', 'l', 'r'
    resizeBoxOriginal: null,  // Original box before resize
    editingBoxIndex: -1,
    
    // Canvas & Zoom
    canvas: null,
    ctx: null,
    zoom: 1,
    offset: { x: 0, y: 0 },
    offsetRange: { x: 0, y: 0 },
    
    // Dragging
    isDragging: false,
    dragStart: { x: 0, y: 0 },
    
    // Image
    imgWidth: 0,
    imgHeight: 0,
    loadedImage: null,
    containerSize: { width: 0, height: 0 },
    
    // Modes
    singleLabelMode: false,
    autoSaveMode: true,
    
    // Flags
    isDirty: false,
};

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', async () => {
    state.canvas = document.getElementById('annotationCanvas');
    state.ctx = state.canvas.getContext('2d');
    
    await loadConfig();
    await loadImages();
    await loadStats();
    
    setupCanvasEvents();
    setupKeyboardEvents();
    setupButtonEvents();
    setupToggleEvents();
    
    if (state.images.length > 0) {
        selectImage(0);
    }
    
    updateModeIndicators();
    
    window.addEventListener('resize', () => {
        if (state.loadedImage) {
            handleSetCanvas();
        }
    });
});

// ============================================================================
// API Calls
// ============================================================================

async function loadConfig() {
    try {
        const res = await fetch('/api/config');
        const data = await res.json();
        state.classes = data.classes;
        if (state.classes.length > 0) {
            state.currentClass = state.classes[0].id;
        }
        renderClassList();
        renderLabelPopup();
        renderEditPopup();
    } catch (e) {
        console.error('Failed to load config:', e);
    }
}

async function loadImages() {
    try {
        const res = await fetch('/api/images');
        const data = await res.json();
        state.images = data.images;
        renderImageGallery();
    } catch (e) {
        console.error('Failed to load images:', e);
    }
}

async function loadStats() {
    try {
        const res = await fetch('/api/stats');
        const data = await res.json();
        document.getElementById('stats').textContent = data.progress;
    } catch (e) {
        console.error('Failed to load stats:', e);
    }
}

async function loadLabels(filename) {
    try {
        const res = await fetch(`/api/labels/${encodeURIComponent(filename)}`);
        const data = await res.json();
        state.boxes = data.boxes;
        state.history = [];
        state.isDirty = false;
    } catch (e) {
        console.error('Failed to load labels:', e);
        state.boxes = [];
    }
}

async function saveLabels() {
    if (!state.currentImage) return;
    
    try {
        const btn = document.getElementById('btnSave');
        btn.classList.add('saving');
        
        const res = await fetch(`/api/labels/${encodeURIComponent(state.currentImage)}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                filename: state.currentImage,
                boxes: state.boxes
            })
        });
        
        if (res.ok) {
            state.isDirty = false;
            updateImageItem(state.currentIndex, state.boxes.length > 0);
            showToast('💾 Đã lưu!', 'success');
            // Update stats in background (non-blocking)
            loadStats();
        }
        
        setTimeout(() => btn.classList.remove('saving'), 200);
    } catch (e) {
        console.error('Failed to save labels:', e);
        showToast('❌ Lưu thất bại', 'error');
    }
}

// ============================================================================
// UI Rendering
// ============================================================================

function renderClassList() {
    const container = document.getElementById('classList');
    container.innerHTML = state.classes.map((cls, i) => `
        <div class="class-item ${i === 0 ? 'active' : ''}" 
             data-class="${cls.id}"
             style="background: ${cls.color}">
            <kbd>${i + 1}</kbd>
            ${cls.name}
        </div>
    `).join('');
    
    container.querySelectorAll('.class-item').forEach(item => {
        item.addEventListener('click', () => {
            selectClass(parseInt(item.dataset.class));
        });
    });
}

function renderLabelPopup() {
    const container = document.getElementById('labelPopupList');
    container.innerHTML = state.classes.map((cls, i) => `
        <div class="label-popup-item" 
             data-class="${cls.id}"
             style="background: ${cls.color}">
            <kbd>${i + 1}</kbd>
            ${cls.name}
        </div>
    `).join('');
    
    container.querySelectorAll('.label-popup-item').forEach(item => {
        item.addEventListener('click', () => {
            assignLabelToBox(parseInt(item.dataset.class));
        });
    });
    
    // Close button
    document.getElementById('popupClose').addEventListener('click', () => {
        cancelPendingBox();
    });
}

function renderEditPopup() {
    const container = document.getElementById('editPopupList');
    container.innerHTML = state.classes.map((cls, i) => `
        <div class="label-popup-item" 
             data-class="${cls.id}"
             style="background: ${cls.color}">
            <kbd>${i + 1}</kbd>
            ${cls.name}
        </div>
    `).join('');
    
    container.querySelectorAll('.label-popup-item').forEach(item => {
        item.addEventListener('click', () => {
            assignLabelToEditingBox(parseInt(item.dataset.class));
        });
    });
    
    // Close button
    document.getElementById('editPopupClose').addEventListener('click', () => {
        hideEditPopup();
    });
    
    // Redraw button
    document.getElementById('btnRedraw').addEventListener('click', () => {
        startRedrawBox();
    });
}

function renderImageGallery() {
    const gallery = document.getElementById('imageGallery');
    gallery.innerHTML = state.images.map((img, i) => `
        <div class="image-item" data-index="${i}">
            <img class="image-thumb" src="/api/image/${encodeURIComponent(img)}" 
                 alt="${img}" loading="lazy">
            <span class="image-index">#${i + 1}</span>
            <span class="image-name">${img}</span>
            <span class="confirm-badge" title="Đã xác nhận">✓</span>
        </div>
    `).join('');
    
    gallery.querySelectorAll('.image-item').forEach(item => {
        item.addEventListener('click', () => {
            selectImage(parseInt(item.dataset.index));
        });
    });
    
    updateCounter();
}

function selectClass(classId) {
    state.currentClass = classId;
    document.querySelectorAll('.class-item').forEach(item => {
        item.classList.toggle('active', parseInt(item.dataset.class) === classId);
    });
}

function updateImageItem(index, hasLabels) {
    const items = document.querySelectorAll('.image-item');
    if (items[index]) {
        items[index].classList.toggle('labeled', hasLabels);
    }
}

function updateCounter() {
    document.getElementById('imageCounter').textContent = 
        `${state.currentIndex + 1}/${state.images.length}`;
}

function updateStatusBar() {
    document.getElementById('currentFile').textContent = 
        state.currentImage || 'Chưa chọn ảnh';
    document.getElementById('boxCount').textContent = 
        `${state.boxes.length} box`;
}

function updateModeIndicators() {
    document.getElementById('toggleDraw').checked = state.drawMode;
    document.getElementById('toggleSingleLabel').checked = state.singleLabelMode;
    document.getElementById('toggleAutoSave').checked = state.autoSaveMode;
    
    const container = document.getElementById('canvasContainer');
    if (state.drawMode) {
        container.style.cursor = 'none';
        document.getElementById('crosshairV').style.display = 'block';
        document.getElementById('crosshairH').style.display = 'block';
    } else {
        container.style.cursor = state.zoom > 1 ? 'grab' : 'default';
        document.getElementById('crosshairV').style.display = 'none';
        document.getElementById('crosshairH').style.display = 'none';
    }
}

function showToast(message, type = 'info') {
    let toast = document.getElementById('toast');
    if (!toast) {
        toast = document.createElement('div');
        toast.id = 'toast';
        toast.style.cssText = `
            position: fixed;
            bottom: 50px;
            left: 50%;
            transform: translateX(-50%);
            padding: 10px 20px;
            border-radius: 8px;
            color: white;
            font-weight: 500;
            z-index: 1000;
            transition: opacity 0.3s;
        `;
        document.body.appendChild(toast);
    }
    
    toast.textContent = message;
    toast.style.background = type === 'success' ? '#00d26a' : 
                             type === 'error' ? '#ff4757' : '#6c5ce7';
    toast.style.opacity = '1';
    
    setTimeout(() => toast.style.opacity = '0', 2000);
}

// ============================================================================
// Label Popup
// ============================================================================

function showLabelPopup(x, y) {
    const popup = document.getElementById('labelPopup');
    popup.style.left = `${x}px`;
    popup.style.top = `${y}px`;
    popup.classList.add('show');
}

function hideLabelPopup() {
    document.getElementById('labelPopup').classList.remove('show');
}

function assignLabelToBox(classId) {
    if (!state.pendingBox) return;
    
    state.pendingBox.class_id = classId;
    
    if (state.singleLabelMode) {
        state.history.push([...state.boxes]);
        state.boxes = [state.pendingBox];
    } else {
        state.history.push([...state.boxes]);
        state.boxes.push(state.pendingBox);
        
        // Turn off draw mode after drawing - user needs to press Q to draw again
        state.drawMode = false;
        updateModeIndicators();
    }
    
    state.isDirty = true;
    state.selectedBox = state.boxes.length - 1;
    state.pendingBox = null;
    
    hideLabelPopup();
    redrawCanvas();
    updateStatusBar();
}

function cancelPendingBox() {
    state.pendingBox = null;
    hideLabelPopup();
    redrawCanvas();
}

// ============================================================================
// Edit Box Popup
// ============================================================================

function showEditPopup(x, y) {
    const popup = document.getElementById('editPopup');
    popup.style.left = `${x}px`;
    popup.style.top = `${y}px`;
    popup.classList.add('show');
}

function hideEditPopup() {
    document.getElementById('editPopup').classList.remove('show');
    state.editingBoxIndex = -1;
}

function assignLabelToEditingBox(classId) {
    if (state.editingBoxIndex < 0 || state.editingBoxIndex >= state.boxes.length) return;
    
    state.history.push([...state.boxes]);
    state.boxes[state.editingBoxIndex].class_id = classId;
    state.isDirty = true;
    
    const cls = state.classes.find(c => c.id === classId);
    showToast(`📝 Đổi thành: ${cls ? cls.name : classId}`);
    
    hideEditPopup();
    redrawCanvas();
}

function startRedrawBox() {
    if (state.editingBoxIndex < 0 || state.editingBoxIndex >= state.boxes.length) return;
    
    // Delete the selected box and enable draw mode
    state.history.push([...state.boxes]);
    state.boxes.splice(state.editingBoxIndex, 1);
    state.selectedBox = -1;
    state.isDirty = true;
    
    // Turn on draw mode
    state.drawMode = true;
    updateModeIndicators();
    
    hideEditPopup();
    redrawCanvas();
    updateStatusBar();
    showToast('✏️ Vẽ lại bbox - bấm Q để hủy');
}

// ============================================================================
// Image Selection & Loading
// ============================================================================

async function selectImage(index) {
    if (index < 0 || index >= state.images.length) return;
    
    // Close popup if open
    hideLabelPopup();
    state.pendingBox = null;
    
    if (state.isDirty && state.autoSaveMode) {
        await saveLabels();
    }
    
    state.currentIndex = index;
    state.currentImage = state.images[index];
    state.selectedBox = -1;
    
    state.zoom = 1;
    state.offset = { x: 0, y: 0 };
    
    document.querySelectorAll('.image-item').forEach((item, i) => {
        item.classList.toggle('active', i === index);
        item.classList.toggle('confirmed', state.confirmedImages.has(state.images[i]));
    });
    
    const activeItem = document.querySelector('.image-item.active');
    if (activeItem) {
        activeItem.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
    
    await loadLabels(state.currentImage);
    await loadImageToCanvas(state.currentImage);
    
    updateCounter();
    updateStatusBar();
    
    document.getElementById('noImage').classList.add('hidden');
}

async function loadImageToCanvas(filename) {
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => {
            state.loadedImage = img;
            state.imgWidth = img.width;
            state.imgHeight = img.height;
            
            handleSetCanvas();
            resolve();
        };
        img.src = `/api/image/${encodeURIComponent(filename)}`;
    });
}

function handleSetCanvas() {
    const container = document.getElementById('canvasContainer');
    const canvas = state.canvas;
    const img = state.loadedImage;
    
    if (!img || !container) return;
    
    const containerWidth = container.clientWidth - 40;
    const containerHeight = container.clientHeight - 40;
    
    state.containerSize = { width: containerWidth, height: containerHeight };
    
    const scaleToFit = Math.min(containerWidth / img.width, containerHeight / img.height, 1);
    
    const viewWidth = img.width * scaleToFit * state.zoom;
    const viewHeight = img.height * scaleToFit * state.zoom;
    
    const canvasWidth = Math.min(containerWidth, viewWidth);
    const canvasHeight = Math.min(containerHeight, viewHeight);
    
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;
    
    state.offsetRange = {
        x: Math.max(0, (viewWidth - containerWidth) / 2),
        y: Math.max(0, (viewHeight - containerHeight) / 2)
    };
    
    state.offset.x = Math.max(-state.offsetRange.x, Math.min(state.offsetRange.x, state.offset.x));
    state.offset.y = Math.max(-state.offsetRange.y, Math.min(state.offsetRange.y, state.offset.y));
    
    redrawCanvas();
}

// ============================================================================
// Canvas Drawing
// ============================================================================

function redrawCanvas() {
    const { ctx, canvas, loadedImage, zoom, offset } = state;
    
    if (!loadedImage) return;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const scaleToFit = Math.min(
        state.containerSize.width / loadedImage.width, 
        state.containerSize.height / loadedImage.height, 
        1
    );
    
    const imgDrawWidth = loadedImage.width * scaleToFit * zoom;
    const imgDrawHeight = loadedImage.height * scaleToFit * zoom;
    
    const drawX = (canvas.width - imgDrawWidth) / 2 + offset.x;
    const drawY = (canvas.height - imgDrawHeight) / 2 + offset.y;
    
    ctx.drawImage(loadedImage, drawX, drawY, imgDrawWidth, imgDrawHeight);
    
    state.boxes.forEach((box, i) => {
        drawBox(box, i === state.selectedBox, drawX, drawY, imgDrawWidth, imgDrawHeight);
    });
    
    // Draw pending box
    if (state.pendingBox) {
        drawBox(state.pendingBox, true, drawX, drawY, imgDrawWidth, imgDrawHeight);
    }
    
    if (state.isDrawing) {
        drawTempBox();
    }
    
    if (zoom !== 1) {
        ctx.fillStyle = 'rgba(0,0,0,0.7)';
        ctx.fillRect(10, 10, 60, 24);
        ctx.fillStyle = 'white';
        ctx.font = '12px sans-serif';
        ctx.fillText(`${zoom}x`, 28, 27);
    }
}

function drawBox(box, isSelected, imgX, imgY, imgW, imgH) {
    const { ctx } = state;
    
    const x = imgX + (box.cx - box.w / 2) * imgW;
    const y = imgY + (box.cy - box.h / 2) * imgH;
    const w = box.w * imgW;
    const h = box.h * imgH;
    
    const cls = state.classes.find(c => c.id === box.class_id);
    const color = cls ? cls.color : '#FF0000';
    
    ctx.strokeStyle = color;
    ctx.lineWidth = isSelected ? 3 : 2;
    ctx.strokeRect(x, y, w, h);
    
    if (isSelected) {
        ctx.fillStyle = color + '30';
        ctx.fillRect(x, y, w, h);
        
        // Draw resize handles (8 squares)
        const handleSize = 8;
        ctx.fillStyle = color;
        
        // Corners
        ctx.fillRect(x - handleSize/2, y - handleSize/2, handleSize, handleSize); // tl
        ctx.fillRect(x + w - handleSize/2, y - handleSize/2, handleSize, handleSize); // tr
        ctx.fillRect(x - handleSize/2, y + h - handleSize/2, handleSize, handleSize); // bl
        ctx.fillRect(x + w - handleSize/2, y + h - handleSize/2, handleSize, handleSize); // br
        
        // Edges
        ctx.fillRect(x + w/2 - handleSize/2, y - handleSize/2, handleSize, handleSize); // t
        ctx.fillRect(x + w/2 - handleSize/2, y + h - handleSize/2, handleSize, handleSize); // b
        ctx.fillRect(x - handleSize/2, y + h/2 - handleSize/2, handleSize, handleSize); // l
        ctx.fillRect(x + w - handleSize/2, y + h/2 - handleSize/2, handleSize, handleSize); // r
    }
    
    if (box.class_id !== -1) {
        const label = cls ? cls.name.substring(0, 12) : `Class ${box.class_id}`;
        ctx.fillStyle = color;
        ctx.font = 'bold 11px sans-serif';
        const textWidth = ctx.measureText(label).width;
        ctx.fillRect(x, y - 16, textWidth + 6, 16);
        ctx.fillStyle = 'white';
        ctx.fillText(label, x + 3, y - 4);
    }
}

function drawTempBox() {
    const { ctx, startX, startY, currentX, currentY } = state;
    
    const x = Math.min(startX, currentX);
    const y = Math.min(startY, currentY);
    const w = Math.abs(currentX - startX);
    const h = Math.abs(currentY - startY);
    
    ctx.strokeStyle = '#FF0000';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
}

// Helper to get box coordinates in canvas space
function getBoxCanvasCoords(boxIndex) {
    if (boxIndex < 0 || boxIndex >= state.boxes.length) return null;
    
    const box = state.boxes[boxIndex];
    const scaleToFit = Math.min(
        state.containerSize.width / state.imgWidth, 
        state.containerSize.height / state.imgHeight, 
        1
    );
    const imgDrawWidth = state.imgWidth * scaleToFit * state.zoom;
    const imgDrawHeight = state.imgHeight * scaleToFit * state.zoom;
    const imgX = (state.canvas.width - imgDrawWidth) / 2 + state.offset.x;
    const imgY = (state.canvas.height - imgDrawHeight) / 2 + state.offset.y;
    
    return {
        x: imgX + (box.cx - box.w / 2) * imgDrawWidth,
        y: imgY + (box.cy - box.h / 2) * imgDrawHeight,
        w: box.w * imgDrawWidth,
        h: box.h * imgDrawHeight,
        imgX, imgY, imgDrawWidth, imgDrawHeight
    };
}

// Detect which resize handle was clicked
function getResizeHandle(canvasX, canvasY, boxIndex) {
    const coords = getBoxCanvasCoords(boxIndex);
    if (!coords) return null;
    
    const { x, y, w, h } = coords;
    const handleSize = 12; // Slightly larger for easier clicking
    
    const handles = [
        { name: 'tl', cx: x, cy: y },
        { name: 'tr', cx: x + w, cy: y },
        { name: 'bl', cx: x, cy: y + h },
        { name: 'br', cx: x + w, cy: y + h },
        { name: 't', cx: x + w/2, cy: y },
        { name: 'b', cx: x + w/2, cy: y + h },
        { name: 'l', cx: x, cy: y + h/2 },
        { name: 'r', cx: x + w, cy: y + h/2 },
    ];
    
    for (const handle of handles) {
        if (Math.abs(canvasX - handle.cx) < handleSize && 
            Math.abs(canvasY - handle.cy) < handleSize) {
            return handle.name;
        }
    }
    return null;
}

// ============================================================================
// Canvas Events
// ============================================================================

function setupCanvasEvents() {
    const container = document.getElementById('canvasContainer');
    const canvas = state.canvas;
    
    container.addEventListener('wheel', handleScrollZoom, { passive: false });
    
    canvas.addEventListener('mousedown', handleMouseDown);
    canvas.addEventListener('mousemove', handleMouseMove);
    canvas.addEventListener('mouseup', handleMouseUp);
    canvas.addEventListener('mouseleave', handleMouseLeave);
    
    container.addEventListener('mousemove', updateCrosshair);
    container.addEventListener('mouseleave', hideCrosshair);
    
    canvas.addEventListener('contextmenu', e => e.preventDefault());
}

function handleScrollZoom(e) {
    e.preventDefault();
    
    if (e.deltaY < 0) {
        state.zoom = Math.min(state.zoom + 1, 10);
    } else {
        state.zoom = Math.max(1, state.zoom - 1);
        if (state.zoom === 1) {
            state.offset = { x: 0, y: 0 };
        }
    }
    handleSetCanvas();
}

function updateCrosshair(e) {
    if (!state.drawMode) return;
    
    const container = document.getElementById('canvasContainer');
    const rect = container.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    // Make sure crosshair is visible
    document.getElementById('crosshairV').style.display = 'block';
    document.getElementById('crosshairH').style.display = 'block';
    
    document.getElementById('crosshairV').style.left = `${x}px`;
    document.getElementById('crosshairH').style.top = `${y}px`;
}

function hideCrosshair() {
    document.getElementById('crosshairV').style.display = 'none';
    document.getElementById('crosshairH').style.display = 'none';
}

function showCrosshair() {
    if (!state.drawMode) return;
    document.getElementById('crosshairV').style.display = 'block';
    document.getElementById('crosshairH').style.display = 'block';
}

function getCanvasCoords(e) {
    const rect = state.canvas.getBoundingClientRect();
    return {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
    };
}

function handleMouseDown(e) {
    const coords = getCanvasCoords(e);
    
    // Middle button always pans
    if (e.button === 1) {
        state.isDragging = true;
        state.dragStart = { x: e.clientX, y: e.clientY };
        state.canvas.style.cursor = 'grabbing';
        return;
    }
    
    if (e.button !== 0) return;
    
    // Check if clicking on a resize handle of selected box
    if (state.selectedBox >= 0) {
        const handle = getResizeHandle(coords.x, coords.y, state.selectedBox);
        if (handle) {
            // Start resizing
            state.isResizing = true;
            state.resizeHandle = handle;
            state.resizeBoxOriginal = { ...state.boxes[state.selectedBox] };
            state.startX = coords.x;
            state.startY = coords.y;
            state.canvas.style.cursor = getResizeCursor(handle);
            return;
        }
    }
    
    // Check if clicking on a box (works in both draw mode and select mode)
    const clickedBox = findBoxAtPoint(coords.x, coords.y);
    
    if (clickedBox >= 0) {
        // Select the box
        state.selectedBox = clickedBox;
        redrawCanvas();
        return;
    }
    
    // Not clicking on a box
    if (state.drawMode) {
        // Deselect any selected box first
        if (state.selectedBox >= 0) {
            state.selectedBox = -1;
            redrawCanvas();
        }
        // Start drawing
        state.isDrawing = true;
        state.startX = coords.x;
        state.startY = coords.y;
        state.currentX = coords.x;
        state.currentY = coords.y;
        state.selectedBox = -1;
    } else if (state.zoom > 1) {
        // Deselect first
        state.selectedBox = -1;
        // Pan mode when zoomed and not clicking on box
        state.isDragging = true;
        state.dragStart = { x: e.clientX, y: e.clientY };
        state.canvas.style.cursor = 'grabbing';
        redrawCanvas();
    } else {
        // Deselect
        state.selectedBox = -1;
        redrawCanvas();
    }
}

function getResizeCursor(handle) {
    const cursors = {
        'tl': 'nwse-resize', 'br': 'nwse-resize',
        'tr': 'nesw-resize', 'bl': 'nesw-resize',
        't': 'ns-resize', 'b': 'ns-resize',
        'l': 'ew-resize', 'r': 'ew-resize'
    };
    return cursors[handle] || 'default';
}

function handleMouseMove(e) {
    const coords = getCanvasCoords(e);
    
    if (state.isDragging) {
        const dx = e.clientX - state.dragStart.x;
        const dy = e.clientY - state.dragStart.y;
        state.dragStart = { x: e.clientX, y: e.clientY };
        
        state.offset.x += dx;
        state.offset.y += dy;
        
        state.offset.x = Math.max(-state.offsetRange.x, Math.min(state.offsetRange.x, state.offset.x));
        state.offset.y = Math.max(-state.offsetRange.y, Math.min(state.offsetRange.y, state.offset.y));
        
        redrawCanvas();
        return;
    }
    
    // Handle resizing
    if (state.isResizing && state.selectedBox >= 0) {
        const dx = coords.x - state.startX;
        const dy = coords.y - state.startY;
        
        // Get image draw coordinates
        const scaleToFit = Math.min(
            state.containerSize.width / state.imgWidth, 
            state.containerSize.height / state.imgHeight, 
            1
        );
        const imgDrawWidth = state.imgWidth * scaleToFit * state.zoom;
        const imgDrawHeight = state.imgHeight * scaleToFit * state.zoom;
        
        // Convert delta to normalized coordinates
        const ndx = dx / imgDrawWidth;
        const ndy = dy / imgDrawHeight;
        
        const orig = state.resizeBoxOriginal;
        const box = state.boxes[state.selectedBox];
        const handle = state.resizeHandle;
        
        // Calculate new box based on handle
        let left = orig.cx - orig.w / 2;
        let top = orig.cy - orig.h / 2;
        let right = orig.cx + orig.w / 2;
        let bottom = orig.cy + orig.h / 2;
        
        if (handle.includes('l')) left += ndx;
        if (handle.includes('r')) right += ndx;
        if (handle.includes('t')) top += ndy;
        if (handle.includes('b')) bottom += ndy;
        
        // Ensure min size
        const minSize = 0.01;
        if (right - left < minSize) right = left + minSize;
        if (bottom - top < minSize) bottom = top + minSize;
        
        // Clamp to image bounds
        left = Math.max(0, left);
        top = Math.max(0, top);
        right = Math.min(1, right);
        bottom = Math.min(1, bottom);
        
        // Update box
        box.cx = (left + right) / 2;
        box.cy = (top + bottom) / 2;
        box.w = right - left;
        box.h = bottom - top;
        
        redrawCanvas();
        return;
    }
    
    // Update cursor for resize handles when hovering
    if (state.selectedBox >= 0 && !state.isDrawing) {
        const handle = getResizeHandle(coords.x, coords.y, state.selectedBox);
        if (handle) {
            state.canvas.style.cursor = getResizeCursor(handle);
        } else {
            state.canvas.style.cursor = state.drawMode ? 'crosshair' : 'default';
        }
    }
    
    document.getElementById('coords').textContent = 
        `x: ${(coords.x / state.canvas.width).toFixed(3)}, y: ${(coords.y / state.canvas.height).toFixed(3)}`;
    
    if (state.isDrawing) {
        state.currentX = coords.x;
        state.currentY = coords.y;
        redrawCanvas();
    }
}

function handleMouseUp(e) {
    if (state.isDragging) {
        state.isDragging = false;
        state.canvas.style.cursor = state.drawMode ? 'crosshair' : (state.zoom > 1 ? 'grab' : 'default');
        return;
    }
    
    // Handle resize completion
    if (state.isResizing) {
        // Save history (before the resize)
        state.history.push([...state.boxes.map(b => ({ ...b }))]);
        // Replace the current box with resized version (already updated in mousemove)
        state.history[state.history.length - 1][state.selectedBox] = { ...state.resizeBoxOriginal };
        
        state.isResizing = false;
        state.resizeHandle = null;
        state.resizeBoxOriginal = null;
        state.isDirty = true;
        state.canvas.style.cursor = 'default';
        
        showToast('📐 Đã resize box');
        redrawCanvas();
        updateStatusBar();
        return;
    }
    
    if (!state.isDrawing) return;
    
    const coords = getCanvasCoords(e);
    state.currentX = coords.x;
    state.currentY = coords.y;
    state.isDrawing = false;
    
    const x1 = Math.min(state.startX, coords.x);
    const y1 = Math.min(state.startY, coords.y);
    const x2 = Math.max(state.startX, coords.x);
    const y2 = Math.max(state.startY, coords.y);
    
    const w = x2 - x1;
    const h = y2 - y1;
    
    if (w < 5 || h < 5) {
        redrawCanvas();
        return;
    }
    
    const scaleToFit = Math.min(
        state.containerSize.width / state.imgWidth, 
        state.containerSize.height / state.imgHeight, 
        1
    );
    const imgDrawWidth = state.imgWidth * scaleToFit * state.zoom;
    const imgDrawHeight = state.imgHeight * scaleToFit * state.zoom;
    const imgX = (state.canvas.width - imgDrawWidth) / 2 + state.offset.x;
    const imgY = (state.canvas.height - imgDrawHeight) / 2 + state.offset.y;
    
    const box = {
        class_id: state.singleLabelMode ? state.currentClass : -1,  // -1 means pending
        cx: Math.max(0, Math.min(1, (x1 + w/2 - imgX) / imgDrawWidth)),
        cy: Math.max(0, Math.min(1, (y1 + h/2 - imgY) / imgDrawHeight)),
        w: Math.max(0, Math.min(1, w / imgDrawWidth)),
        h: Math.max(0, Math.min(1, h / imgDrawHeight))
    };
    
    if (state.singleLabelMode) {
        // Single label mode: use current class directly
        box.class_id = state.currentClass;
        state.history.push([...state.boxes]);
        state.boxes = [box];
        state.isDirty = true;
        state.selectedBox = 0;
        redrawCanvas();
        updateStatusBar();
    } else {
        // Multi-label mode: show popup to select class
        state.pendingBox = box;
        redrawCanvas();
        
        // Show popup near the drawn box
        const canvasRect = state.canvas.getBoundingClientRect();
        showLabelPopup(canvasRect.left + x2 + 10, canvasRect.top + y1);
    }
}

function handleMouseLeave() {
    if (state.isDrawing) {
        state.isDrawing = false;
        redrawCanvas();
    }
    if (state.isDragging) {
        state.isDragging = false;
    }
}

function findBoxAtPoint(x, y) {
    const scaleToFit = Math.min(
        state.containerSize.width / state.imgWidth, 
        state.containerSize.height / state.imgHeight, 
        1
    );
    const imgDrawWidth = state.imgWidth * scaleToFit * state.zoom;
    const imgDrawHeight = state.imgHeight * scaleToFit * state.zoom;
    const imgX = (state.canvas.width - imgDrawWidth) / 2 + state.offset.x;
    const imgY = (state.canvas.height - imgDrawHeight) / 2 + state.offset.y;
    
    for (let i = state.boxes.length - 1; i >= 0; i--) {
        const box = state.boxes[i];
        const bx = imgX + (box.cx - box.w / 2) * imgDrawWidth;
        const by = imgY + (box.cy - box.h / 2) * imgDrawHeight;
        const bw = box.w * imgDrawWidth;
        const bh = box.h * imgDrawHeight;
        
        if (x >= bx && x <= bx + bw && y >= by && y <= by + bh) {
            return i;
        }
    }
    return -1;
}

// ============================================================================
// Keyboard Events
// ============================================================================

function setupKeyboardEvents() {
    document.addEventListener('keydown', handleKeyDown);
}

function handleKeyDown(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    
    const key = e.key.toLowerCase();
    
    // If popup is open, number keys select label
    if (state.pendingBox && /^[1-9]$/.test(key)) {
        const classIndex = parseInt(key) - 1;
        if (classIndex < state.classes.length) {
            assignLabelToBox(state.classes[classIndex].id);
        }
        return;
    }
    
    // Number keys 1-9 for class selection
    if (/^[1-9]$/.test(key)) {
        const classIndex = parseInt(key) - 1;
        if (classIndex < state.classes.length) {
            selectClass(state.classes[classIndex].id);
            showToast(`Nhãn: ${state.classes[classIndex].name}`);
        }
        return;
    }
    
    switch (key) {
        case 'q':
            state.drawMode = !state.drawMode;
            updateModeIndicators();
            showToast(state.drawMode ? '✏️ Chế độ vẽ BẬT' : '👆 Chế độ chọn');
            break;
            
        case 'e':
            saveLabels();
            break;
            
        case 'a':
        case 'arrowup':
            navigateImage(-1);
            e.preventDefault();
            break;
            
        case 'd':
        case 'arrowdown':
            navigateImage(1);
            e.preventDefault();
            break;
            
        case 'r':
        case 'delete':
        case 'backspace':
            deleteSelectedBox();
            e.preventDefault();
            break;
            
        case 'f':
            editSelectedBox();
            break;
            
        case 'k':
            state.singleLabelMode = !state.singleLabelMode;
            updateModeIndicators();
            showToast(state.singleLabelMode ? '1️⃣ Nhãn đơn BẬT' : '📦 Nhiều nhãn');
            break;
            
        case 'l':
            state.autoSaveMode = !state.autoSaveMode;
            updateModeIndicators();
            showToast(state.autoSaveMode ? '💾 Tự lưu BẬT' : '✋ Lưu thủ công');
            break;
            
        case 'z':
            if (e.ctrlKey || e.metaKey) {
                undo();
                e.preventDefault();
            } else {
                confirmImage();
            }
            break;
            
        case 'x':
            unconfirmImage();
            break;
            
        case 'escape':
            if (state.pendingBox) {
                cancelPendingBox();
            } else {
                state.isDrawing = false;
                state.selectedBox = -1;
                redrawCanvas();
            }
            break;
            
        case '?':
            document.getElementById('helpModal').classList.toggle('show');
            break;
    }
}

// ============================================================================
// Button & Toggle Events
// ============================================================================

function setupButtonEvents() {
    document.getElementById('btnSave').addEventListener('click', saveLabels);
    document.getElementById('btnUndo').addEventListener('click', undo);
    document.getElementById('btnClear').addEventListener('click', clearAll);
    document.getElementById('btnPrev').addEventListener('click', () => navigateImage(-1));
    document.getElementById('btnNext').addEventListener('click', () => navigateImage(1));
}

function setupToggleEvents() {
    document.getElementById('toggleDraw').addEventListener('change', (e) => {
        state.drawMode = e.target.checked;
        updateModeIndicators();
    });
    
    document.getElementById('toggleSingleLabel').addEventListener('change', (e) => {
        state.singleLabelMode = e.target.checked;
    });
    
    document.getElementById('toggleAutoSave').addEventListener('change', (e) => {
        state.autoSaveMode = e.target.checked;
    });
}

// ============================================================================
// Actions
// ============================================================================

function deleteSelectedBox() {
    if (state.selectedBox < 0) {
        showToast('⚠️ Chưa chọn box');
        return;
    }
    
    state.history.push([...state.boxes]);
    state.boxes.splice(state.selectedBox, 1);
    state.selectedBox = -1;
    state.isDirty = true;
    
    redrawCanvas();
    updateStatusBar();
    showToast('🗑️ Đã xóa box');
}

function editSelectedBox() {
    if (state.selectedBox < 0) {
        showToast('⚠️ Chưa chọn box');
        return;
    }
    
    // Store the box being edited
    state.editingBoxIndex = state.selectedBox;
    
    // Get box position on canvas for popup placement
    const box = state.boxes[state.selectedBox];
    const scaleToFit = Math.min(
        state.containerSize.width / state.imgWidth, 
        state.containerSize.height / state.imgHeight, 
        1
    );
    const imgDrawWidth = state.imgWidth * scaleToFit * state.zoom;
    const imgDrawHeight = state.imgHeight * scaleToFit * state.zoom;
    const imgX = (state.canvas.width - imgDrawWidth) / 2 + state.offset.x;
    const imgY = (state.canvas.height - imgDrawHeight) / 2 + state.offset.y;
    
    const bx = imgX + (box.cx + box.w / 2) * imgDrawWidth;
    const by = imgY + (box.cy - box.h / 2) * imgDrawHeight;
    
    // Show edit popup
    const canvasRect = state.canvas.getBoundingClientRect();
    showEditPopup(canvasRect.left + bx + 10, canvasRect.top + by);
}

function confirmImage() {
    if (!state.currentImage) return;
    
    state.confirmedImages.add(state.currentImage);
    updateImageItem(state.currentIndex, state.boxes.length > 0);
    document.querySelectorAll('.image-item')[state.currentIndex]?.classList.add('confirmed');
    showToast('✅ Đã xác nhận ảnh');
    
    if (state.isDirty) {
        saveLabels();
    }
}

function unconfirmImage() {
    if (!state.currentImage) return;
    
    state.confirmedImages.delete(state.currentImage);
    document.querySelectorAll('.image-item')[state.currentIndex]?.classList.remove('confirmed');
    showToast('❌ Đã hủy xác nhận');
}

function undo() {
    if (state.history.length === 0) {
        showToast('⚠️ Không có gì để hoàn tác');
        return;
    }
    
    state.boxes = state.history.pop();
    state.selectedBox = -1;
    state.isDirty = true;
    
    redrawCanvas();
    updateStatusBar();
    showToast('↩️ Hoàn tác');
}

function clearAll() {
    if (state.boxes.length === 0) return;
    
    if (confirm('Xóa tất cả box trong ảnh này?')) {
        state.history.push([...state.boxes]);
        state.boxes = [];
        state.selectedBox = -1;
        state.isDirty = true;
        
        redrawCanvas();
        updateStatusBar();
        showToast('🗑️ Đã xóa tất cả');
    }
}

function navigateImage(delta) {
    const newIndex = state.currentIndex + delta;
    if (newIndex >= 0 && newIndex < state.images.length) {
        selectImage(newIndex);
    }
}
