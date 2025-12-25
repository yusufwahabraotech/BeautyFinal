import os
import torch
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import trimesh
from scipy.spatial import ConvexHull
import tempfile
import sys

# Add repositories to path
sys.path.insert(0, '/app/pifuhd')
sys.path.insert(0, '/app/lightweight-human-pose-estimation.pytorch')

# Import pose estimation modules
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose
import demo

app = FastAPI(title="Body Measurement API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load pose estimation model at startup
print("Loading pose estimation model...")
net = PoseEstimationWithMobileNet()
checkpoint_path = '/app/lightweight-human-pose-estimation.pytorch/checkpoint_iter_370000.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu')
load_state(net, checkpoint)

# Move to GPU if available
if torch.cuda.is_available():
    net = net.cuda()
    print("✓ Pose estimation model loaded to GPU")
else:
    print("✓ Pose estimation model loaded to CPU (no GPU available)")

# ==================== Helper Functions ====================

def get_rect(net, images, height_size):
    """Extract bounding rectangles from images using pose estimation."""
    net.eval()
    
    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    
    for image_path in images:
        rect_path = image_path.replace(f'.{image_path.split(".")[-1]}', '_rect.txt')
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Inference
        heatmaps, pafs, scale, pad = demo.infer_fast(
            net, img, height_size, stride, upsample_ratio, 
            cpu=not torch.cuda.is_available()
        )
        
        # Extract keypoints
        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):
            total_keypoints_num += extract_keypoints(
                heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num
            )
        
        # Group keypoints into poses
        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        
        # Adjust keypoints to original image coordinates
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        
        # Calculate bounding rectangles for each pose
        rects = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
                
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            valid_keypoints = []
            
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
                    valid_keypoints.append([pose_keypoints[kpt_id, 0], pose_keypoints[kpt_id, 1]])
            
            valid_keypoints = np.array(valid_keypoints)
            
            # Determine bounding box based on detected keypoints
            if len(valid_keypoints) > 0:
                if pose_entries[n][10] != -1.0 or pose_entries[n][13] != -1.0:  # Hip keypoints
                    pmin = valid_keypoints.min(0)
                    pmax = valid_keypoints.max(0)
                    center = (0.5 * (pmax[:2] + pmin[:2])).astype(int)
                    radius = int(0.65 * max(pmax[0]-pmin[0], pmax[1]-pmin[1]))
                    
                elif pose_entries[n][8] != -1.0 and pose_entries[n][11] != -1.0:  # Shoulder/hip alternative
                    center = (0.5 * (pose_keypoints[8] + pose_keypoints[11])).astype(int)
                    radius = int(1.45*np.sqrt(((center[None,:] - valid_keypoints)**2).sum(1)).max(0))
                    center[1] += int(0.05*radius)
                    
                else:
                    center = np.array([img.shape[1]//2, img.shape[0]//2])
                    radius = max(img.shape[1]//2, img.shape[0]//2)
                
                x1 = center[0] - radius
                y1 = center[1] - radius
                rects.append([x1, y1, 2*radius, 2*radius])
        
        # Save rectangles to file for pifuhd
        np.savetxt(rect_path, np.array(rects), fmt='%d')

def get_slice_vertices(mesh, y_level, tolerance=0.03):
    """Get mesh vertices at a specific height level."""
    return mesh.vertices[np.abs(mesh.vertices[:,1]-y_level) < tolerance]

def approx_circ(width, depth):
    """Approximate circumference from width and depth using Ramanujan's formula."""
    return np.pi*(3*(width+depth) - np.sqrt((3*width+depth)*(width+3*depth)))/2

def measure_shoulder(mesh, num_slices=200, tolerance=0.02, smooth_window=5):
    """Measure shoulder width at upper torso."""
    y_min, y_max = mesh.bounds[:, 1]
    y_levels = np.linspace(y_min + 0.5*(y_max - y_min), y_max, num_slices)
    widths = []
    
    for y in y_levels:
        slice_vertices = get_slice_vertices(mesh, y, tolerance)
        if len(slice_vertices) >= 3:
            hull = ConvexHull(slice_vertices[:, [0,2]])
            widths.append(hull.volume)
        else:
            widths.append(0)
    
    widths = np.convolve(widths, np.ones(smooth_window)/smooth_window, mode='same')
    max_index = np.argmax(widths)
    shoulder_y = y_levels[max_index]
    slice_vertices = get_slice_vertices(mesh, shoulder_y, tolerance)
    shoulder_width = np.max(slice_vertices[:,0]) - np.min(slice_vertices[:,0])
    
    return shoulder_width, shoulder_y

def measure_waist(mesh, num_slices=300, tolerance=0.01, smooth_window=5):
    """Measure waist circumference at mid-torso."""
    y_min, y_max = mesh.bounds[:,1]
    y_levels = np.linspace(y_min + 0.4*(y_max - y_min), y_min + 0.45*(y_max - y_min), num_slices)
    widths = []
    
    for y in y_levels:
        slice_vertices = get_slice_vertices(mesh, y, tolerance)
        if len(slice_vertices) >= 3:
            hull = ConvexHull(slice_vertices[:, [0,2]])
            widths.append(hull.volume)
        else:
            widths.append(np.inf)
    
    widths = np.convolve(widths, np.ones(smooth_window)/smooth_window, mode='same')
    local_min_indices = [i for i in range(1,len(widths)-1) if widths[i]<widths[i-1] and widths[i]<widths[i+1]]
    
    if not local_min_indices:
        min_index = np.argmin(widths)
    else:
        mid_index = len(y_levels)//2
        min_index = min(local_min_indices, key=lambda i: abs(i-mid_index))
    
    waist_y = y_levels[min_index]
    slice_vertices = get_slice_vertices(mesh, waist_y, tolerance)
    waist_width = np.max(slice_vertices[:,0]) - np.min(slice_vertices[:,0])
    
    if len(slice_vertices) >= 3:
        waist_depth = np.max(slice_vertices[:,2]) - np.min(slice_vertices[:,2])
        waist_circ = approx_circ(waist_width, waist_depth)
    else:
        waist_circ = waist_width
    
    return waist_width, waist_y, waist_circ

def measure_hip(front_mesh, side_mesh, num_slices=200, tolerance=0.03):
    """Measure hip circumference at lower torso."""
    y_min, y_max = front_mesh.bounds[:,1]
    y_levels = np.linspace(y_min + 0.4*(y_max - y_min), y_min + 0.41*(y_max - y_min), num_slices)
    max_width = 0
    hip_y = 0
    
    for y in y_levels:
        slice_vertices = get_slice_vertices(front_mesh, y, tolerance)
        if len(slice_vertices) >= 3:
            width = np.max(slice_vertices[:,0]) - np.min(slice_vertices[:,0])
            if width > max_width:
                max_width = width
                hip_y = y
    
    slice_vertices = get_slice_vertices(front_mesh, hip_y, tolerance)
    hip_width = np.max(slice_vertices[:,0]) - np.min(slice_vertices[:,0]) if len(slice_vertices) >= 2 else 0
    
    slice_side = get_slice_vertices(side_mesh, hip_y, tolerance)
    hip_depth = np.max(slice_side[:,2]) - np.min(slice_side[:,2]) if len(slice_side) >= 2 else 0
    
    hip_circ = approx_circ(hip_width, hip_depth)
    return hip_width, hip_depth, hip_circ, hip_y

def measure_bust(mesh, num_slices=200, tolerance_ratio=0.01):
    """Measure bust circumference at upper chest."""
    y_min, y_max = mesh.bounds[:,1]
    height = y_max - y_min
    tolerance = tolerance_ratio * height
    y_levels = np.linspace(y_min + 0.55*height, y_min + 0.65*height, num_slices)
    max_width = 0
    bust_y = 0
    
    for y in y_levels:
        slice_vertices = get_slice_vertices(mesh, y, tolerance)
        if len(slice_vertices)>=3:
            width = np.max(slice_vertices[:,0]) - np.min(slice_vertices[:,0])
            if width > max_width:
                max_width = width
                bust_y = y
    
    slice_vertices = get_slice_vertices(mesh, bust_y, tolerance)
    bust_width = np.max(slice_vertices[:,0]) - np.min(slice_vertices[:,0]) if len(slice_vertices) >= 2 else 0
    bust_depth = np.max(slice_vertices[:,2]) - np.min(slice_vertices[:,2]) if len(slice_vertices) >= 2 else 0
    bust_circ = approx_circ(bust_width, bust_depth)
    bust_y_from_bottom = bust_y - y_min
    
    return bust_width, bust_depth, bust_circ, bust_y_from_bottom

def measure_neck(mesh, num_slices=200, tolerance_ratio=0.01):
    """Measure neck circumference at top of torso."""
    y_min, y_max = mesh.bounds[:,1]
    height = y_max - y_min
    tolerance = tolerance_ratio * height
    y_levels = np.linspace(y_min + 0.9*height, y_min + height, num_slices)
    min_width = np.inf
    neck_y = 0
    
    for y in y_levels:
        slice_vertices = get_slice_vertices(mesh, y, tolerance)
        if len(slice_vertices)>=3:
            width = np.max(slice_vertices[:,0]) - np.min(slice_vertices[:,0])
            if width < min_width:
                min_width = width
                neck_y = y
    
    slice_vertices = get_slice_vertices(mesh, neck_y, tolerance)
    neck_width = np.max(slice_vertices[:,0]) - np.min(slice_vertices[:,0]) if len(slice_vertices) >= 2 else 0
    neck_depth = np.max(slice_vertices[:,2]) - np.min(slice_vertices[:,2]) if len(slice_vertices) >= 2 else 0
    neck_circ = approx_circ(neck_width, neck_depth)
    neck_y_from_bottom = neck_y - y_min
    
    return neck_width, neck_depth, neck_circ, neck_y_from_bottom

def measure_arm_length(front_mesh, side_mesh, shoulder_y, wrist_y=None):
    """Measure arm length from shoulder to wrist."""
    y_min, y_max = front_mesh.bounds[:,1]
    if wrist_y is None:
        wrist_y = y_min + 0.35*(y_max - y_min)
    
    arm_slice_shoulder = get_slice_vertices(front_mesh, shoulder_y, 0.1)
    arm_slice_wrist = get_slice_vertices(front_mesh, wrist_y, 0.05)
    
    if len(arm_slice_shoulder)>=1 and len(arm_slice_wrist)>=1:
        shoulder_centroid = np.mean(arm_slice_shoulder, axis=0)
        wrist_centroid = np.mean(arm_slice_wrist, axis=0)
        arm_length = np.linalg.norm(shoulder_centroid - wrist_centroid)
    else:
        arm_length = 0
    
    return arm_length

def measure_inseam(mesh, num_slices=200, tolerance_ratio=0.01):
    """Measure inseam length (crotch to ankle)."""
    y_min, y_max = mesh.bounds[:,1]
    height = y_max - y_min
    tolerance = tolerance_ratio * height
    y_levels = np.linspace(y_min, y_min + 0.4*height, num_slices)
    min_y = y_min
    
    for y in y_levels:
        slice_vertices = get_slice_vertices(mesh, y, tolerance)
        if len(slice_vertices)>=3:
            min_y = y
            break
    
    inseam = y_max - min_y
    return inseam

def m_to_cm(x):
    """Convert meters to centimeters."""
    return round(x*100, 2)

# ==================== API Endpoints ====================

@app.get("/")
async def root():
    """Root endpoint - API status."""
    return {
        "status": "running",
        "message": "Body Measurement API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "process": "/process (POST)"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "Body Measurement API is running",
        "gpu_available": torch.cuda.is_available()
    }

@app.post("/process")
async def process_images(front_image: UploadFile = File(...), side_image: UploadFile = File(...)):
    """
    Process front and side images to generate body measurements.
    
    Requires:
    - front_image: JPEG/PNG image of person's front view
    - side_image: JPEG/PNG image of person's side view
    
    Returns:
    - JSON with measurements in cm and inches
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded images
            front_path = os.path.join(temp_dir, "front.jpg")
            side_path = os.path.join(temp_dir, "side.jpg")
            
            with open(front_path, "wb") as f:
                f.write(await front_image.read())
            with open(side_path, "wb") as f:
                f.write(await side_image.read())
            
            if not os.path.exists(front_path) or not os.path.exists(side_path):
                raise HTTPException(status_code=400, detail="Images not saved properly")
            
            # Extract pose rectangles
            get_rect(net, [front_path], 512)
            get_rect(net, [side_path], 512)
            
            # Run PIFuHD reconstruction
            os.chdir('/app/pifuhd')
            os.system(f"python -m apps.simple_test -r 256 --use_rect -i {temp_dir}")
            os.chdir('/app')
            
            # Find generated mesh files
            front_obj_path = os.path.join(temp_dir, "results", "pifuhd_final", "recon", "front.obj")
            side_obj_path = os.path.join(temp_dir, "results", "pifuhd_final", "recon", "side.obj")
            
            # Fallback paths if naming differs
            if not (os.path.exists(front_obj_path) and os.path.exists(side_obj_path)):
                results_dir = os.path.join(temp_dir, "results", "pifuhd_final", "recon")
                if os.path.exists(results_dir):
                    files = os.listdir(results_dir)
                    obj_files = [f for f in files if f.endswith('.obj')]
                    if len(obj_files) >= 2:
                        front_obj_path = os.path.join(results_dir, obj_files[0])
                        side_obj_path = os.path.join(results_dir, obj_files[1])
            
            if not (os.path.exists(front_obj_path) and os.path.exists(side_obj_path)):
                raise HTTPException(
                    status_code=500, 
                    detail="Mesh generation failed. Check that images show full body clearly."
                )
            
            # Load meshes
            front_mesh = trimesh.load(front_obj_path)
            side_mesh = trimesh.load(side_obj_path)
            
            # Scale meshes to realistic proportions
            reference_height_m = 1.70
            for mesh in [front_mesh, side_mesh]:
                y_min, y_max = mesh.bounds[:,1]
                mesh_height = y_max - y_min
                if mesh_height > 0:
                    scale_factor = reference_height_m / mesh_height
                    mesh.apply_scale(scale_factor)
            
            # Calculate measurements
            shoulder_width, shoulder_y = measure_shoulder(front_mesh)
            waist_width, waist_y, waist_circ = measure_waist(front_mesh)
            hip_width, hip_depth, hip_circ, hip_y = measure_hip(front_mesh, side_mesh)
            bust_width, bust_depth, bust_circ, bust_y = measure_bust(front_mesh)
            neck_width, neck_depth, neck_circ, neck_y = measure_neck(front_mesh)
            arm_length = measure_arm_length(front_mesh, side_mesh, shoulder_y)
            inseam = measure_inseam(front_mesh)
            
            # Format results
            measurements = {
                "shoulder_width_cm": m_to_cm(shoulder_width),
                "waist_circumference_cm": m_to_cm(waist_circ),
                "hip_circumference_cm": m_to_cm(hip_circ),
                "bust_circumference_cm": m_to_cm(bust_circ),
                "neck_circumference_cm": m_to_cm(neck_circ),
                "arm_length_cm": m_to_cm(arm_length),
                "inseam_cm": m_to_cm(inseam),
                "shoulder_width_inches": round(m_to_cm(shoulder_width) / 2.54, 2),
                "waist_circumference_inches": round(m_to_cm(waist_circ) / 2.54, 2),
                "hip_circumference_inches": round(m_to_cm(hip_circ) / 2.54, 2),
                "bust_circumference_inches": round(m_to_cm(bust_circ) / 2.54, 2),
                "neck_circumference_inches": round(m_to_cm(neck_circ) / 2.54, 2),
                "arm_length_inches": round(m_to_cm(arm_length) / 2.54, 2),
                "inseam_inches": round(m_to_cm(inseam) / 2.54, 2)
            }
            
            return JSONResponse(content={
                "success": True,
                "measurements": measurements,
                "message": "Body measurements calculated successfully"
            })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "message": "An error occurred while processing the images"
            }
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
