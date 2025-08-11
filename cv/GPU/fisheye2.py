import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import time

class PyTorchFisheyeCorrector:
    """
    Performs fisheye correction on the GPU using PyTorch.
    """
    def __init__(self, lens_mm=2.8, device=None):
        self.lens_mm = lens_mm
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.K = None
        self.D = None
        self.grid = None
        self.load_parameters()

    def load_parameters(self):
        param_filename = f'camera_params_{self.lens_mm}mm.npz'
        if os.path.exists(param_filename):
            params = np.load(param_filename)
            self.K = params['K']
            self.D = params['D']
            print(f"✅ Loaded fisheye parameters from {param_filename}")
        else:
            print(f"⚠️ No saved parameters found. Using defaults.")
            focal_length = 1200
            self.K = np.array([[focal_length, 0, 1920], [0, focal_length, 1080], [0, 0, 1]], dtype=np.float32)
            self.D = np.array([0.15, -0.05, 0.01, 0.0], dtype=np.float32)

    def _compute_remap_grid(self, width, height):
        if self.K is None or self.D is None:
            return False
        
        print("Computing sampling grid for PyTorch...")
        scale_x, scale_y = width / 3840.0, height / 2160.0
        adjusted_K = self.K.copy()
        adjusted_K[0, 0] *= scale_x
        adjusted_K[1, 1] *= scale_y
        adjusted_K[0, 2] *= scale_x
        adjusted_K[1, 2] *= scale_y

        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            adjusted_K, self.D, (width, height), np.eye(3), balance=0.4)
        
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            adjusted_K, self.D, np.eye(3), new_K, (width, height), cv2.CV_32FC1)

        grid_x = torch.from_numpy(map1) / ((width - 1) / 2) - 1
        grid_y = torch.from_numpy(map2) / ((height - 1) / 2) - 1
        
        self.grid = torch.stack((grid_x, grid_y), dim=2).unsqueeze(0).to(self.device)
        print(f"✅ Grid computed and moved to {self.device}")
        return True

    def correct_gpu(self, frame_tensor_batch):
        if frame_tensor_batch is None:
            return None

        N, C, H, W = frame_tensor_batch.shape
        
        if self.grid is None or self.grid.shape[2] != H or self.grid.shape[1] != W:
            if not self._compute_remap_grid(W, H):
                 print("⚠️ Correction disabled - returning original frame.")
                 return frame_tensor_batch

        grid_batch = self.grid.repeat(N, 1, 1, 1)
        corrected_batch = F.grid_sample(frame_tensor_batch, grid_batch, mode='bilinear', align_corners=True)
        return corrected_batch

# --- Main Test Execution ---
if __name__ == "__main__":
    image_path = 'test.jpg'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corrector = PyTorchFisheyeCorrector(lens_mm=2.8, device=device)

    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"❌ Error: Could not load image from '{image_path}'")
        exit()

    # Convert from BGR (OpenCV) to RGB, then to a PyTorch tensor
    rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1)
    image_tensor = image_tensor.unsqueeze(0).float().to(device) / 255.0

    print(f"\nImage tensor created with shape: {image_tensor.shape} on {image_tensor.device}")

    print("\nApplying GPU correction...")
    start_time = time.time()
    corrected_tensor = corrector.correct_gpu(image_tensor)
    end_time = time.time()
    print(f"Correction applied in { (end_time - start_time) * 1000:.2f} ms")

    # Convert back to NumPy array in RGB format
    corrected_rgb_np = corrected_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()

    # ** THE FIX IS HERE **
    # Convert from floating point RGB back to 8-bit BGR for OpenCV
    corrected_image_bgr = cv2.cvtColor((corrected_rgb_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    output_filename = 'corrected_output.jpg'
    comparison_filename = 'comparison_output.jpg'
    
    cv2.imwrite(output_filename, corrected_image_bgr)
    print(f"✅ Corrected image saved to '{output_filename}'")

    # Create and save the side-by-side comparison image
    h, w = original_image.shape[:2]
    display_width = 1280
    if w > display_width:
        scale = display_width / w
        h_new, w_new = int(h * scale), int(w * scale)
        original_display = cv2.resize(original_image, (w_new, h_new))
        corrected_display = cv2.resize(corrected_image_bgr, (w_new, h_new))
    else:
        original_display = original_image
        corrected_display = corrected_image_bgr

    comparison_image = np.hstack((original_display, corrected_display))
    cv2.putText(comparison_image, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(comparison_image, 'Corrected (PyTorch GPU)', (original_display.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imwrite(comparison_filename, comparison_image)
    print(f"✅ Comparison image saved to '{comparison_filename}'")

    cv2.imshow('Fisheye Correction Test', comparison_image)
    print("\nShowing comparison window. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()