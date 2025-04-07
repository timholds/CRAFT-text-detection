from typing import List, Tuple, Optional
import os
import time
import torch
from torch.autograd import Variable
from PIL import Image
import numpy as np
import cv2
from huggingface_hub import hf_hub_url, hf_hub_download

from CRAFT.craft import CRAFT, init_CRAFT_model
from CRAFT.refinenet import RefineNet, init_refiner_model
from CRAFT.craft_utils import adjustResultCoordinates, getDetBoxes
from CRAFT.imgproc import resize_aspect_ratio, normalizeMeanVariance


HF_MODELS = {
    'craft': dict(
        repo_id='boomb0om/CRAFT-text-detector',
        filename='craft_mlt_25k.pth',
    ),
    'refiner': dict(
        repo_id='boomb0om/CRAFT-text-detector',
        filename='craft_refiner_CTW1500.pth',
    )
}

    
def preprocess_image(image: np.ndarray, canvas_size: int, mag_ratio: bool):
    # resize
    img_resized, target_ratio, size_heatmap = resize_aspect_ratio(
        image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio
    )
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    return x, ratio_w, ratio_h


class CRAFTModel:
    
    def __init__(
        self, 
        cache_dir: str,
        device: torch.device,
        local_files_only: bool = False,
        use_refiner: bool = True,
        fp16: bool = True,
        canvas_size: int = 1280,
        mag_ratio: float = 1.5,
        text_threshold: float = 0.7,
        link_threshold: float = 0.4,
        low_text: float = 0.4
    ):
        self.cache_dir = cache_dir
        self.use_refiner = use_refiner
        self.device = device
        self.fp16 = fp16
        
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.low_text = low_text
        
        # loading models
        paths = {}
        for model_name in ['craft', 'refiner']:
            config = HF_MODELS[model_name]
            paths[model_name] = os.path.join(cache_dir, config['filename'])
            if not local_files_only:
                paths[model_name] = hf_hub_download(
                    repo_id=config['repo_id'],
                    filename=config['filename'],
                    cache_dir=cache_dir
                )
            
        self.net = init_CRAFT_model(paths['craft'], device, fp16=fp16)
        if self.use_refiner:
            self.refiner = init_refiner_model(paths['refiner'], device)
        else:
            self.refiner = None
        
    def get_text_map(self, x: torch.Tensor, ratio_w: int, ratio_h: int) -> Tuple[np.ndarray, np.ndarray]:
        x = x.to(self.device)

        # forward pass
        with torch.no_grad():
            y, feature = self.net(x)

        # make score and link map
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()

        # refine link
        if self.refiner:
            with torch.no_grad():
                y_refiner = self.refiner(y, feature)
            score_link = y_refiner[0,:,:,0].cpu().data.numpy()
            
        return score_text, score_link

    def get_batch_polygons(self, batch_images: torch.Tensor, ratios_w: torch.Tensor, ratios_h: torch.Tensor):
        """Batch process pre-normalized images on GPU"""
        # Forward pass
        with torch.no_grad():
            y, feature = self.net(batch_images.to(self.device)) 
            if self.refiner:
                y_refiner = self.refiner(y, feature)
                link_scores = y_refiner[..., 0]  # [B, H, W]
            else:
                link_scores = y[..., 1]  # [B, H, W]
            
            text_scores = y[..., 0]  # [B, H, W]

        batch_size = batch_images.size(0)
        # Process each image in the batch (minimize CPU transfers)
        batch_polys = []
        for b_idx in range(batch_size):
            # Extract scores for this image
            text_score = text_scores[b_idx].cpu().numpy()
            link_score = link_scores[b_idx].cpu().numpy()
            
            # Get current ratios
            curr_ratio_w = ratios_w[b_idx].item() if isinstance(ratios_w, torch.Tensor) else ratios_w
            curr_ratio_h = ratios_h[b_idx].item() if isinstance(ratios_h, torch.Tensor) else ratios_h
            
            # Use existing OpenCV-based post-processing
            boxes, polys = getDetBoxes(
                text_score, link_score,
                self.text_threshold, self.link_threshold,
                self.low_text, False  # Don't need detailed polygons, just boxes
            )
            
            # Adjust coordinates
            boxes = adjustResultCoordinates(boxes, curr_ratio_w, curr_ratio_h)
            
            # Convert to tensor and add to batch
            image_polys = []
            if len(boxes) > 0:
                # Ensure boxes is in a list format before processing
                boxes = boxes.tolist() if isinstance(boxes, np.ndarray) else boxes
                for box in boxes:
                    # Convert to tensor (4 corner points)
                    box_tensor = torch.tensor(box, dtype=torch.float32, device=self.device)
                    image_polys.append(box_tensor)
                    
            batch_polys.append(image_polys)

        return batch_polys
    
    def _convex_hull(self, x_coords, y_coords):
        """Simple convex hull approximation for GPU tensors"""
        # For character detection, a simple bounding box is often sufficient
        min_x = torch.min(x_coords)
        max_x = torch.max(x_coords)
        min_y = torch.min(y_coords)
        max_y = torch.max(y_coords)

        # Create rectangle corners
        pts = torch.tensor([
            [min_x, min_y],
            [max_x, min_y],
            [max_x, max_y],
            [min_x, max_y]
        ], device=x_coords.device)

        return pts

    def get_polygons(self, image: Image.Image) -> List[List[List[int]]]:
        x, ratio_w, ratio_h = preprocess_image(np.array(image), self.canvas_size, self.mag_ratio)
        
        score_text, score_link = self.get_text_map(x, ratio_w, ratio_h)
        
        # Post-processing
        boxes, polys = getDetBoxes(
            score_text, score_link, 
            self.text_threshold, self.link_threshold, 
            self.low_text, True
        )
        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: 
                polys[k] = boxes[k]
            else:
                polys[k] = adjustResultCoordinates(polys[k], ratio_w, ratio_h)

        res = []
        for poly in polys:
            res.append(poly.astype(np.int32).tolist())
        return res
    
    def _get_boxes_preproc(self, x, ratio_w, ratio_h) -> List[List[List[int]]]:
        score_text, score_link = self.get_text_map(x, ratio_w, ratio_h)
        
        # Post-processing
        boxes, polys = getDetBoxes(
            score_text, score_link, 
            self.text_threshold, self.link_threshold, 
            self.low_text, False
        )
        
        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
        boxes_final = []
        if len(boxes)>0:
            boxes = boxes.astype(np.int32).tolist()
            for box in boxes:
                boxes_final.append([box[0], box[2]])

        return boxes_final
    
    def get_boxes(self, image: Image.Image) -> List[List[List[int]]]:
        x, ratio_w, ratio_h = preprocess_image(np.array(image), self.canvas_size, self.mag_ratio)
        
        boxes_final = self._get_boxes_preproc(x, ratio_w, ratio_h)
        return boxes_final
