import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F

def overlay_heatmap_on_image(img_tensor, heatmap_tensor):
    """
    img_tensor: (3, H, W) torch.Tensor
    heatmap_tensor: (h, w) or (1, h, w) or (1, 1, h, w)
    Returns: numpy RGB image with overlay
    """
    if heatmap_tensor.ndim == 2:
        heatmap_tensor = heatmap_tensor.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    elif heatmap_tensor.ndim == 3:
        heatmap_tensor = heatmap_tensor.unsqueeze(0)  # (1,1,H,W)

    # Resize heatmap to match input image
    heatmap_resized = F.interpolate(
        heatmap_tensor, size=img_tensor.shape[1:], mode='bilinear', align_corners=False
    ).squeeze().cpu().numpy()

    heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-6)
    heatmap_colored = plt.cm.jet(heatmap_resized)[..., :3]  # RGB only

    base_img = to_pil_image(img_tensor)
    base_arr = np.array(base_img) / 255.0

    overlay = (0.5 * base_arr + 0.5 * heatmap_colored)
    return np.clip(overlay, 0, 1)


class GradCAMWrapper:
    def __init__(self, target_branch):
        """
        target_branch: Pass either model.face_branch or model.body_branch 
        so GradCAM knows which CNN to attach to!
        """
        self.branch = target_branch
        self.gradients = None
        self.activations = None

        if self.branch.backbone_type == 'resnet18':
            self.target_layer = self.branch.cnn[-2]  # Layer 4
        elif self.branch.backbone_type == 'custom':
            for m in reversed(self.branch.cnn.features):
                if isinstance(m, torch.nn.Conv2d):
                    self.target_layer = m
                    break
        else:
            raise ValueError("Unsupported CNN backbone for GradCAM.")
                    
        print(f"[GradCAM] Target layer set to: {self.target_layer}")
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
            # print("[GradCAM] forward_hook triggered:", output.shape)

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
            # print("[GradCAM] backward_hook triggered:", grad_out[0].shape)

        self.target_layer.register_forward_hook(forward_hook)
        try:
            self.target_layer.register_full_backward_hook(backward_hook)
        except AttributeError:
            self.target_layer.register_backward_hook(backward_hook)

    def generate_sequence(self, x_seq, lengths, model_forward_kwargs, class_idx=None):
        """
        Adapted to fit the new multimodal model structure.
        model_forward_kwargs: A dictionary containing the inputs for the full model forward pass.
                              e.g. {'face_seq': ..., 'body_seq': ..., 'lm_seq': ...}
        """
        self.activations = None
        self.gradients = None

        # Determine the batch and time dimensions from the target sequence
        B, T, C, H, W = x_seq.shape
        
        with torch.enable_grad():
            x_seq.requires_grad = True
            
            # Pass everything through the full model
            output = self.branch.parent_model(lengths=lengths, **model_forward_kwargs)
            
            scores = output.sum(dim=1) if class_idx is None else output[:, class_idx]
            scores.sum().backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("[GradCAM] Hooks not triggered. Ensure forward + backward were called.")

        grads = self.gradients                                # (B*T, C, h, w)
        acts = self.activations                               # (B*T, C, h, w)
        weights = grads.mean(dim=(2, 3), keepdim=True)        # (B*T, C, 1, 1)
        cam = (weights * acts).sum(dim=1).clamp(min=0)        # (B*T, h, w)

        # Normalize per-frame
        cam = cam.view(-1, cam.shape[-2], cam.shape[-1])        # (B*T, H, W)
        cam_min = cam.view(cam.size(0), -1).min(dim=1, keepdim=True)[0].unsqueeze(-1)
        cam_max = cam.view(cam.size(0), -1).max(dim=1, keepdim=True)[0].unsqueeze(-1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-6)  # (B*T, H, W)

        return cam.view(B, T, cam.shape[-2], cam.shape[-1])  # (B, T, H, W)