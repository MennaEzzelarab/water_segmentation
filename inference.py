import torch
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import os

MEAN = [ 395.3473,  493.9571,  815.2462,  955.2073, 2068.3069, 1949.9069,
        1333.5028,  103.6337,  114.8657,  327.6829,   34.9512,    9.7407]
STD = [ 283.5301,  334.6441,  420.5426,  578.0315, 1055.7791, 1171.2211,
     925.1953,   49.8946, 1558.2714,  494.9977,   20.1714,   27.8250]

def load_model(weights_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize UNet model (matching the requested best model)
    model = smp.Unet(
        encoder_name="resnet34", 
        encoder_weights=None, # Don't need imagenet weights since we load our own
        in_channels=12, 
        classes=1
    )
    
    # Load state dict
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        print(f"WARNING: Model weights not found at {weights_path}. Running with untrained weights!")
        
    model.to(device).float()
    model.eval()
    return model, device

def run_prediction(model, device, tif_path, output_png_path):
    # Load and normalize the 12-channel tif
    with rasterio.open(tif_path) as src:
        image = src.read().astype(np.float32)

    # Normalize image
    mean_arr = np.array(MEAN).reshape(-1, 1, 1)
    std_arr = np.array(STD).reshape(-1, 1, 1)
    img_normalized = (image - mean_arr) / (std_arr + 1e-8)
    
    # Convert to tensor and add batch dimension
    img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = torch.sigmoid(model(img_tensor.float()).squeeze())
        pred_mask = (output > 0.5).cpu().numpy()

    # Reconstruct RGB for visualization (channels 3, 2, 1 usually correspond to indices 3, 2, 1 in the 12-band tif)
    # The original code uses rgb = img_tensor[[3,2,1],:,:]
    # But for visualization we want the original unnormalized image or denormalized to [0,1]
    rgb = image[[3, 2, 1], :, :]
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    rgb = np.transpose(rgb, (1, 2, 0)) # CHW to HWC

    # Plot and save
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    axs[0].imshow(rgb)
    axs[0].set_title("Input (RGB)")
    axs[0].axis('off')
    
    axs[1].imshow(pred_mask, cmap='Blues')
    axs[1].set_title("Predicted Water Mask")
    axs[1].axis('off')
    
    # Ensure output exists
    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_png_path, bbox_inches='tight', transparent=True)
    plt.close(fig)

    # Calculate actual surface area of the water mask in square meters
    # A generic GeoTIFF transform gives the physical width of one pixel in src.transform[0]
    # and the physical height of one pixel in abs(src.transform[4])
    try:
        pixel_width = src.transform[0]
        pixel_height = abs(src.transform[4])
        # Count the number of pixels predicted as water
        water_pixel_count = np.sum(pred_mask)
        # Area in square meters
        water_area_sqm = water_pixel_count * (pixel_width * pixel_height)
    except Exception as e:
        print(f"Failed to calculate area from image metadata: {e}")
        water_area_sqm = 0

    return water_area_sqm

