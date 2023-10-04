import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        input_point = np.array([[x, y]])
        input_label = np.array([1])  # 1 indicates it's a foreground point

        # Predict masks
        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        # Display the results
        for i, (mask, score) in enumerate(zip(masks, scores)):
            overlay = image.copy()
            mask_color = cv2.merge([mask.astype('uint8')] * 3)  # Convert to 3-channel
            mask_color *= 255  # Scale the mask
            cv2.addWeighted(mask_color, 0.6, overlay, 1 - 0.6, 0, overlay)
            cv2.imshow(f"Mask {i+1}, Score: {score:.3f}", overlay)


# Load your image
image_path = "/home/jc/Pictures/frame_20231003_214200.png"  # Replace with your image path
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize SAM model
sam_checkpoint = "/home/jc/Downloads/sam_vit_h_4b8939.pth"  # Replace with your checkpoint path
model_type = "default"  # or "vit_h" based on your downloaded model
device = "cpu"  # Change to "cuda" if you want to use GPU

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

# Process the image to produce an image embedding
predictor.set_image(image)

cv2.imshow('image', image)
cv2.setMouseCallback('image', click_event)

while True:
    cv2.imshow('image', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
