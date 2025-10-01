import cv2 as cv
import os

# ---- Settings ----
# Change this to your image file path
input_image_path = "data/2/2.1.png"

# Output will be saved with "_bw" suffix in the same directory
# Or you can specify a custom output path:
# output_image_path = "output/my_bw_image.png"

# ---- Convert to Black and White ----
# Read the image
img = cv.imread(input_image_path)

if img is None:
    print(f"[ERROR] Could not read image: {input_image_path}")
    print("Please check the file path is correct.")
else:
    # Convert to grayscale
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Generate output filename
    dir_name = os.path.dirname(input_image_path)
    base_name = os.path.basename(input_image_path)
    name_without_ext, ext = os.path.splitext(base_name)
    output_image_path = os.path.join(dir_name, f"{name_without_ext}_bw{ext}")
    
    # Save the grayscale image
    success = cv.imwrite(output_image_path, gray_img)
    
    if success:
        print(f"[SUCCESS] Black and white image saved to: {output_image_path}")
    else:
        print(f"[ERROR] Failed to save image to: {output_image_path}")