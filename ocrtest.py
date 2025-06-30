# from paddleocr import PaddleOCR
# from PIL import Image, ImageDraw, ImageFont
# Image.MAX_IMAGE_PIXELS = None

# # Initialize OCR engine
# ocr = PaddleOCR(use_angle_cls=True, lang="cs")

# img_path = './dataset/raster_val.jpg'
# slice = {'horizontal_stride': 300, 'vertical_stride': 300, 'merge_x_thres': 50, 'merge_y_thres': 50}
# results = ocr.ocr(img_path, cls=True, slice=slice)

# # Load image
# image = Image.open(img_path).convert("RGB")
# draw = ImageDraw.Draw(image)
# font = ImageFont.truetype('../paddleocr/doc/fonts/simfang.ttf', size=20)  # Adjust size as needed

# # Process and draw results
# for res in results:
#     for line in res:
#         box = [tuple(point) for point in line[0]]
#         # Finding the bounding box
#         box = [(min(point[0] for point in box), min(point[1] for point in box)),
#                (max(point[0] for point in box), max(point[1] for point in box))]
#         txt = line[1][0]
#         draw.rectangle(box, outline="red", width=2)  # Draw rectangle
#         draw.text((box[0][0], box[0][1] - 25), txt, fill="blue", font=font)  # Draw text above the box

# # Save result
# image.save("result.jpg")

from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
import re

Image.MAX_IMAGE_PIXELS = None

# Create digit-only dictionary
with open('digits.txt', 'w') as f:
    f.write('0\n1\n2\n3\n4\n5\n6\n7\n8\n9\n.\n')

# Initialize OCR engine with digit-specific settings
ocr = PaddleOCR(
    use_angle_cls=True, 
    lang="en",  # Switch to English for better digit recognition
    rec_char_dict_path='digits.txt',  # Use digits-only dictionary
    rec_char_type='digit',  # Specify digit recognition
    det_db_box_thresh=0.6,  # Higher confidence threshold
    det_limit_side_len=960  # Limit detection size for better accuracy
)

img_path = './dataset/raster_val.jpg'
slice = {'horizontal_stride': 300, 'vertical_stride': 300, 'merge_x_thres': 50, 'merge_y_thres': 50}
results = ocr.ocr(img_path, cls=True, slice=slice)

# Load image
image = Image.open(img_path).convert("RGB")
draw = ImageDraw.Draw(image)
font = ImageFont.truetype('../paddleocr/doc/fonts/simfang.ttf', size=20)

# Process and draw results - only for numeric detections
for res in results:
    for line in res:
        txt = line[1][0]
        confidence = float(line[1][1])
        
        # Only process if text is numeric and confidence is high
        if re.match(r'^-?\d*\.?\d+$', txt) and confidence > 0.8:
            box = [tuple(point) for point in line[0]]
            box = [(min(point[0] for point in box), min(point[1] for point in box)),
                   (max(point[0] for point in box), max(point[1] for point in box))]
            
            # Draw detection
            draw.rectangle(box, outline="red", width=2)
            # Add confidence score to display
            label = f"{txt} ({confidence:.2f})"
            draw.text((box[0][0], box[0][1] - 25), label, fill="blue", font=font)

# Save result
image.save("result_numbers_only.jpg")