#This code is customized and updated by Sher Asghar.
#This code can remove the background and crop the image to passport size (35mm x 45mm standard, 413x531 pixels) around the detected face.
#The code uses the rembg library for background removal and OpenCV for face detection.
#The processed images are saved in the output folder with the same file names.
#The input folder should contain images in common formats like PNG, JPG, JPEG, or BMP.
#The output folder will be created if it does not exist.
#The code also handles cases where no face is detected in the image.
#You can adjust the crop margin and other parameters as needed for your specific use case.
#Make sure to install the required libraries using pip install rembg opencv-python pillow numpy.
#You may need to adjust the input and output folder paths in the code to match your file system.
#The code is written in Python and can be run using a Python interpreter or a Jupyter notebook.
import os
import io
import numpy as np  # Import numpy
from rembg import remove
from PIL import Image
import cv2

def process_images(input_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load OpenCV's pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            try:
                # Remove background using rembg
                with open(input_path, 'rb') as file:
                    img_data = remove(file.read())

                # Convert to PIL Image for further processing
                img = Image.open(io.BytesIO(img_data)).convert("RGBA")

                # Convert to OpenCV format for face detection
                opencv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGR)

                # Detect faces in the image
                gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)

                if len(faces) == 0:
                    print(f"No face detected in {filename}, skipping.")
                    continue

                # Get the first detected face
                (x, y, w, h) = faces[0]

                # Calculate a safe crop area around the face
                crop_margin = 30  # Adjust margin if needed
                x1 = max(x - crop_margin, 0)
                y1 = max(y - crop_margin, 0)
                x2 = min(x + w + crop_margin, opencv_img.shape[1])
                y2 = min(y + h + crop_margin, opencv_img.shape[0])

                # Crop the image around the face
                cropped_img = img.crop((x1, y1, x2, y2))

                # Resize and crop to passport size (35mm x 45mm standard, 413x531 pixels)
                cropped_img = cropped_img.resize((413, 531), Image.Resampling.LANCZOS)

                # Convert to RGB before saving as JPEG
                cropped_img = cropped_img.convert("RGB")

                # Save the processed image
                cropped_img.save(output_path)
                print(f"Processed and saved: {output_path}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    input_folder = r"C:\Users\SherAsghar\Documents\GitHub\Image-Background-Remover\inputimages"
    output_folder = r"C:\Users\SherAsghar\Documents\GitHub\Image-Background-Remover\outputimages"
    process_images(input_folder, output_folder)
#Customized and updated by Sher Asghar
#Linkedin: https://www.linkedin.com/in/sher-asghar-3b6b3b1b/