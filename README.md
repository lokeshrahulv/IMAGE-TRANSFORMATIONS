# IMAGE-TRANSFORMATIONS


## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:
Import numpy module as np and pandas as pd.

### Step2:
Assign the values to variables in the program.

### Step3:
Get the values from the user appropriately.

### Step4:
Continue the program by implementing the codes of required topics.

### Step5:
Thus the program is executed in google colab.

## Program:
```python
Developed By: LOKESH RAHUL V V
Register Number:212222100024
```
## i)Image Translation
```python
!pip install opencv-python-headless

import cv2
import numpy as np
from matplotlib import pyplot as plt

def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

image_url = 'dog.jpeg'  
image = cv2.imread(image_url)

tx = 50 
ty = 30  
translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])  
translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))

print("Original Image:")
show_image(image)
print("Translated Image:")
show_image(translated_image)
```
## ii) Image Scaling
```python
!pip install opencv-python-headless

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to display images in Colab
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Load an image from URL or file path
image_url = 'nature1.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Define scale factors
scale_x = 1.5  # Scaling factor along x-axis
scale_y = 1.5  # Scaling factor along y-axis

# Apply scaling to the image
scaled_image = cv2.resize(image, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

# Display original and scaled images
print("Original Image:")
show_image(image)
print("Scaled Image:")
show_image(scaled_image)
```
## iii)Image shearing
```python
!pip install opencv-python-headless

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to display images in Colab
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Load an image from URL or file path
image_url = 'nature2.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Define shear parameters
shear_factor_x = 0.5  # Shear factor along x-axis
shear_factor_y = 0.2  # Shear factor along y-axis

# Define shear matrix
shear_matrix = np.float32([[1, shear_factor_x, 0], [shear_factor_y, 1, 0]])

# Apply shear to the image
sheared_image = cv2.warpAffine(image, shear_matrix, (image.shape[1], image.shape[0]))

# Display original and sheared images
print("Original Image:")
show_image(image)
print("Sheared Image:")
show_image(sheared_image)
```
## iv)Image Reflection
```python
!pip install opencv-python-headless

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to display images in Colab
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Load an image from URL or file path
image_url = 'nature3.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Reflect the image horizontally
reflected_image_horizontal = cv2.flip(image, 1)

# Reflect the image vertically
reflected_image_vertical = cv2.flip(image, 0)

# Reflect the image both horizontally and vertically
reflected_image_both = cv2.flip(image, -1)

# Display original and reflected images
print("Original Image:")
show_image(image)
print("Reflected Horizontally:")
show_image(reflected_image_horizontal)
print("Reflected Vertically:")
show_image(reflected_image_vertical)
print("Reflected Both:")
show_image(reflected_image_both)
```
## v)Image Rotation
```python
!pip install opencv-python-headless

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to display images in Colab
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Load an image from URL or file path
image_url = 'nature4.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Define rotation angle in degrees
angle = 45

# Get image height and width
height, width = image.shape[:2]

# Calculate rotation matrix
rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

# Perform image rotation
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

# Display original and rotated images
print("Original Image:")
show_image(image)
print("Rotated Image:")
show_image(rotated_image)
```
## vi)Image Cropping
```python
!pip install opencv-python-headless

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Function to display images in Colab
def show_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Load an image from URL or file path
image_url = 'nature5.jpg'  # Replace with your image URL or file path
image = cv2.imread(image_url)

# Define cropping coordinates (x, y, width, height)
x = 100  # Starting x-coordinate
y = 50   # Starting y-coordinate
width = 200  # Width of the cropped region
height = 150  # Height of the cropped region

# Perform image cropping
cropped_image = image[y:y+height, x:x+width]

# Display original and cropped images
print("Original Image:")
show_image(image)
print("Cropped Image:")
show_image(cropped_image)
```
## Output:
### i)Image Translation
![Screenshot 2024-03-08 142459](https://github.com/lokeshrahulv/IMAGE-TRANSFORMATIONS/assets/118423842/840dc438-c2d4-4233-bd83-392de541298d)
![Screenshot 2024-03-08 142620](https://github.com/lokeshrahulv/IMAGE-TRANSFORMATIONS/assets/118423842/c5e8cd04-0cba-4b24-9493-e1888bbe0ce8)

### ii) Image Scaling
![Screenshot 2024-03-08 142705](https://github.com/lokeshrahulv/IMAGE-TRANSFORMATIONS/assets/118423842/49529a94-396f-4ea3-bbe0-51638aa2dec8)
![Screenshot 2024-03-08 142737](https://github.com/lokeshrahulv/IMAGE-TRANSFORMATIONS/assets/118423842/9d8e1f9f-c581-4982-8fe4-19fb2acc4d07)

### iii)Image shearing
![Screenshot 2024-03-08 142807](https://github.com/lokeshrahulv/IMAGE-TRANSFORMATIONS/assets/118423842/d9b8ba9f-05b7-400a-a567-7a0bab784f47)
![Screenshot 2024-03-08 142830](https://github.com/lokeshrahulv/IMAGE-TRANSFORMATIONS/assets/118423842/b7214150-35c2-4d31-913a-1cac3c4fee8a)

### iv)Image Reflection
![Screenshot 2024-03-08 142910](https://github.com/lokeshrahulv/IMAGE-TRANSFORMATIONS/assets/118423842/d26a555d-0c64-46fe-94c4-0b34ac689f64)
![Screenshot 2024-03-08 143040](https://github.com/lokeshrahulv/IMAGE-TRANSFORMATIONS/assets/118423842/03c16575-fdb7-428d-9d1f-28258a0a5384)
![Screenshot 2024-03-08 143100](https://github.com/lokeshrahulv/IMAGE-TRANSFORMATIONS/assets/118423842/58dffbff-773d-413e-a136-f64fbdb5e4fe)

### v)Image Rotation
![Screenshot 2024-03-08 143140](https://github.com/lokeshrahulv/IMAGE-TRANSFORMATIONS/assets/118423842/f716a34a-1bdd-43f2-a63c-7bdd7a66eda1)
![Screenshot 2024-03-08 143205](https://github.com/lokeshrahulv/IMAGE-TRANSFORMATIONS/assets/118423842/f8df12f4-2319-4f1b-8cd3-3ce4960f5b70)


### vi)Image Cropping
![image](https://github.com/lokeshrahulv/IMAGE-TRANSFORMATIONS/assets/118423842/77f9400c-bc10-4ae3-9ae8-8b14427b1096)
![image](https://github.com/lokeshrahulv/IMAGE-TRANSFORMATIONS/assets/118423842/0f9aed49-ac3c-4252-8fa4-dee7aa88ee48)

## Result: 
Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
