import streamlit as st #for web app UI
# Import the Streamlit drawable canvas component for drawing on the web page
from streamlit_drawable_canvas import st_canvas
# Import OpenCV for image processing tasks
import cv2
# Import Keras function to load a pre-trained model
from keras.models import load_model
import numpy as np
import warnings

warnings.filterwarnings('ignore') #filter warnings and ignore them

disgitsrecg=[] #stores recognized digists precdicted by model
res=""
#same as above but as string
#UDF for digit prediction
def predict():

    global res #because this has to be modified
    #load our pre trained model
    print("LOADING THE MODEL")
    model=load_model('mnist.multidigit_recog.h5')

    # Define path to the image saved from the canvas

    image_folder = "./" #saving image in the same directory as pwd and as app.py
    # means the image is in the current working directory i.e. folder where streamlit run app.py

    filename = f'img.jpg' #load user's image drawn on canvas
    print("Reading the image")
    image = cv2.imread(image_folder + filename, cv2.IMREAD_COLOR) #reads the user input image in colour format
    print("******converting digit's image to gray scale now*********")
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    print("Unblurring usimg gaussian gray")
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) #inputs image, kernel size (should be odd) and by default Std deviation, these will smoothen the image
    #default border type

    # Apply adaptive thresholding to convert the grayscale image to binary (black & white)

    # Adaptive thresholding is better than global thresholding in varying lighting conditions

    th = cv2.adaptiveThreshold(

        blurred,                   # Source image

        255,                          # Maximum value to use with THRESH_BINARY

        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, # Use a weighted sum of neighborhood values

        cv2.THRESH_BINARY_INV,        # Invert the output (digits become white)

        15, 8                             # Block size and constant C

    )

    # Find contours (continuous lines or curves that bound the white regions)
    print("Finding Contours")
    # In a binary image (black and white), contours
    #are the boundaries of the white regions ( like digits) against the black background.
    contours = cv2.findContours(

        th,                                                           # Binary image

        cv2.RETR_EXTERNAL,                 # Only retrieve external contours

        cv2.CHAIN_APPROX_SIMPLE  # Compress horizontal, vertical, and diagonal segments, means only removing redundant points (e.g., for a rectangle, only the four corners are stored instead of every pixel along the edge), saving memory and computation

    )[0]

    print("""Why Find Contours?
          Object Detection: Contours help you locate and isolate objects (like handwritten digits) in an image.

          Shape Analysis: You can analyze the shape, size, or position of objects.

          Segmentation: Contours help segment individual objects for further processing or classification.""")


    # Loop through each contour (likely each digit drawn)

    for cnt in contours:

       #It finds the smallest upright rectangle (aligned with the axes) that fully contains the contour.
       #x, y: Coordinates of the top-left corner of the rectangle.
        #w, h: Width and height of the rectangle.'''
        x, y, w, h = cv2.boundingRect(cnt) 

        #The bounding box gives you a quick and easy way to isolate the region of interest (ROI) in the image that contains the dig
        print("Drawing a blue rectangle so that to isolate ROI")
       # Draw a blue rectangle around each detected digit

        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1) #top left corner, bootom right corner and 255,0,0 is colur blue in BGR
        #and 1 is the tickness of our bounding boc=x(ractanf=gle border)
        #we ae craeting bounding box for visualization and debugging: it helps you see if your digit extraction is working correctly
        
        print("Cropping the digit now")
        digit = th[y:y + h, x:x + w] #extarcting the rectangular region
        #th is numpy array, y:y +h is selects rows, x to x+w selects columns but exculuding x+w
        #Why not just use the whole image? Because our model expects a single digit, and we want to classify or process digits individually

        # Resize the digit to 18x18 pixels (MNIST model expects 28x28)
        print("resizing the cropped image of the digit of 18X18")
        #original in mnist in 20x20 box for the actual digit and then placed in 28x28 image with padding, with resizing we ensure it fits properly

        resized_digit = cv2.resize(digit, (18, 18))

        # Pad the resized digit with 5 pixels of black pixels (zeros) on top, bottom, letft right

        # This results in a 28x28 image as expected by the model

        # (18+5+5 = 28 for both height and width)

        padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)

       #Keras models trained on MNIST expect input arrays of shape (1, 28, 28, 1) for a single image

        digit = padded_digit.reshape(1, 28, 28, 1) #(batch size, h, w, channels)

        # Normalize pixel values from [0, 255] to [0, 1]
        print("normalizing the pixel between 0-1 from 0-255 ")
        digit = digit / 255.0
        #Neural networks train and predict more efficiently when input features are on a similar scale.

        print("running the digit image through the trained model!")

        pred = model.predict(digit)[0]

       # Get the digit with the highest probability (model's final prediction)

        final_pred = np.argmax(pred) 
        #np.argmax(pred) returns the index of the highest value in the pred array
        #The class with the highest probability is the model’s best guess for the digit in the image
        #This is the value I display to the user as the recognized digit.

        # Appending predicted digit to the global list

        disgitsrecg.append(int(final_pred))

        #adding up predicted digit to the result string
        res = res + " " + str(final_pred)

        # Prepare text showing prediction and confidence percentage

        data = str(final_pred) + ' ' + str(int(max(pred) * 100)) + '%'

        # Define font settings for overlaying prediction on the image

        font = cv2.FONT_HERSHEY_SIMPLEX #font style in open cv

        fontScale = 0.5 #text size

        color = (255, 255, 255)  #setting text to White color

        thickness = 1 #line thickness on the text

        # Writes prediction text on the image at the top-left corner of the bounding box

        cv2.putText(image, data, (x, y), font, fontScale, color, thickness)
        #input image, data with confidence %,top let corner

print("********Now building UI Interface*******")


st.title("Drawable Canvas")
st.markdown("""

Draw digits on the canvas, get the image data back into Python!

""")
# Create a canvas where users can draw digits using streamlit

canvas_result = st_canvas(

    stroke_width=10,                   # Thickness of the brush

    stroke_color='red',                # Color of the brush

    height=150                               # Height of the canvas

    )

with st.sidebar:
     st.header("Model Hyperparameters")
     st.markdown("""
             **Model Training Hyperparameters**
             - **Optimizer:** Adam
             - **Batch Size:** 64
             - **Epochs:** 5
             - **Loss Function:** Categorical Crossentropy
             - **Dropout Rate:** 0.2
             - **MaxPooling 2D Pool Size:** (2,2)
             - **Conv 2D Layers:** 3 (32, 64, 64 filters)
             - **Kernel Size:** (3,3)
             - **Dense Units:** 64
             - **Validation Split:** 0.1
             - **Early Stopping Patience:** 5
             - **Image Shape:** (28,28,1)
             - **Normalization:** 0-1
             - **Activation Function:** ReLU
             - **Output Activation:** Softmax
             """)

    # Check if the user has drawn something

if canvas_result.image_data is not None:

        # Save the drawn image to a file for processing

        cv2.imwrite(f"img.jpg",  canvas_result.image_data)

        # Create a "Predict" button

if st.button("Predict"):

        predict()                                                   # Call the predict function

        st.write('The predicted digit:', res)  # Display the result on the app
 
 

 