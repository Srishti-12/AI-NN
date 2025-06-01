import streamlit as st # for web app UI
# Import the Streamlit drawable canvas component for drawing on the web page
from streamlit_drawable_canvas import st_canvas
# Import OpenCV for image processing tasks
import cv2
# Import Keras function to load a pre-trained model
from keras.models import load_model
import numpy as np
import warnings
import os # Import os for path handling and cleanup of temporary files

warnings.filterwarnings('ignore') # Filter warnings and ignore them

st.set_page_config(layout="centered", page_title="Handwritten Digit Recognition") #For better position amd layouts

# Using @st.cache_resource ensures that the model is loaded from disk only once
# when the Streamlit app starts, not every time a user interacts with it!
# This dramatically improves the app's performance and responsiveness
@st.cache_resource
def load_my_model():
    st.info("Loading the machine learning model. This happens once per session.")
    try:
        model = load_model('mnist.multidigit_recog.h5')
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}. "
                 f"Please ensure 'mnist.multidigit_recog.h5' is in the same directory and the keras libraries installed properly")
        return None

# Load the model globally when the app.py script runs
model = load_my_model()

# This function handles all the image preprocessing and model prediction
def get_digit_predictions(image_path):
    if model is None:
        return [], None, "Model not loaded. Cannot perform prediction."

    predicted_digits_list = []  # Stores recognized digits (as integers) precdicted by model
    predicted_string_result = "" # Stores recognized digits (as a combined string)

    # --- Debugging: Check if image path exists and can be read ---
    if not os.path.exists(image_path):
        st.error(f"Image file not found: {image_path}. Please try drawing again.")
        return [], None, "Error: Image file not found."

    print("******Reading the image*********")
    image = cv2.imread(image_path, cv2.IMREAD_COLOR) #reading user input in colour format
    if image is None:
        st.error(f"Could not read image from {image_path}. It might be corrupted or not a valid image file.")
        return [], None, "Error: Could not read image."

    #  Original Drawn Image (for visualization)
    print("Original image just after user entered")
    st.subheader("Intermediate Image Processing Steps (for debugging):")
    st.image(image, caption="1. Original Drawn Image (BGR)", channels="BGR", use_container_width=True)
    

    # Converting to grayscale
    print("******converting digit's image to gray scale now*********")
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    st.image(gray, caption="2. Grayscale Image", use_column_width=True)

    # Apply Gaussian blur to smooth the image and reduce noise
    print("Removing blur using GaussianBlur")
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    st.image(blurred, caption="3. Blurred Image", use_column_width=True)

    print("@@@@@@************ADAPTIVE THRESHOLDING STARTED**********@@@@@@@")

    # Applying adaptive thresholding to convert the grayscale image to binary (black & white).
    # This turns the red drawing into white pixels on a black background.
    # The (11, 2) is with what I am experimeting and it's improving the threshold
    # Block size (first number) must be odd. C (second number) is a constant subtracted.
    # Increasing the block size had also improved it's reading image.
    # when noise appears as digits I am experimenting with decreasing block size and/or increasing C.
    th = cv2.adaptiveThreshold(
        src=blurred,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV, # Invert: drawn lines become white, background black
        blockSize=15, # Current block size (must be odd)
        C=5         # Current constant C , increasing or decreasing this has direct impact again started with 2
    )
    print("Displays inverted image in colour, that is BNW version")
    st.image(th, caption=f"Thresholded (Binary Inverted) Image (Block Size: {11}, C: {2})", use_column_width=True)

    print("Contours")

    # Finding contours (outlines of detected digits)
    # cv2.findContours returns a tuple we only need the contours themselves.
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #th is source image, counter retrieve mode(only retrieves the outermost contours),removing redundant points (making it simpler)

    if not contours:
        st.warning("No distinct shapes (digits) found in your drawing after processing.")
        return [], None, "No digits found"

    print("****adding this step as it started performing well for multi digit recognition")

    # This is crucial for multi-digit recognition (e.g., "123" vs "312").
    # It ensures digits are processed and reported in the order they appear horizontally(left to right)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    print("looping through the detected digits in image")

     # Loop through each detected contour (potential digit)
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt) # Get bounding box coordinates and dimensions

        print("CONTOUR FILTERING STARTED:")

        if w < 15 or h < 15: #(w < 15 or h < 15) for more aggressive filtering
            st.info(f"Skipping very small contour at ({x},{y}) with size {w}x{h}.")
            continue # Skip this contour, it's probably noise

        # Draw a green rectangle around each detected digit on the *original* image (for visualization)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) # Green color (BGR), thickness 2
        #this step gives me clarity as to what is getting captured

    # Extracting the digit region of interest (ROI) from the thresholded image
        print("extracting the digit region of interest")
        digit_roi = th[y:y + h, x:x + w]
        st.image(digit_roi, caption=f"{i+1} Cropped Digit ROI (Binary)", use_column_width=True)
        #Displays excat roi

        # Resizing the digit ROI to 18x18 pixels (MNIST model expects 28x28)
        # Using INTER_AREA interpolation for image shrinking so that it fits 28x28 perfectly
        resized_digit = cv2.resize(digit_roi, (18, 18), interpolation=cv2.INTER_AREA)
        st.image(resized_digit, caption=f"{i+1} Resized to 18x18", use_column_width=True)
        #Displays resized image

        # Padding the resized digit with 5 pixels of black pixels (0s) on top, bottom, left, right
        # This results in a 28x28 image (18 + 5 + 5 = 28) as expected by the model in keras
        padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)
        st.image(padded_digit, caption=f"{i+1} Padded to 28x28 (Final image for model)", use_column_width=True)
        #displays padded image

        # Reshaping for the Keras model (batch_size, height, width, channels)
        # For a single grayscale image  (1, 28, 28, 1)
        final_digit_for_model = padded_digit.reshape(1, 28, 28, 1)

        print("***Normalizing****")
        #Neural networks train and predict more efficiently when input features are on a similar scale
        final_digit_for_model = final_digit_for_model / 255.0

        print("running the digit image through the trained model!")
        pred = model.predict(final_digit_for_model, verbose=0)[0]
        #verbose=0 tells Keras not to print any progress bar or 
        #output during the prediction process. It runs silently, with no messages shown in the terminal or console

        # Get the digit with the highest probability (model's final prediction)
        final_pred = np.argmax(pred)

        #np.argmax(pred) returns the index of the highest value in the pred array
        #The class with the highest probability is the model’s best guess for the digit in the image
        #This is the value I display to the user as the recognized digit

        predicted_digits_list.append(int(final_pred)) #adds predicted digitd to a list
        predicted_string_result += str(final_pred) # Concatenate results into a string

        # Prepare text to display prediction and confidence percentage
        confidence = int(max(pred) * 100)
        data_text = f"{final_pred} ({confidence}%)" #digit with comfidence

        # Define font settings for overlaying prediction text on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.7
        color = (0, 255, 0) # Green color for text (BGR)
        thickness = 1

        # Writes prediction text on the image, adjusted to appear above the bounding box
        text_y_position = y - 10 if y > 10 else y + h + 20 # Place above if space, else below
        cv2.putText(image, data_text, (x, text_y_position), font, fontScale, color, thickness)
        #imporover here as well:
        # Writes prediction text on the image at the top-left corner of the bounding box

        #cv2.putText(image, data, (x, y), font, fontScale, color, thickness)
        #input image, data with confidence %,top let corner
        #Goal: Place the prediction text (e.g., "3 98%") above the bounding box if there’s space; otherwise, place it below.

        #y: The y-coordinate of the top-left corner of the bounding box.

        #h: The height of the bounding box
        #f y > 10 (i.e., the bounding box is not too close to the top of the image), the text is placed 10 pixels above the bounding box: y - 10.
        #if y <= 10 (i.e., the bounding box is near the top edge), placing text above would go out of bounds, so the text is placed below the box: y + h + 15.
    
    # Returning the list of predicted digits, the image with boxes/text, and the result string
    return predicted_digits_list, image, predicted_string_result

# --- Streamlit UI Layout (Executed immediately when the app.py starts) ---
print("********Now building UI Interface*******")
#st.set_page_config(layout="centered", page_title="Handwritten Digit Recognition") #For better position amd layouts

st.title("Handwritten Digit Recognition App")
st.markdown("""
Draw one or more digits on the canvas below andf our model will try to recognize them!
""")

# --- IMPROVING: Enhanced Canvas Properties ---
# Added background_color to ensure black background (consistent with MNIST training).
# Increased height and width for more drawing space.
canvas_result = st_canvas(
    stroke_width=10, #brush
    stroke_color='red', # Color of the brush
    background_color='#000000', # Set background to black (important for consistent preprocessing)
    height=200, # Increased height for more drawing area
    width=500,  # Defined width for better control
    drawing_mode="freedraw", #as it's fredrawing with hands
    key="canvas",
    display_toolbar=True # Allows user to clear/undo drawings easily
)

#Here, key="canvas" tells Streamlit:
#“This widget’s state should be tracked under the name 'canvas'.”

print("SIDEBARD FOR HYPERPARAMS")

# Sidebar for model hyperparameters (remains the same)
with st.sidebar:
    st.header("Model Training Hyperparameters")
    st.markdown("""
             **Model Training Hyperparameters**
             - **Optimizer:** RMSprop
             - **Batch Size:** 64
             - **Epochs:** 5
             - **Loss Function:** Categorical_Crossentropy
             - **Dropout Rate:** 0.2
             - **MaxPooling 2D Pool Size:** (2,2)
             - **Conv 2D Layers:** 3 (32, 64, 64 filters)
             - **Kernel Size:** (3,3)
             - **Dense Units:** 64
             - **Validation Split:** 0.1
             - **Image Shape:** (28,28,1)
             - **Normalization:** 0-1
             - **Activation Function:** ReLU
             - **Output Activation:** Softmax
             """)
temp_image_path = None
# --- Handle Canvas Output and Prediction Button ---
if canvas_result.image_data is not None:
    # Convert image data from Streamlit canvas (RGBA) to OpenCV (BGR).
    img_array = np.array(canvas_result.image_data)
    if img_array.shape[-1] == 4: # This checks if the last dimension is 4, meaning the image has 4 channels (Red, Green, Blue, Alpha).
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

# Define a temporary path for the image to be saved and processed.
    temp_image_path = "drawn_digit.png"
    cv2.imwrite(temp_image_path, img_array) # Save the drawn image to disk temporarily
#Using ,png instaed og .jpg here to save every pixel as it's lossless compression  algorithm, don't want to lose any pixel

#Adding this as I was getting name error for predict:
# Initializing variables to None/empty string *before* the button click logic
# NameError when the script first runs or reruns
    processed_image_with_boxes = None
    final_predicted_string = ""
    predicted_digits_list = []
    

# Create a "Predict" button
if st.button("Predict Digits"):
    # Displaying a spinner while prediction is ongoing
    with st.spinner('Analyzing your drawing and predicting digits...'):
    # Call the prediction logic fu  nction
        predicted_digits_list, processed_image_with_boxes, final_predicted_string = get_digit_predictions(temp_image_path)

    #As seen above this def retutns predicted list, image by juser with box, and result with confidence %

    # Display results based on whether digits were found
    if processed_image_with_boxes is not None and final_predicted_string not in ["No digits found", "Error: Image file not found.", "Error: Could not read image."]:
        st.subheader("Recognized Digits:")
        st.success(f"The model predicted: **{final_predicted_string}**")
        # Display the processed image with bounding boxes and predictions drawn on it
        st.image(processed_image_with_boxes, caption="Processed Image with Predictions", channels="BGR", use_container_width=True)
        st.markdown(f"**Individual predictions:** {', '.join(map(str, predicted_digits_list))}") # Show individual list
    else:
            # Handliing cases where no digits were found or an error occurred
        st.warning(final_predicted_string) # Display the specific warning/error message from the function

#   Clean up the temporary image file ---

if temp_image_path is not None and os.path.exists(temp_image_path):
    os.remove(temp_image_path) # removing any residual files after processing it
else:
    # If no image data is available from the canvas (e.g., on initial load)
    # or if temp_image_path was never assigned a real path
    st.info("Draw a digit or multiple digits on the canvas above and click 'Predict Digits'.")



#IMPROVEMTS:
#1. th (Adaptive threshold) created a function and put it inside it
#2 Changed Contour used  (if w < 10 or h < 10: continue)
#3.Enhance canvas config
#4. green box for understanding what  digits are detected and extracted
#5.added spinner etc more User friendly UI and debugging codes

