import cv2
import numpy as np

def detect_and_label_circles_of_body(imagepath, labeled_data):
    """
    Detects pink color regions in the image, fits circles around them, and labels body parts.

    Parameters:
        image (numpy.ndarray): The input image.
        labeled_data (dict): Dictionary to store labeled body points.

    Returns:
        numpy.ndarray: Image with circles and labels drawn.
        dict: Updated dictionary with labeled body points.
    """
    # Convert the image from BGR to HSV color space
    image = cv2.imread(imagepath)
    image = cv2.resize(image,(600,720))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range of pink color in HSV
    lower_pink = np.array([160, 70, 70])   # Broader lower bound of pink color
    upper_pink = np.array([180, 255, 255])

    # Create a mask for the pink color
    mask = cv2.inRange(hsv, lower_pink, upper_pink)

    # Find contours of the detected pink regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # List to store the circle data
    circles_data = []

    # Iterate through each contour
    for contour in contours:
        # Fit a minimum enclosing circle around the contour
        (x, y), radius = cv2.minEnclosingCircle(contour)
        radius = int(radius)  # Convert radius to an integer
        
        # Check if the radius is greater than 5
        if radius >1:
            # Store the center and radius in the list
            circles_data.append(((int(x), int(y)), radius))

    # Sort the circles based on their y-coordinate (vertical position)
    circles_data.sort(key=lambda c: c[0][1])  # Sort by y-coordinate

    # Divide circles into shoulders, hands, hips, and knees
    if len(circles_data) >= 8:
        # Most upper two circles are shoulders
        shoulders = sorted(circles_data[:2], key=lambda c: c[0][0])  # Sort by x for left/right
        # Next two circles are hands
        hands = sorted(circles_data[2:4], key=lambda c: c[0][0])  # Sort by x for left/right
        # Middle two circles are hips
        hips = sorted(circles_data[4:6], key=lambda c: c[0][0])  # Sort by x for left/right
        # Most lower two circles are knees
        knees = sorted(circles_data[6:8], key=lambda c: c[0][0])  # Sort by x for left/right

        # Assign labels
        body_labels = [
            ("Right Shoulder", shoulders[0][0]), ("Left Shoulder", shoulders[1][0]),
            ("Right Hand", hands[0][0]), ("Left Hand", hands[1][0]),
            ("Right Hip", hips[0][0]), ("Left Hip", hips[1][0]),
            ("Right Knee", knees[0][0]), ("Left Knee", knees[1][0])
        ]
    else:
        body_labels = [(f"Circle {i+1}", c[0]) for i, c in enumerate(circles_data)]

    # Draw and label the circles
    for label, center in body_labels:
        x, y = center
        cv2.circle(image, center, 5, (0, 255, 0), thickness=-1)  # Green filled circle
        cv2.putText(image, label, (x - 40, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        labeled_data[label] = center

    return image, labeled_data

def detect_and_label_eyes(image, labeled_data, target_size=(600, 720)):
    """
    Detects eyes (left and right) in the image, labels them, and adjusts their positions based on the target size.

    Parameters:
        image (numpy.ndarray): The input image.
        labeled_data (dict): Dictionary to store labeled eye points.
        target_size (tuple): The target size (width, height) to scale the eye positions.

    Returns:
        numpy.ndarray: Image with eyes and labels drawn.
        dict: Updated dictionary with labeled eye points scaled to target size.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the classifiers
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)

    # Calculate scaling factors for width and height
    actual_size = (image.shape[1], image.shape[0])  # (width, height)
    scale_width = target_size[0] / actual_size[0]
    scale_height = target_size[1] / actual_size[1]

    for (x, y, w, h) in faces:
        # Scale the face ROI to match the target size
        # x = int(x * scale_width)
        # y = int(y * scale_height)
        # w = int(w * scale_width)
        # h = int(h * scale_height)
        x = int(x )
        y = int(y )
        w = int(w )
        h = int(h )

        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # Sort detected eyes by x-coordinates (left-to-right in the image)
        eyes = sorted(eyes, key=lambda e: e[0])
        # print("eyessss",eyes)
        if len(eyes) >= 2:
            # Right Eye
            ex, ey, ew, eh = eyes[0]
            right_eye_center = (x + ex + ew // 2, y + ey + eh // 2)
            right_eye_center_scaled = (int(right_eye_center[0] * scale_width), int(right_eye_center[1] * scale_height))
            cv2.circle(image, right_eye_center, 5, (0, 0, 255), -1)
            # cv2.putText(image, "Right Eye", (right_eye_center[0] - 50, right_eye_center[1]), 
                        # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            labeled_data["Right Eye"] = right_eye_center_scaled

            # Left Eye
            ex, ey, ew, eh = eyes[1]
            left_eye_center = (x + ex + ew // 2, y + ey + eh // 2)
            left_eye_center_scaled = (int(left_eye_center[0] * scale_width), int(left_eye_center[1] * scale_height))
            cv2.circle(image, left_eye_center, 5, (0, 0, 255), -1)
            # cv2.putText(image, "Left Eye", (left_eye_center[0] + 10, left_eye_center[1]), 
                        # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            labeled_data["Left Eye"] = left_eye_center_scaled
        # print("eyeee points",labeled_data)

    return image, labeled_data


def main(image_path):
    """
    Main function to detect and label eyes and body points in a single image.
    
    Parameters:
        image_path (str): Path to the input image.
    """
    # Load and resize the image
    RealImage = cv2.imread(image_path)
    image = RealImage.copy()
    image = cv2.resize(image, (1300, 1720))

    # Dictionary to store all labeled points
    labeled_data = {}

    # Detect and label eyes
    image, labeled_data = detect_and_label_eyes(image, labeled_data, target_size=(600, 720))

    # Detect and label body points
    image, labeled_data = detect_and_label_circles_of_body(image_path, labeled_data)
    RealImage = cv2.resize(RealImage,(600,720))
    for key,val in labeled_data.items():
        print("valll",val)
        cv2.circle(RealImage,val,7,(60, 179, 113),-1)
    cv2.imwrite("Output.jpg",RealImage)
    
    
    
    # cv2.imwrite("output.jpg",image)
    # # Display the final image
    # cv2.imshow('Labeled Image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print("ffinal label data",labeled_data)
    # Print the labeled data
    # print("Labeled Data:")
    # for label, point in labeled_data.items():
    #     print(f"{label}: {point}")
    # target_size = (600, 720)

    # # Actual size (1200, 1720)
    # actual_size = (1200, 1720)

    # # Calculate scaling factors
    # scale_width = target_size[0] / actual_size[0]  # 0.5
    # scale_height = target_size[1] / actual_size[1]  # ~0.4186
    # scaled_points = {key: (int(x * scale_width), int(y * scale_height)) for key, (x, y) in labeled_data.items()}
    # print("scaled ",scaled_points)


if __name__ == "__main__":

    
    image_path = "inputImg.jpg"
    main(image_path)
