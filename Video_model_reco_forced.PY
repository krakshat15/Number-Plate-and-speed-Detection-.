import cv2
import pytesseract
import pandas as pd
import openpyxl

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Initialize video capture object
cap = cv2.VideoCapture(0)

# Load license plate detection algorithm
plate_cascade = cv2.CascadeClassifier('D:\PROJECT\AKSHAT\splate_number.xml')

# Initialize empty DataFrame for storing recognized license plates
data = pd.DataFrame(columns=['License Plate'])


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Preprocess the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 100, 200)
    thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Detect license plate regions
    plates = plate_cascade.detectMultiScale(thresh, 1.1, 4)
    
    # Extract license plate characters and recognize them
    for (x, y, w, h) in plates:
        plate = gray[y:y+h, x:x+w]
        plate = cv2.threshold(plate, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        plate = cv2.medianBlur(plate, 3)
        plate_text = pytesseract.image_to_string(plate, config='--psm 11')
        new_row = pd.DataFrame({'License Plate': [plate_text]})
        data = pd.concat([data, new_row], ignore_index=True)

        
    # Display the resulting frame
    cv2.imshow('frame', frame)
    
    # Save the recognized license plates to Excel file
    data.to_excel('recognized_plates.xlsx', index=False)
    
    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

# Release the capture object and destroy windows
cap.release()
cv2.destroyAllWindows()

import datetime
import cv2

def process_image(image):
    # Detect license plate
    plate_image, plate_coords = detect_license_plate(image)

    # Extract license plate text
    plate_text = recognize_license_plate(plate_image)

    # Get current date and time
    now = datetime.datetime.now()

    # Create a new row with license plate text and date/time
    new_row = pd.DataFrame({'License Plate': [plate_text], 'Date/Time': [now]})

    # Append the new row to the existing data DataFrame
    global data
    data = data.append(new_row, ignore_index=True)

    # Save the DataFrame to Excel file
    data.to_excel('recognized_plates.xlsx', index=False)

    # Draw green box around license plate on image
    x1, y1, x2, y2 = plate_coords
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display license plate text and date/time
    print('License plate:', plate_text)
    print('Date/time:', now)

    # Display the image with the green box
    cv2.imshow('License Plate Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
