import easyocr

# Initialize the reader
reader = easyocr.Reader(['en'])

# Path to the image
image_path = "img3.jpg"

# Perform OCR on the image
result = reader.readtext(image_path)

# Extract the text from the result
extracted_text = [text for (bbox, text, _) in result]

# Display the extracted text
for text in extracted_text:
    print(text)
