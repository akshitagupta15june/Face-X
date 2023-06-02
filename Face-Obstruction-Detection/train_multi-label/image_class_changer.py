import os
import json
from tkinter import Tk, Label, Button
from PIL import Image, ImageTk
import csv

# Set the directory where your images are located
image_directory = "data"
image_filenames = sorted(f for f in os.listdir(image_directory) if f.endswith(('.png', '.jpg', '.jpeg')))
print(image_filenames)

# Load the last processed image index
try:
    with open('last_image.json', 'r') as f:
        last_image_index = json.load(f)
    image_filenames = image_filenames[last_image_index:]
except FileNotFoundError:
    last_image_index = 0  # Default value if the file does not exist

image_iterator = iter(image_filenames)

# Set your labels
labels = ['glasses', 'hand', 'mask', 'none', 'other', 'sunglasses']

# Load existing data
csv_file_name = 'dataset_5_small.csv'
try:
    with open(csv_file_name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        csv_data = list(csv_reader)  # Load existing CSV data into memory
    header = csv_data.pop(0)  # Separate the header and data
except FileNotFoundError:
    header = ['id'] + labels
    csv_data = []

# Create the root window
root = Tk()

# Add the assigned classes label to the window
assigned_classes_label = Label(root)
assigned_classes_label.grid(row=2, column=0, columnspan=3)

# Function to load the next image
# Function to load the next image and update the assigned classes label
def load_next_image():
    # Select the next image
    image_filename = next(image_iterator)
    image_path = os.path.join(image_directory, image_filename)

    # Load the image
    image = Image.open(image_path)
    # Resize image for display
    image = image.resize((300, 300), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(image)

    # Update the assigned classes label
    matching_row = next((row for row in csv_data if row[0] == image_filename), None)
    if matching_row is not None:
        assigned_classes = [label for i, label in enumerate(labels, start=1) if matching_row[i] == '1']
        assigned_classes_label['text'] = 'Assigned classes: ' + ', '.join(assigned_classes)
    else:
        assigned_classes_label['text'] = 'No assigned classes'

    return photo, image_filename

# Add the image to the window
img_label = Label(root)
img_name_label = Label(root)  # Label to display the image name
img_label.image, img_name_label['text'] = load_next_image()
img_label.configure(image=img_label.image)
img_name_label.configure(text=img_name_label['text'])

img_label.grid(row=0, column=0, columnspan=3)
img_name_label.grid(row=1, column=0, columnspan=3)  # Display the image name below the image


# Dictionary to store button states
button_states = {label: None for label in labels}  # Initialize to None instead of 0

# Function to change button color
def button_click(button, label):
    button.config(bg="green")
    button_states[label] = 1
def button_click_no(button, label):
    button.config(bg="green")
    button_states[label] = 0

# Function to reset button color
def reset_buttons(buttons):
    for button in buttons:
        button.config(bg=root.cget("bg"))
    for label in labels:
        button_states[label] = None  # Reset button states

# Function to return command function
def make_command(button, label):
    return lambda: button_click(button, label)
def make_command_no(button, label):
    return lambda: button_click_no(button, label)

# Add the labels and buttons to the window
button_list = []
for index, label in enumerate(labels):
    label_frame = Label(root, text=label)
    label_frame.grid(row=index+3, column=0)  # Start from 3rd row to leave space for the image and the assigned classes label

    yes_button = Button(root, text="Yes")
    yes_button['command'] = make_command(yes_button, label)
    yes_button.grid(row=index+3, column=1)
    button_list.append(yes_button)
    no_button = Button(root, text="No")
    no_button['command'] = make_command_no(no_button, label)
    no_button.grid(row=index+3, column=2)
    button_list.append(no_button)

# Function to update the current image's data
def update_data():
    image_name = img_name_label['text']  # Get the image name from the label

    # Get the row that matches the current image, or None if it does not exist
    matching_row = next((row for row in csv_data if row[0] == image_name), None)

    if matching_row is not None:
        # Update the matching row
        for i, label in enumerate(labels, start=1):  # Start at 1 to skip the 'id' column
            if button_states[label] is not None:
                matching_row[i] = button_states[label]
    else:
        # Add a new row for the current image
        new_row = [image_name] + [button_states[label] if button_states[label] is not None else 0 for label in labels]
        csv_data.append(new_row)


# Modify the next_image function like this:
def next_image():
    update_data()  # Update the current image's data before moving to the next image
    img_label.image, img_name_label['text'] = load_next_image()
    img_label.configure(image=img_label.image)
    img_name_label.configure(text=img_name_label['text'])
    reset_buttons(button_list)
    global last_image_index
    last_image_index += 1
    with open('last_image.json', 'w') as f:
        json.dump(last_image_index, f)

# Add "Next" button
next_button = Button(root, text="Next", command=next_image)
next_button.grid(row=len(labels)+3, column=1)  # Put the button below the labels

# Define a function to save the updated data and destroy the window
def on_close():
    update_data()  # Update the current image's data before closing
    with open(csv_file_name, 'w', newline='') as csv_file:  # Write the updated data back to the CSV file
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(header)
        csv_writer.writerows(csv_data)
    root.destroy()

# Set what should happen when the window is closed
root.protocol("WM_DELETE_WINDOW", on_close)

root.mainloop()