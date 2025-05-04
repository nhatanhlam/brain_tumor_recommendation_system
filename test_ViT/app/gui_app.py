import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tkinter import filedialog, messagebox
import tkinter as tk
from PIL import Image, ImageTk
from predict import predict  # Import predict function

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        try:
            # Display the image
            img = Image.open(file_path).resize((300, 300))
            img_tk = ImageTk.PhotoImage(img)
            panel.configure(image=img_tk)
            panel.image = img_tk  # Force update

            # Get the prediction result
            prediction, confidence = predict(file_path)  # Update to unpack tuple
            result_label.config(
                text=f"Prediction: {prediction}\nConfidence: {confidence:.2f}%",
                fg="blue"
            )  # Update the result label
        except FileNotFoundError:
            result_label.config(text="Error: File not found.", fg="red")
            messagebox.showerror("Error", "The selected image file could not be found.")
        except Exception as e:
            result_label.config(text="Error: Unable to process the image", fg="red")
            messagebox.showerror("Error", f"An error occurred during prediction:\n{str(e)}")

# GUI setup
app = tk.Tk()
app.title("Brain Tumor Detection System")
app.geometry("400x550")
app.resizable(False, False)

# Title Label
title_label = tk.Label(app, text="Brain Tumor Detection System", font=("Arial", 18, "bold"), pady=10)
title_label.pack()

# Upload Button
upload_btn = tk.Button(app, text="Upload Image", command=upload_image, font=("Arial", 12), bg="#4CAF50", fg="white")
upload_btn.pack(pady=10)

# Image Panel
panel = tk.Label(app, bg="gray", width=300, height=300)
panel.pack(pady=10)

# Result Label
result_label = tk.Label(app, text="Prediction: ", font=("Arial", 14))
result_label.pack(pady=10)

app.mainloop()
