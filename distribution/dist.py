import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import cv2

def calculate_tumor_brain_ratio(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Gaussian blur to smooth the image
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Use Otsu's thresholding to segment brain and tumor
    _, brain_mask = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours for the brain region
    contours, _ = cv2.findContours(brain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the largest contour assuming it is the brain
    brain_contour = max(contours, key=cv2.contourArea)
    brain_area = cv2.contourArea(brain_contour)

    # Apply thresholding to highlight the tumor region
    _, tumor_mask = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)

    # Find contours for the tumor region
    tumor_contours, _ = cv2.findContours(tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sum up all tumor areas
    tumor_area = sum(cv2.contourArea(cnt) for cnt in tumor_contours)

    # Calculate the ratio
    if brain_area > 0:
        ratio = tumor_area / brain_area
    else:
        ratio = 0

    # Determine tumor type based on the ratio (Example classification)
    if ratio > 0.1:
        tumor_type = "High-Grade Glioma (HGG)"
    else:
        tumor_type = "Low-Grade Glioma (LGG)"

    return tumor_area, brain_area, ratio, tumor_type

def process_images(folder_path):
    results = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(folder_path, filename)
            tumor_area, brain_area, ratio, tumor_type = calculate_tumor_brain_ratio(image_path)
            results.append([filename, tumor_area, brain_area, ratio, tumor_type])

    df = pd.DataFrame(results, columns=["Filename", "Tumor Area", "Brain Area", "Ratio", "Tumor Type"])

    # Summary statistics
    hgg_count = df[df["Tumor Type"] == "High-Grade Glioma (HGG)"].shape[0]
    lgg_count = df[df["Tumor Type"] == "Low-Grade Glioma (LGG)"].shape[0]
    total_tumors = df.shape[0]

    print(f"Total Images Processed: {total_tumors}")
    print(f"HGG Count: {hgg_count}")
    print(f"LGG Count: {lgg_count}")

    print(df)
    return df

# Example usage
folder_path = "./"
process_images(folder_path)
