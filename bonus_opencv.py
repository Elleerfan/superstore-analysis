import cv2
import os
import numpy as np

print("OpenCV loaded!")

chart_files = [
    "category_chart.png",
    "region_chart.png",
    "monthly_trend.png",
    "heatmap.png",
    "top_products.png",
    "actual_vs_predicted.png",
    "training_progress.png"
]

images = []
for file in chart_files:
    img = cv2.imread(file)
    img = cv2.resize(img, (600, 400))
    images.append(img)
    print(f"✅ Loaded: {file}")

print(f"\nTotal charts loaded: {len(images)}")

row1 = np.hstack(images[0:3])
row2 = np.hstack(images[3:6])
row3 = np.hstack([images[6], np.zeros((400, 1200, 3), dtype=np.uint8)])

report = np.vstack([row1, row2, row3])

cv2.imwrite("final_report.png", report)
print("✅ Final report saved as final_report.png!")
