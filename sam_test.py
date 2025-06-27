from ultralytics import SAM
import cv2

# Load a model
model = SAM("sam2.1_b.pt")

# Display model information (optional)
model.info()

image_path = "./extracted_frames/1.jpg"
point_list_1 = [[1681, 1801], [1772, 1799], [1912, 1721], [2109, 1774], [2048, 1910], [2304, 1882], [2318, 1935], [2435, 1824], [2499, 1815], [2485, 1874], [2574, 1796], [2647, 1846], [2516, 1904], [2597, 1921], [2677, 1927], [2785, 1890], [2921, 1941], [2924, 1860], [2977, 1874], [3028, 1861], [3247, 1999], [3235, 1916], [3317, 1914], [3426, 2043], [3519, 2059], [3587, 2035]]
point_list_2 = []
point_list_3 = []

results = model(image_path, points=point_list_2)

image = results[0].plot(labels=False)
cv2.imshow("sam_test", image)
cv2.waitKey()
'''
for i, res in enumerate(results):
    normalized_bboxes = res.boxes.xywhn  # x, y, w, h in [0,1] format
    output_path = image_path.replace(".jpg", ".txt")
    with open(output_path, "w", encoding="utf-8") as f:
        for nbbox in normalized_bboxes:
            x, y, w, h = nbbox
            f.write("0 {} {} {} {}\n".format(x, y, w, h))  # class_id = 0
'''