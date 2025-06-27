from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="./runs/detect/train/weights/best.pt",
    confidence_threshold=0.25,
    device="cpu",  # or 'cuda:0'
)

result = get_sliced_prediction(
    "./frame_0.jpg",
    detection_model,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)

result.image.save("output.jpg")