from paddleocr import PaddleOCR

# This downloads all PP-OCRv4 English models
ocr_engine = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    ocr_version='PP-OCRv4',
    use_gpu=False
)
