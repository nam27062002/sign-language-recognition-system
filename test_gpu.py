import cv2
import platform

print("=== System Info ===")
print("OS:", platform.platform())
print("OpenCV version:", cv2.__version__)
print("OpenCL available:", cv2.ocl.haveOpenCL())
print("CPU cores:", cv2.getNumberOfCPUs())

test_image = cv2.UMat(cv2.imread("dataset/raw/Test_Alphabet/A/A_0.png"))
if test_image is None:
    print("❗ Không đọc được ảnh")
else:
    print("✅ UMat image created successfully")
    try:
        blurred = cv2.GaussianBlur(test_image, (5,5), 0)
        print("✅ GaussianBlur works with UMat")
    except Exception as e:
        print("❌ GaussianBlur error:", str(e))