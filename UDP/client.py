﻿import socket
from enum import IntEnum
import os
import time
import cv2
import numpy as np
import json

class KeyData(IntEnum):
    None_ = 0
    LetterPrediction = 1
    HandRecognition = 2
    RawImageProcessing = 3

def send_all(sock, data):
    """Gửi tất cả dữ liệu qua socket"""
    total_sent = 0
    while total_sent < len(data):
        sent = sock.send(data[total_sent:])
        if sent == 0:
            raise RuntimeError("Socket connection broken")
        total_sent += sent

def recv_all(sock, n):
    """Nhận đủ n bytes từ socket"""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            raise RuntimeError("Socket connection broken")
        data.extend(packet)
    return data

def send_image_for_processing(sock, image_path, enhance_quality=True, upscale_method='bicubic'):
    """Gửi ảnh để xử lý và nhận về ảnh chỉ chứa bàn tay"""
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
            print(f"Đọc file ảnh thành công: {len(image_data)} bytes")
    except Exception as e:
        print(f"Không đọc được file ảnh: {e}")
        return None
        
    # Tạo options JSON
    options = {
        "enhance_quality": enhance_quality,
        "upscale_method": upscale_method
    }
    options_json = json.dumps(options).encode('utf-8')
    
    # Ghép options và image data
    full_payload = options_json + image_data

    # Đóng gói dữ liệu theo định dạng: [payload_length(4bytes)][key_data(4bytes)][payload]
    payload_length = len(full_payload).to_bytes(4, byteorder="little")
    key_bytes = int(KeyData.RawImageProcessing).to_bytes(4, byteorder="little")
    packet = payload_length + key_bytes + full_payload
    
    print(f"Kích thước gói tin: {len(packet)} bytes")
    print(f"Payload length: {int.from_bytes(payload_length, byteorder='little')} bytes")
    print(f"Key: {int.from_bytes(key_bytes, byteorder='little')}")
    print(f"Options: {options}")

    try:
        # Gửi dữ liệu
        send_all(sock, packet)
        print(f"Đã gửi gói tin ({len(packet)} bytes)")

        # Đọc phản hồi theo định dạng: [payload_length(4bytes)][key_data(4bytes)][payload]
        response_length = int.from_bytes(recv_all(sock, 4), byteorder="little")
        response_key = int.from_bytes(recv_all(sock, 4), byteorder="little")
        response_payload = recv_all(sock, response_length)
        
        if response_key == KeyData.RawImageProcessing:
            print(f"📥 Nhận ảnh đã xử lý: {len(response_payload)} bytes")
            
            # Hiển thị ảnh
            image = cv2.imdecode(np.frombuffer(response_payload, np.uint8), cv2.IMREAD_COLOR)
            cv2.imshow("Processed Hand", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            return True
        else:
            response_str = response_payload.decode("utf-8")
            print(f"📥 Nhận phản hồi: Key = {response_key}, Payload = {response_str}")
            return None

    except socket.timeout:
        print("❗ Không nhận được phản hồi từ server (timeout).")
        return None
    except Exception as e:
        print(f"❗ Lỗi khi gửi/nhận dữ liệu: {e}")
        return None

def send_image(sock, image_path):
    """Gửi một ảnh và nhận kết quả"""
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
            print(f"Đọc file ảnh thành công: {len(image_data)} bytes")
    except Exception as e:
        print(f"Không đọc được file ảnh: {e}")
        return None

    # Đóng gói dữ liệu theo định dạng: [payload_length(4bytes)][key_data(4bytes)][payload]
    payload_length = len(image_data).to_bytes(4, byteorder="little")
    key_bytes = int(KeyData.LetterPrediction).to_bytes(4, byteorder="little")
    packet = payload_length + key_bytes + image_data
    
    print(f"Kích thước gói tin: {len(packet)} bytes")
    print(f"Payload length: {int.from_bytes(payload_length, byteorder='little')} bytes")
    print(f"Key: {int.from_bytes(key_bytes, byteorder='little')}")

    try:
        # Gửi dữ liệu
        send_all(sock, packet)
        print(f"Đã gửi gói tin ({len(packet)} bytes)")

        # Đọc phản hồi theo định dạng: [payload_length(4bytes)][key_data(4bytes)][payload]
        response_length = int.from_bytes(recv_all(sock, 4), byteorder="little")
        response_key = int.from_bytes(recv_all(sock, 4), byteorder="little")
        response_payload = recv_all(sock, response_length).decode("utf-8")
        
        print(f"📥 Nhận phản hồi: Key = {response_key}, Payload = {response_payload}")
        return response_payload

    except socket.timeout:
        print("❗ Không nhận được phản hồi từ server (timeout).")
        return None
    except Exception as e:
        print(f"❗ Lỗi khi gửi/nhận dữ liệu: {e}")
        return None

def main():
    server_ip = "127.0.0.1"
    server_port = 5005
    
    # Danh sách các thư mục chứa ảnh
    image_dirs = [
        "dataset/raw/Train_Alphabet/A",
        "dataset/raw/Train_Alphabet/B",
        "dataset/raw/Train_Alphabet/C",
        # Thêm các thư mục khác nếu cần
    ]
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)

    try:
        sock.connect((server_ip, server_port))
        print(f"Đã kết nối đến {server_ip}:{server_port}")
        
        # Chọn chế độ
        print("\nChọn chế độ:")
        print("1. Dự đoán ký tự")
        print("2. Xử lý ảnh bàn tay")
        mode = input("Nhập lựa chọn của bạn (1/2): ")
        
        if mode == "1":
            # Duyệt qua từng thư mục
            for dir_path in image_dirs:
                print(f"\nĐang xử lý thư mục: {dir_path}")
                
                # Lấy danh sách các file ảnh trong thư mục
                image_files = [f for f in os.listdir(dir_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
                
                # Gửi từng ảnh
                for image_file in image_files:
                    image_path = os.path.join(dir_path, image_file)
                    print(f"\nĐang xử lý: {image_path}")
                    
                    # Gửi ảnh và nhận kết quả
                    result = send_image(sock, image_path)
                    
                    if result:
                        print(f"Kết quả nhận dạng: {result}")
                    
                    # Đợi một chút giữa các lần gửi để tránh quá tải
                    time.sleep(0.5)
        
        elif mode == "2":
            # Chọn một ảnh để xử lý
            image_path = input("Nhập đường dẫn đến file ảnh: ")
            
            if not os.path.exists(image_path):
                print("File ảnh không tồn tại!")
                return
                
            # Chọn phương pháp upscale
            print("\nChọn phương pháp upscale:")
            print("1. Bicubic (mặc định)")
            print("2. Lanczos")
            print("3. Detail Enhancement")
            print("4. Không upscale")
            
            upscale_choice = input("Nhập lựa chọn của bạn (1-4): ")
            
            enhance_quality = True
            upscale_method = 'bicubic'
            
            if upscale_choice == "2":
                upscale_method = 'lanczos'
            elif upscale_choice == "3":
                upscale_method = 'detail_enhance'
            elif upscale_choice == "4":
                enhance_quality = False
            
            # Gửi ảnh để xử lý
            success = send_image_for_processing(
                sock, 
                image_path, 
                enhance_quality=enhance_quality,
                upscale_method=upscale_method
            )
            
            if success:
                print("Ảnh đã được xử lý và hiển thị")

    except Exception as e:
        print(f"Lỗi kết nối: {e}")
    finally:
        sock.close()

if __name__ == "__main__":
    main()
