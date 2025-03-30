import socket
from enum import IntEnum

class KeyData(IntEnum):
    None_ = 0
    LetterPrediction = 1
    HandRecognition = 2

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

def main():
    server_ip = "127.0.0.1"
    server_port = 5005
    image_path = "dataset/raw/Train_Alphabet/A/A_0.png"
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
            print(f"Đọc file ảnh thành công: {len(image_data)} bytes")
    except Exception as e:
        print(f"Không đọc được file ảnh: {e}")
        return

    # Đóng gói dữ liệu theo định dạng: [payload_length(4bytes)][key_data(4bytes)][payload]
    payload_length = len(image_data).to_bytes(4, byteorder="little")
    key_bytes = int(KeyData.LetterPrediction).to_bytes(4, byteorder="little")
    packet = payload_length + key_bytes + image_data
    
    print(f"Kích thước gói tin: {len(packet)} bytes")
    print(f"Payload length: {int.from_bytes(payload_length, byteorder='little')} bytes")
    print(f"Key: {int.from_bytes(key_bytes, byteorder='little')}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)

    try:
        sock.connect((server_ip, server_port))
        print(f"Đã kết nối đến {server_ip}:{server_port}")
        
        # Gửi dữ liệu
        send_all(sock, packet)
        print(f"Đã gửi gói tin ({len(packet)} bytes)")

        # Đọc phản hồi theo định dạng: [payload_length(4bytes)][key_data(4bytes)][payload]
        response_length = int.from_bytes(recv_all(sock, 4), byteorder="little")
        response_key = int.from_bytes(recv_all(sock, 4), byteorder="little")
        response_payload = recv_all(sock, response_length).decode("utf-8")
        
        print(f"📥 Nhận phản hồi: Key = {response_key}, Payload = {response_payload}")

    except socket.timeout:
        print("❗ Không nhận được phản hồi từ server (timeout).")
    except Exception as e:
        print(f"❗ Lỗi khi gửi/nhận dữ liệu: {e}")
    finally:
        sock.close()

if __name__ == "__main__":
    main()
