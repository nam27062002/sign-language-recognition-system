import socket
from enum import IntEnum

class KeyData(IntEnum):
    None_ = 0
    LetterPrediction = 1

def main():
    server_ip = "10.241.2.88"
    server_port = 5005
    image_path = "D:\\PBL5_Python\\dataset\\asl_dataset\\test\\B_test.jpg"

    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
    except Exception as e:
        print(f"Không đọc được file ảnh: {e}")
        return

    key_bytes = int(KeyData.LetterPrediction).to_bytes(4, byteorder="little")
    packet = key_bytes + image_data
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(5)

    try:
        sock.sendto(packet, (server_ip, server_port))
        print(f"Đã gửi gói tin ({len(packet)} bytes) đến {server_ip}:{server_port}")
        response, addr = sock.recvfrom(65536)
        if len(response) < 4:
            print("❗ Phản hồi nhận được ít hơn 4 byte.")
            return

        response_key = int.from_bytes(response[:4], byteorder="little")
        response_payload = response[4:].decode("utf-8")
        print(f"📥 Nhận phản hồi từ {addr}: Key = {response_key}, Payload = {response_payload}")

    except socket.timeout:
        print("❗ Không nhận được phản hồi từ server (timeout).")
    except Exception as e:
        print(f"❗ Lỗi khi gửi/nhận dữ liệu: {e}")
    finally:
        sock.close()

if __name__ == "__main__":
    main()
