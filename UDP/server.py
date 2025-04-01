import sys
import os
import platform
import cv2
import numpy as np
from datetime import datetime
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import asyncio
from enum import IntEnum
from training.model_pytorch import SignLanguageRecognizer, ModelType


class KeyData(IntEnum):
    None_ = 0
    LetterPrediction = 1
    HandRecognition = 2
    RawImageProcessing = 3


class TCPProtocol:
    def __init__(self, md:SignLanguageRecognizer):
        self.model = md
        self.transport = None
        self.buffer = bytearray()
        self.debug_mode = False
        self.processing = False
        
        # Thư mục lưu ảnh đã xử lý
        self.processed_dir = os.path.join(root_dir, "processed_hands")
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def save_processed_image(self, image_bytes):
        try:
            # Tạo timestamp để đặt tên file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"hand_{timestamp}.png"
            filepath = os.path.join(self.processed_dir, filename)
            
            # Chuyển đổi bytes thành ảnh và lưu
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            cv2.imwrite(filepath, image)
            
            if self.debug_mode:
                print(f"Saved processed image to: {filepath}")
                
            return filepath
        except Exception as e:
            if self.debug_mode:
                print(f"Error saving processed image: {e}")
            return None

    def set_debug_mode(self, enable: bool):
        self.debug_mode = enable
        print(f"Debug mode {'On' if enable else 'Off'}")

    def connection_made(self, transport):
        self.transport = transport
        if self.debug_mode:
            print("Connection made with client.")

    def data_received(self, data: bytes):
        if self.processing:
            return
            
        self.processing = True
        try:
            self.buffer.extend(data)
            while len(self.buffer) >= 8:
                payload_length = int.from_bytes(self.buffer[:4], byteorder='little', signed=False)
                if len(self.buffer) < 4 + 4 + payload_length:
                    break

                key_value = int.from_bytes(self.buffer[4:8], byteorder='little', signed=False)
                key_data = KeyData(key_value) if key_value in KeyData._value2member_map_ else None
                payload = self.buffer[8:8 + payload_length]
                if self.debug_mode:
                    print(f"Received | KeyData = {key_data}, payload length = {len(payload)}")

                if key_data == KeyData.LetterPrediction:
                    try:
                        result = self.model.predict(input_data=payload)
                        if result is None:
                            error_msg = "Cannot process image"
                            if self.debug_mode:
                                print(error_msg)
                            self.send_response(KeyData.None_, error_msg.encode('utf-8'))
                            del self.buffer[:8 + payload_length]
                            return
                        
                        predicted_label, confidence = result
                        if self.debug_mode:
                            print(f"Predicted: {predicted_label}, Confidence: {confidence:.4f}")
                        response_str = f"Predicted: {predicted_label}, Confidence: {confidence:.4f}"
                        response_key_data = KeyData.LetterPrediction
                        self.send_response(response_key_data, response_str.encode('utf-8'))
                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        if self.debug_mode:
                            print(error_msg)
                        self.send_response(KeyData.None_, error_msg.encode('utf-8'))
                        del self.buffer[:8 + payload_length]
                        return

                elif key_data == KeyData.HandRecognition:
                    try:
                        has_hand = self.model.detect_hand(payload)
                        response_str = "true" if has_hand else "false"
                        if self.debug_mode:
                            print(f"Hand detected: {response_str}")
                        self.send_response(KeyData.HandRecognition, response_str.encode('utf-8'))
                    except Exception as e:
                        error_msg = f"Hand detection error: {str(e)}"
                        if self.debug_mode:
                            print(error_msg)
                        self.send_response(KeyData.None_, error_msg.encode('utf-8'))
                
                elif key_data == KeyData.RawImageProcessing:
                    try:
                        processed_image = self.model.process_hand_image(payload)
                        if processed_image is None:
                            error_msg = "Cannot process image, no hand detected"
                            if self.debug_mode:
                                print(error_msg)
                            self.send_response(KeyData.None_, error_msg.encode('utf-8'))
                        else:
                            # Lưu ảnh đã xử lý
                            saved_path = self.save_processed_image(processed_image)
                            
                            # if self.debug_mode:
                            print(f"Processed image size: {len(processed_image)} bytes")
                            if saved_path:
                                print(f"Saved to: {saved_path}")
                                    
                            self.send_response(KeyData.RawImageProcessing, processed_image)
                    except Exception as e:
                        error_msg = f"Image processing error: {str(e)}"
                        if self.debug_mode:
                            print(error_msg)
                        self.send_response(KeyData.None_, error_msg.encode('utf-8'))
                
                del self.buffer[:8 + payload_length]
        finally:
            self.processing = False

    def send_response(self, key_data: KeyData, payload: bytes):
        try:
            key_bytes = int(key_data).to_bytes(4, 'little')
            length_bytes = len(payload).to_bytes(4, 'little')
            final_bytes = length_bytes + key_bytes + payload
            self.transport.write(final_bytes)
            if self.debug_mode:
                print(f"Sent response: {len(final_bytes)} bytes")
        except Exception as e:
            if self.debug_mode:
                print(f"Error sending response: {e}")

    def connection_lost(self, exc):
        if self.debug_mode:
            print("Connection lost with client.")

    def eof_received(self):
        if self.debug_mode:
            print("EOF received, closing connection")
        return False

class AsyncTCPServer:
    def __init__(self, ip="127.0.0.1", port=5005, model_type = ModelType.CNN, model_path = None):
        self.ip = ip
        self.port = port
        if model_path is None:
            model_path = os.path.join(root_dir, "models", "CNN", "cnn_best_model_v1.pth")
        self.model = SignLanguageRecognizer(model_type, model_path)
        self.max_connections = 100
        self.connections = set()

    async def start_server(self):
        loop = asyncio.get_running_loop()
        server_config = {
            'backlog': self.max_connections,
            'reuse_address': True,
            'start_serving': True
        }
        
        if platform.system() != 'Windows':
            server_config['reuse_port'] = True
            
        server = await loop.create_server(
            lambda: TCPProtocol(self.model),
            self.ip,
            self.port,
            **server_config
        )
        print(f"TCP server bound to {self.ip}:{self.port}")
        async with server:
            await server.serve_forever()


def main():
    server = AsyncTCPServer(
        ip="127.0.0.1",
        port=5005,
    )
    asyncio.run(server.start_server())


if __name__ == "__main__":
    main()