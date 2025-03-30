import sys
import os
import platform
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

import asyncio
from enum import IntEnum
from training.model_pytorch import SignLanguageRecognizer, ModelType


class KeyData(IntEnum):
    None_ = 0
    LetterPrediction = 1
    HandRecognition = 2


class TCPProtocol:
    def __init__(self, md:SignLanguageRecognizer):
        self.model = md
        self.transport = None
        self.buffer = bytearray()
        self.debug_mode = False  # Default debug mode is off
        self.processing = False  # Flag to prevent concurrent processing

    def set_debug_mode(self, enable: bool):
        self.debug_mode = enable
        print(f"Debug mode {'On' if enable else 'Off'}")

    def connection_made(self, transport):
        self.transport = transport
        if self.debug_mode:
            print("Connection made with client.")

    def data_received(self, data: bytes):
        if self.processing:  # Skip if already processing
            return
            
        self.processing = True
        try:
            self.buffer.extend(data)
            while len(self.buffer) >= 8:  # Ensure we have at least 8 bytes (4 for length + 4 for key)
                payload_length = int.from_bytes(self.buffer[:4], byteorder='little', signed=False)
                if len(self.buffer) < 4 + 4 + payload_length:  # Check if we have enough data
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
                
                # Remove processed data from buffer
                del self.buffer[:8 + payload_length]
        finally:
            self.processing = False

    def send_response(self, key_data: KeyData, payload: bytes):
        """Send response with format: [payload_length(4bytes)][key_data(4bytes)][payload]"""
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
        self.max_connections = 100  # Số lượng kết nối tối đa
        self.connections = set()  # Theo dõi các kết nối đang hoạt động

    async def start_server(self):
        loop = asyncio.get_running_loop()
        server_config = {
            'backlog': self.max_connections,  # Allow multiple pending connections
            'reuse_address': True,  # Allow reuse of address
            'start_serving': True
        }
        
        # Add reuse_port only on Unix-like systems
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