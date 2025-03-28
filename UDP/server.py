﻿import asyncio
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

    def connection_made(self, transport):
        self.transport = transport
        print("Connection made with client.")

    def data_received(self, data: bytes):
        self.buffer.extend(data)
        while len(self.buffer) >= 8:
            payload_length = int.from_bytes(self.buffer[:4], byteorder='little', signed=False)
            if len(self.buffer) < 4 + 4 + payload_length:
                break

            key_value = int.from_bytes(self.buffer[4:8], byteorder='little', signed=False)
            key_data = KeyData(key_value) if key_value in KeyData._value2member_map_ else None
            payload = self.buffer[8:8 + payload_length]
            print(f"Received | KeyData = {key_data}, payload length = {len(payload)}")

            if key_data == KeyData.LetterPrediction:
                try:
                    predicted_label, confidence = self.model.predict(input_data=payload)
                    print(f"Predicted: {predicted_label}, Confidence: {confidence:.4f}")
                    response_str = f"Predicted: {predicted_label}, Confidence: {confidence:.4f}"
                    response_key_data = KeyData.LetterPrediction
                    self.send_response(response_key_data, response_str.encode('utf-8'))
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    print(error_msg)
                    self.send_response(KeyData.None_, error_msg.encode('utf-8'))

            elif key_data == KeyData.HandRecognition:
                try:
                    has_hand = self.model.detect_hand(payload)
                    response_str = "true" if has_hand else "false"
                    print(f"Hand detected: {response_str}")
                    self.send_response(KeyData.HandRecognition, response_str.encode('utf-8'))
                except Exception as e:
                    error_msg = f"Hand detection error: {str(e)}"
                    print(error_msg)
                    self.send_response(KeyData.None_, error_msg.encode('utf-8'))
            del self.buffer[:8 + payload_length]

    def send_response(self, key_data: KeyData, payload: bytes):
        key_bytes = int(key_data).to_bytes(4, 'little')
        length_bytes = len(payload).to_bytes(4, 'little')
        final_bytes = length_bytes + key_bytes + payload
        self.transport.write(final_bytes)

    def connection_lost(self, exc):
        print("Connection lost with client.")

    def eof_received(self):
        print("EOF received, closing connection")
        return False

class AsyncTCPServer:
    def __init__(self, ip="127.0.0.1", port=5005, model_type = ModelType.CNN, model_path = "../models/CNN/cnn_best_model_v1.pth"):
        self.ip = ip
        self.port = port
        self.model = SignLanguageRecognizer(model_type, model_path)

    async def start_server(self):
        loop = asyncio.get_running_loop()
        server = await loop.create_server(lambda: TCPProtocol(self.model), self.ip, self.port)
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