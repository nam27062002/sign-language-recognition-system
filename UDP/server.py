import asyncio
import socket
from enum import IntEnum
from SignLanguageRecognizer import SignLanguageRecognizer


class KeyData(IntEnum):
    None_ = 0
    LetterPrediction = 1


class UDPProtocol(asyncio.DatagramProtocol):
    def __init__(self, asl_classifier: SignLanguageRecognizer, buffer_size=65536):
        super().__init__()
        self.asl_classifier = asl_classifier
        self.buffer_size = buffer_size
        self.transport = None

    def connection_made(self, transport):
        self.transport = transport
        print("Connection made; protocol is ready to receive data.")

    def datagram_received(self, data: bytes, addr):
        if len(data) < 4:
            print("Received data < 4 bytes. Skipping.")
            return
        key_value = int.from_bytes(data[:4], byteorder='little', signed=False)
        key_data = KeyData(key_value) if key_value in KeyData._value2member_map_ else None
        payload = data[4:]
        print(f"Packet from {addr} | KeyData = {key_data}, payload length = {len(payload)}")

        if key_data == KeyData.LetterPrediction:
            try:
                predicted_label, confidence = self.asl_classifier.predict(input_data=payload)
                print(f"Predicted: {predicted_label}, Confidence: {confidence:.4f}")
                response_str = f"Predicted: {predicted_label}, Confidence: {confidence:.4f}"
                response_key_data = KeyData.LetterPrediction
                response_key_bytes = int(response_key_data).to_bytes(4, 'little')
                payload_bytes = response_str.encode('utf-8')
                final_bytes = response_key_bytes + payload_bytes
                self.transport.sendto(final_bytes, addr)
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                print(error_msg)
                response_key_bytes = int(KeyData.None_).to_bytes(4, 'little')
                payload_bytes = error_msg.encode('utf-8')
                final_bytes = response_key_bytes + payload_bytes
                self.transport.sendto(final_bytes, addr)


class AsyncUDPServer:
    def __init__(self, ip="127.0.0.1", port=5005, buffer_size=65536, model_path="../models/CNN/cnn_best_model.keras"):
        self.ip = ip
        self.port = port
        self.buffer_size = buffer_size
        self.asl_classifier = SignLanguageRecognizer(model_path)

    async def start_server(self):
        loop = asyncio.get_running_loop()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.buffer_size)
        sock.bind((self.ip, self.port))
        print(f"UDP server bound to {self.ip}:{self.port}")
        transport, protocol = await loop.create_datagram_endpoint(
            lambda: UDPProtocol(self.asl_classifier, self.buffer_size),
            sock=sock
        )
        print("Server ready to receive data...")
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            transport.close()
            print("Server stopped.")


def main():
    server = AsyncUDPServer(
        ip="127.0.0.1",
        port=5005,
        buffer_size=1024 * 1024,
        model_path="../models/CNN/cnn_best_model.keras"
    )
    asyncio.run(server.start_server())


if __name__ == "__main__":
    main()
