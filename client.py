from multiprocessing.connection import Client

from config import midman_address


def start_client():
    with Client(midman_address) as conn:
        print('Connected to the server:', midman_address)
        conn.send('Hello, server! This is client.')
        while True:
            msg = conn.recv()  # Receive messages from the server
            if msg == 'Goodbye!':
                print('Server:', msg)
                break
            else:
                print('Server:', msg)


if __name__ == '__main__':
    start_client()
