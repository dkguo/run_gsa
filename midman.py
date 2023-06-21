from multiprocessing import Process
from multiprocessing.connection import Listener, Client

from config import server_address, midman_address


def handle_client(conn2client):
    args = conn2client.recv()
    with Client(server_address) as conn2server:
        conn2server.send(args)
        while True:
            try:
                server_msg = conn2server.recv()
                conn2client.send(server_msg)
            except EOFError:
                break
    conn2client.close()


def start_midman():
    listener = Listener(midman_address)
    print('Server started. Listening for connections...')

    while True:
        conn2client = listener.accept()
        print('Connection accepted from:', listener.last_accepted)
        p = Process(target=handle_client, args=[conn2client])
        p.start()


if __name__ == '__main__':
    start_midman()
