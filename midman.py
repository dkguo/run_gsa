from multiprocessing.connection import Listener, Client

midman_address = ('128.2.205.54', 60888)
server_address = ('10.1.1.110', 61888)

def start_midman():
    with Listener(midman_address) as listener:
        print('Server started. Listening for connections...')
        conn2client = listener.accept()
        print('Connection accepted from:', listener.last_accepted)

        client_msg = conn2client.recv()

        with Client(server_address) as conn2server:
            print('Connected to the server:', midman_address)
            conn2server.send(client_msg)
            while True:
                server_msg = conn2server.recv()
                conn2client.send(server_msg)


if __name__ == '__main__':
    start_midman()
