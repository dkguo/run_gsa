from multiprocessing.connection import Listener

server_address = ('10.1.1.110', 68888)


def start_server():
    with Listener(server_address) as listener:
        print('Server started. Listening for connections...')
        conn = listener.accept()  # Wait for a connection from a client
        print('Connection accepted from:', listener.last_accepted)

        msg = conn.recv()
        print('Client:', msg)

        # Send messages to the client
        conn.send('Hello, client!')
        conn.send('This is the server.')
        conn.send('Goodbye!')


if __name__ == '__main__':
    start_server()
