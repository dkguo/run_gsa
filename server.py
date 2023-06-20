from multiprocessing.connection import Listener

address = ('10.10.0.0', 6000)  # Remote server address

def start_server():
    with Listener(address) as listener:
        print('Server started. Listening for connections...')
        conn = listener.accept()  # Wait for a connection from a client
        print('Connection accepted from:', listener.last_accepted)

        # Send messages to the client
        conn.send('Hello, client!')
        conn.send('How are you?')
        conn.send('Goodbye!')

if __name__ == '__main__':
    start_server()