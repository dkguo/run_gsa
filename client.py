from multiprocessing.connection import Client

address = ('128.2.205.54', 6000)  # Remote server address

def start_client():
    with Client(address) as conn:
        print('Connected to the server:', address)
        while True:
            msg = conn.recv()  # Receive messages from the server
            if msg == 'Goodbye!':
                print('Server:', msg)
                break
            else:
                print('Server:', msg)

if __name__ == '__main__':
    start_client()