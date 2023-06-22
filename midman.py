import time
from dataclasses import dataclass
from multiprocessing import Process
from multiprocessing.connection import Listener, Connection

midman_address = ('128.2.205.54', 60888)
predictors = []


@dataclass
class Predictor:
    conn: Connection
    state: str


def handle_client(conn):
    hello_msg = conn.recv()
    if 'predictor' in hello_msg:
        # register predictor
        name = str(len(predictors))
        conn.send(name)
        predictors.append(Predictor(conn, 'idle'))
        print(f'Predictor {name} connected.')
    else:
        args = hello_msg
        while True:
            for predictor in predictors:
                if predictor.state == 'idle':
                    predictor.state = 'busy'
                    predictor.conn.send(args)
                    print(f'Request received. Waiting for Predictor {predictor.name}...')
                    prediction = predictor.conn.recv()
                    print(f'Predictor {predictor.name} finished.')
                    predictor.state = 'idle'
                    conn.send(prediction)
                    return
            time.sleep(10)


def start_midman():
    listener = Listener(midman_address)
    print('Server started. Listening for connections...')

    while True:
        conn = listener.accept()
        print('Connection accepted from:', listener.last_accepted)
        p = Process(target=handle_client, args=[conn])
        p.start()


if __name__ == '__main__':
    start_midman()
