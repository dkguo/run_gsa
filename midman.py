import time
from multiprocessing import Process, Manager, Array
from multiprocessing.connection import Listener, Connection
from typing import NamedTuple

midman_address = ('128.2.205.54', 60888)


class Predictor(NamedTuple):
    name: str
    conn: Connection
    state: str


def handle_client(conn, predictor_conns, predictor_states):
    hello_msg = conn.recv()
    if type(hello_msg) == str and 'predictor' in hello_msg:
        # register predictor
        idx = len(predictor_conns)
        conn.send(str(idx))
        predictor_conns.append(conn)
        predictor_states[idx] = True
        print(f'Predictor {idx} connected.')
    else:
        args = hello_msg
        print('Received arguments.')
        while True:
            for pred_i, (pred_conn, is_idle) in enumerate(zip(predictor_conns, predictor_states)):
                if is_idle:
                    print(f'Sending request to Predictor {pred_i}...')
                    print(args)
                    with predictor_states.get_lock():
                        predictor_states[pred_i] = False
                    pred_conn.send(args)
                    print(f'Request received. Waiting for Predictor {pred_i}...')
                    prediction = pred_conn.recv()
                    print(f'Predictor {pred_i} finished.')
                    with predictor_states.get_lock():
                        predictor_states[pred_i] = True
                    conn.send(prediction)
                    return
            time.sleep(10)


def start_midman():
    manager = Manager()
    predictor_conns = manager.list()
    predictor_states = Array('b', [False] * 100)

    listener = Listener(midman_address)
    print('Server started. Listening for connections...')

    while True:
        conn = listener.accept()
        print('Connection accepted from:', listener.last_accepted)
        p = Process(target=handle_client, args=[conn, predictor_conns, predictor_states])
        p.start()


if __name__ == '__main__':
    start_midman()
