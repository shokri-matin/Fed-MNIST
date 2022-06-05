import socket
import select
import pickle
import tensorflow as tf
from src.model import Model
from src.data_handler import DataHandler

class Server:

    def __init__(self, IP=socket.gethostname(), PORT=12345,
     HEADER_LENGTH=10, number_of_parties=2) -> None:
        
        self.HEADER_LENGTH = HEADER_LENGTH
        self.IP = IP
        self.PORT = PORT

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.server_socket.bind((IP, PORT))
        self.server_socket.listen()

        self.number_of_parties = number_of_parties

        self.socket_list = []
        self.parties = {}

        print("Server initialization is completed on IP: {0}, Port: {1}".format(IP, PORT))

    def receive_message(self, _client_socket):
        try:
            message_header = _client_socket.recv(self.HEADER_LENGTH)
            if not len(message_header):
                return False
            message_length = int(message_header.decode('utf-8').strip())
            return {"header": message_header, "data": _client_socket.recv(message_length)}

        except Exception as e:
            print(f"Server error : {e}")
            return False

    def send_message(self, party, message):

        serialized_data = pickle.dumps(message)

        message_header = f"{len(serialized_data):<{self.HEADER_LENGTH}}".encode('utf-8')

        party.sendall(message_header + serialized_data)

    def accept_clients(self):

        print("Server is waiting for parties, number of parties are: {0}".format(self.number_of_parties))
        # Register all parties
        i = 1
        while i <= self.number_of_parties:
            client_socket, client_address = self.server_socket.accept()
            party = self.receive_message(client_socket)
            self.socket_list.append(client_socket)
            self.parties[client_socket] = party
            i = i + 1

            print(f"Accepted new connection from {client_address[0]} : {client_address[1]} "
                f"username: {party['data'].decode('utf-8')}")

        print("All parties are accepted!")

    def send_weights(self, weights):
        
        for party in self.parties:
            self.send_message(party, weights)

    def recv_gradients(self):

        i = 0
        grads = {}
        flag = True
        while flag:
            read_sockets, _, exception_sockets = select.select(self.socket_list, [], self.socket_list)

            for notified_socket in read_sockets:
                if i != len(self.parties):
                    i = i + 1
                    message_header = notified_socket.recv(self.HEADER_LENGTH)
                    if not len(message_header):
                        return False
                    message_length = int(message_header.decode('utf-8').strip())

                    received_data = b""
                    current_length = 0
                    while current_length < message_length:
                        received_data += notified_socket.recv(message_length - current_length)
                        current_length = len(received_data)

                    gradient = pickle.loads(received_data)
                    grads[notified_socket] = gradient

                if i == len(self.parties):
                    flag = False

        return grads

    def aggregate(self, grads):

        parties = list(grads.keys())
        p0 = grads[parties[0]]
        p1 = grads[parties[1]]
        for i in range(0, len(p0)):
            p0[i] += p1[i]
            p0[i] /= 2

        # p1 = grads[parties[1]]
        # print(p0[1])
        # print(p1[1])
        # p1[1] += p0[1]
        # print(p1[1])
        # for i in range(1, len(parties)):
        #     curr_party = grads[parties[i]]
        #     print(curr_party)
        #     agg_parties[0] += tf.reduce_mean(curr_party[0])
        
        # print(agg_parties)
        # mean_grad = tf.zeros(())
        # for grad in grads:
        #     mean_grad += tf.reduce_mean(grad)
        # mean_grad /= len(grads)
        # aggregated_grad = []
        # parties = grads.keys()
        # aggregated_parties = parties[0]
        # for i in range(1, len(parties)):
        #     curr_party = grads[parties[i]]
        #     for l in curr_party:
        #         print(l)
        
        # aggregated_grad = aggregated_grad/2
        return p0

    def run(self, epochs=5000):   

        print("create a mediator")
        # create a mediator
        mediator = Model()
        # server accepts clients
        print("server accepts clients")
        self.accept_clients()

        epoch=0
        while True:
            try:
                # server sends its weights to clients
                # print("server sends its weights to clients")
                weights = mediator.model.get_weights()
                self.send_weights(weights)

                # server waits for clients to send their grads
                # print("server waits for clients to send their grads")
                gradients = self.recv_gradients()

                # server aggregates gradients recv from clients
                # for party in gradients.keys():
                mediator.optimizer.apply_gradients(zip(self.aggregate(gradients), mediator.model.trainable_weights))

                print('epochs:{}'.format(epoch))
                
                if epoch == epochs:
                    return mediator
                epoch = epoch+1

            except Exception as err:
                print(err)
                print('Server is not longer available')
                exit()

    def test(self, mediator):
        x_ts, y_ts = DataHandler().load()
        test_loss, test_acc = mediator.evaluate(x_ts,  y_ts)

        print('\nTest accuracy:', test_acc)

if __name__ == "__main__":
    server = Server()
    mediator = server.run()
    server.test(mediator)