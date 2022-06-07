import socket
import pickle
from src.model import Model
from data.data_handler import DataHandler

class Client:

    def __init__(self, IP, PORT, HEADER_LENGTH, number_of_parties, username, batch_size) -> None:
        
        self.HEADER_LENGTH = HEADER_LENGTH
        self.IP = IP
        self.PORT = PORT
        self.username = str(username)
        self.BUFFERSIZE = 1024
        self.batchsize = batch_size

        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((IP, PORT))
        self.client_socket.setblocking(True)

        self.datahandler = DataHandler(username,  batch_size, number_of_parties)

        print("Client initialization is completed on IP: {0}, Port: {1}".format(IP, PORT))

    def initialize_party(self):
        return Model()

    def subscribe_server(self, username):
        username = username.encode('utf-8')
        username_header = f"{len(username):<{self.HEADER_LENGTH}}".encode('utf-8')
        self.client_socket.send(username_header + username)

    def recv_weights(self):
        message_header = self.client_socket.recv(self.HEADER_LENGTH)
        if not len(message_header):
            return False
    
        message_length = int(message_header.decode('utf-8').strip())

        received_data = b""
        current_length = 0
        
        while current_length < message_length:
            received_data += self.client_socket.recv(message_length - current_length)
            current_length = len(received_data)
            
        weights = pickle.loads(received_data)
        return weights

class SGDClient(Client):

    def __init__(self, IP=socket.gethostname(), PORT=12345,
                    HEADER_LENGTH=10, number_of_parties=2, username=1, batch_size=32):
        super().__init__(IP, PORT, HEADER_LENGTH, number_of_parties, username, batch_size)

    def send_gradient(self, grads):

        serialized_data = pickle.dumps(grads)

        message_header = f"{len(serialized_data):<{self.HEADER_LENGTH}}".encode('utf-8')

        self.client_socket.sendall(message_header + serialized_data)

    def run(self):

        # create party model
        party = self.initialize_party()

        # subscribe server
        self.subscribe_server(self.username)

        # client waits for server request
        # time = 0
        while True:
            try:
                x_batch_train, y_batch_train = self.datahandler.batch()

                # client gets mediator weights
                weights = self.recv_weights()

                # client assigns weights to model
                party.set_weights(weights)

                # calculate loss values in time t
                # loss = party.loss(x_batch_train, y_batch_train)
                # print('Time :{0}, Loss value: {1}'.format(time, loss))
                # time += 1
                
                # feed data and get gradients
                gradients = party.get_gradient(x_batch_train, y_batch_train)

                # client sends gradient
                self.send_gradient(gradients)

            except Exception as err:
                print(err)
                print('Server is not longer available')
                exit()

class AVGClient(Client):

    def __init__(self, IP=socket.gethostname(), PORT=12345,
                    HEADER_LENGTH=10, number_of_parties=2, username=1, batch_size=32, epochs=1):
        
        self.epochs = epochs
        super().__init__(IP, PORT, HEADER_LENGTH, number_of_parties, username, batch_size)
        

    def send_weights(self, weights):

        serialized_data = pickle.dumps(weights)

        message_header = f"{len(serialized_data):<{self.HEADER_LENGTH}}".encode('utf-8')

        self.client_socket.sendall(message_header + serialized_data)

    def run(self):

        # create party model
        party = self.initialize_party()

        # subscribe server
        self.subscribe_server(self.username)

        # client waits for server request
        # time = 0
        while True:
            try:
                # client gets mediator weights
                weights = self.recv_weights()

                # client assigns weights to model
                party.set_weights(weights)

                # calculate loss values in time t
                # loss = party.loss(x_batch_train, y_batch_train)
                # print('Time :{0}, Loss value: {1}'.format(time, loss))
                # time += 1

                # updating loop
                print(self.epochs)
                for e in range(0,self.epochs):
                    print('epochs : {}'.format(e))
                    for it in range(0,int(30000/self.batchsize)):
                        x_batch_train, y_batch_train = self.datahandler.batch()
                        party.model.train_on_batch(x_batch_train, y_batch_train)

                #
                cweights = party.model.get_weights()

                # client sends gradient
                self.send_weights(cweights)

            except Exception as err:
                print(err)
                print('Server is not longer available')
                exit()

if __name__ == "__main__":
    
    client = AVGClient(username=2)
    client.run()
