#/usr/bin/python

from ann import ann
from mnist_loader import load_data_wrapper


train_d, valid_d, test_d = load_data_wrapper()

HLS = [10, 25, 50]
ETA = [0.5, 0.25, 0.125]
def train_1_hidden_layer_anns(hls=HLS, eta=ETA, mini_batch_size=10, num_epochs=10):
    for h in hls:
        for n in eta:
            network = ann([784, h, 10])
            print(f"*** Training 784x{h}x10 ANN with eta={n}")
            network.mini_batch_sgd(train_d, num_epochs, mini_batch_size, n, test_data=test_d)
            print(f"*** Training 784x{h}x10 ANN DONE...\n\n")


def train_2_hidden_layer_anns(hls=HLS, eta=ETA, mini_batch_size=10, num_epochs=10):
    for h in hls:
        for h_2 in hls:
            for n in eta:
                network = ann([784, h, h_2, 10])
                print(f"*** Training 784x{h}x{h_2}x10 ANN with eta={n}")
                network.mini_batch_sgd(train_d, num_epochs, mini_batch_size, n, test_data=test_d)
                print(f"*** Training 784x{h}x{h_2}x10 ANN DONE...\n\n")

### Uncomment to run
if __name__ == '__main__':
    train_1_hidden_layer_anns(hls=HLS, eta=ETA, mini_batch_size=10, num_epochs=10)
    train_2_hidden_layer_anns(hls=HLS, eta=ETA, mini_batch_size=10, num_epochs=10)
