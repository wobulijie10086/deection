import argparse


def parameter_parser():
    # Experiment parameters
    parser = argparse.ArgumentParser(description='Smart Contracts Reentrancy Detection')

    parser.add_argument('-D', '--dataset', type=str, default='train_data/reentrancy_code_snippets_2000.txt',
                        choices=['train_data/infinite_loop_1317.txt', 'train_data/reentrancy_1671.txt'
                                 'train_data/SmartContract.txt'])
    parser.add_argument('-M', '--model', type=str, default='BLSTM',
                        choices=['BLSTM', 'BLSTM_Attention', 'LSTM_Model', 'GRU-SVM', 'GRU_Model', "GRU-Softmax", "Simple_RNN"])
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-d', '--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--vector_dim', type=int, default=300, help='dimensions of vector')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('-th', '--threshold', type=float, default=0.5, help='threshold')

    return parser.parse_args()
