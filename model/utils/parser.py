import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run MulSetRank.")

    parser.add_argument('--data_path', nargs='?', default='../data/',
                        help='Input data path.')

    parser.add_argument('--dataset', nargs='?', default='citeulike',
                        help='Choose a dataset.')

    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')

    parser.add_argument('--epoch', type=int, default=400,
                        help='Number of epoch.')

    parser.add_argument('--embed_size', type=int, default=256,
                        help='Embedding size.')

    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')

    parser.add_argument('--regs', nargs='?', default='[1e-5]',
                        help='Regularizations.')

    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')

    parser.add_argument('--k', type=int, default=5,
                        help='Size of item set.')

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')

    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]',
                        help='Output sizes of every layer')

    return parser.parse_args()
