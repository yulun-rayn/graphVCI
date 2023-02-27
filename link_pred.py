from gvci.train.link_pred import link_pred

import argparse

def parse_arguments():
    """
    Read arguments if this script is called from a terminal.
    """

    parser = argparse.ArgumentParser()

    # setting arguments
    parser.add_argument('--name', default='default_run')
    parser.add_argument("--graph_path", type=str, required=True)
    parser.add_argument("--artifact_path", type=str, required=True)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--gpu', default='0')

    # model arguments
    parser.add_argument("--graph_mode", type=str, default="sparse", help='dense;sparse')
    parser.add_argument("--hparams", type=str, default="link_hparams.json")

    # training arguments
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--max_minutes", type=int, default=600)
    parser.add_argument("--checkpoint_freq", type=int, default=10)

    return dict(vars(parser.parse_args()))


if __name__ == "__main__":
    link_pred(parse_arguments())
