import argparse

from huggingface_hub import HfFolder


def parse_args():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    # added checkpoints for traning
    parser.add_argument(
        "--model_check",
        type=str,
        default="KBLab/bert-base-swedish-cased-new",
        help="Model id as checkpoint for training.",
    )

    parser.add_argument(
        "--column",
        type=str,
        default="flatten_chunked_text",
        help="str: Column to train on",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Riksarkivet/test_mini_kbuhist2_v5",
        help="Dataset to train on.",
    )
    # add model id and dataset path argument
    parser.add_argument(
        "--model_id",
        type=str,
        default="Gabriel/bert-base-cased-swedish-1800",
        help="Model id to use for training.",
    )
    parser.add_argument(
        "--repository_id",
        type=str,
        default=None,
        help="Hugging Face Repository id for uploading models",
    )
    # add training hyperparameters for epochs, batch size, learning rate, and seed
    parser.add_argument(
        "--epochs", type=int, default=6, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size to use for training.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size to use for testing.",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5, help="Learning rate to use for training."
    )
    parser.add_argument(
        "--wdecay", type=float, default=0.01, help="Weight decay to use for training."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed to use for training."
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=HfFolder.get_token(),
        help="Token to use for uploading models to Hugging Face Hub.",
    )
    args = parser.parse_known_args()
    return args
