"""Optional training-data prefetch, with unchanged default loader semantics."""

from numbers import Integral

from torch.utils.data import DataLoader

from dataloader import mil_collate_fn


def make_loaders(datasets, conf, device, train_workers=0):
    if (not isinstance(train_workers, Integral) or isinstance(train_workers, bool)
            or train_workers not in [0, 1]):
        raise ValueError('train_workers must be 0 or 1.')
    loaders = []
    for index, dataset in enumerate(datasets):
        workers = int(train_workers) if index == 0 else 0
        extra = {'multiprocessing_context': 'spawn', 'prefetch_factor': 2} if workers else {}
        # Nonpersistent workers preserve the ordinary per-epoch base-seed draw.
        loaders.append(DataLoader(
            dataset, batch_size=int(conf.B), shuffle=index == 0, num_workers=workers,
            pin_memory=device == 'cuda', collate_fn=mil_collate_fn, **extra,
        ))
    return loaders
