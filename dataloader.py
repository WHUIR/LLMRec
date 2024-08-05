from torch.utils.data import DataLoader, Dataset
from data_sequential import DataSequential
from torch.utils.data.distributed import DistributedSampler


def get_dataloader(args, tokenizer):
    train_dataset = DataSequential(args, tokenizer, 'train')
    val_dataset = DataSequential(args, tokenizer, 'valid')
    test_dataset = DataSequential(args, tokenizer, 'test')
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        valid_sampler = DistributedSampler(val_dataset)
        test_sampler = DistributedSampler(test_dataset)
    else:
        train_sampler, valid_sampler, test_sampler = None, None, None

    if args.distributed:
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  collate_fn=train_dataset.collate_fn,
                                  shuffle=False,
                                  pin_memory=True,
                                  sampler=train_sampler)
        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size * 4,
                                collate_fn=train_dataset.collate_fn,
                                shuffle=False,
                                pin_memory=True,
                                sampler=valid_sampler)
        test_loader = DataLoader(test_dataset,
                                 batch_size=args.batch_size * 4,
                                 collate_fn=train_dataset.collate_fn,
                                 shuffle=False,
                                 pin_memory=True,
                                 sampler=test_sampler)

    else:
        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  # num_workers=args.num_works,
                                  collate_fn=train_dataset.collate_fn,
                                  shuffle=True
                                  )
        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size * 4,
                                shuffle=False,
                                collate_fn=val_dataset.collate_fn
                                )
        test_loader = DataLoader(test_dataset,
                                 batch_size=args.batch_size * 4,
                                 shuffle=False,
                                 collate_fn=test_dataset.collate_fn
                                 )
    return train_loader, val_loader, test_loader
