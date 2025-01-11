from fvcore.common.registry import Registry
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from omegaconf import DictConfig
from typing import Union
from common import misc

DATASET_REGISTRY = Registry("dataset")

def get_dataset(cfg: DictConfig, split: str, is_training: bool = True) -> Union[ConcatDataset, Dataset]:
    """Constructs and returns dataset(s) based on configuration and split."""
    
    task_name = misc.rgetattr(cfg, "task.name")
    task_config = misc.rgetattr(cfg.task, task_name)
    dataset_names = misc.rgetattr(task_config, split)
    
    if is_training:
        dataset_list = []
        for dataset_name in dataset_names:
            _dataset = DATASET_REGISTRY.get(dataset_name)(misc.rgetattr(cfg.data, dataset_name.replace('Object', '')), split)
            dataset_list.append(_dataset)

        print(split, ' ', '='*50)
        print('Dataset\t\t\tSize')
        total = sum([len(dataset) for dataset in dataset_list])
        for dataset_name, dataset in zip(dataset_names, dataset_list):
            print(f'{dataset_name:<20} {len(dataset):>6} ({len(dataset) / total * 100:.1f}%)')
        print(f'Total\t\t\t{total}')
        print('='*50)
        
        dataset_list = ConcatDataset(dataset_list)
        return dataset_list

    else:
        assert len(dataset_names) == 1, 'Trying to run eval on multiple datasets!'
        dataset = DATASET_REGISTRY.get(dataset_names[0])(misc.rgetattr(cfg.data, dataset_names[0].replace('Object', '')), split)
        return dataset

def build_dataloader(cfg: DictConfig, split: str = 'train', is_training: bool = True) -> DataLoader:
    """Creates a DataLoader with specified configuration and dataset."""
    if cfg.dataloader.num_workers == 0:
        drop_last = True
    else:
        drop_last = True if split == 'train' else False
    
    shuffle = True if split == 'train' else False
   
    if is_training:
        dataset = get_dataset(cfg, split)
        return DataLoader(dataset,
                        batch_size=cfg.dataloader.batch_size,
                        num_workers=cfg.dataloader.num_workers,
                        collate_fn=getattr(dataset.datasets[0], 'collate_fn', None),
                        pin_memory=True, 
                        persistent_workers=True if cfg.dataloader.num_workers > 0 else False,
                        shuffle=shuffle,
                        drop_last=drop_last)
    
    else:
        dataset = get_dataset(cfg, split, is_training=False)
        return DataLoader(dataset,
                        batch_size=cfg.dataloader.batch_size,
                        num_workers=cfg.dataloader.num_workers,
                        collate_fn=getattr(dataset, 'collate_fn', None),
                        pin_memory=True, 
                        persistent_workers=True if cfg.dataloader.num_workers > 0 else False,
                        shuffle=shuffle,
                        drop_last=drop_last)
        

if __name__ == '__main__':
    pass