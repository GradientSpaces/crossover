from common import misc
from preprocess import build

from omegaconf import OmegaConf
import hydra
import logging as log

log.getLogger().setLevel(log.INFO)
log.basicConfig(level=log.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')

@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(cfg: OmegaConf) -> None:
    datasets = misc.rgetattr(cfg, "data.sources")
    
    task_name = misc.rgetattr(cfg, "task.name")
    task_config = misc.rgetattr(cfg.task, task_name)
    
    assert task_name in ['Preprocess', 'PreprocessMultimodal']
    splits = misc.rgetattr(task_config, "splits")
    
    message = "Task -- {}, Datasets -- {}".format(task_name, datasets)
    log.info(message)
    
    for dataset in datasets:
        message = "Currently processing - {}".format(dataset)
        data_config = misc.rgetattr(cfg, "data.{}".format(dataset))
        
        if task_name == 'PreprocessMultimodal':
            modality_config = cfg.modality_info
            processor_name = task_config.processor
            
        if task_name == 'Preprocess':
            task_modality = misc.rgetattr(task_config, "modality")
            modality_config = misc.rgetattr(cfg.modality_info, task_modality)
            processor_name   = misc.rgetattr(data_config, 'processor{}'.format(task_modality))
            message += " Modality - {}".format(task_modality)
        
        for split in splits:
            message += " Split - {}".format(split)
            log.info(message)
            
            process_module = build.build_processor(processor_name, data_config, modality_config, split)
            process_module.run()
    
if __name__ == '__main__':
    main()