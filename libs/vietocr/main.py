import matplotlib.pyplot as plt
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer

config = Cfg.load_config_from_name('vgg_transformer')
# config['weights'] = './weights/transformerocr.pth'
config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
config['cnn']['pretrained'] = False
config['device'] = 'cuda:0'
config['predictor']['beamsearch'] = False

dataset_params = {
    'name':'hw',
    'data_root':'./text_reg_dataset/',
    'train_annotation':'train_line_annotation.txt',
    'valid_annotation':'test_line_annotation.txt',
    'image_height':64
}

params = {
         'print_every': 200,
         'valid_every': 15 * 200,
          'iters': 200000,
          'checkpoint': './checkpoint/transformerocr_checkpoint.pth',    
          'export': './weights/transformerocr.pth',
          'metrics': 10,
          'batch_size': 16
}

config['trainer'].update(params)
config['dataset'].update(dataset_params)
config['device'] = 'cuda:0'

trainer = Trainer(config, pretrained=True)
trainer.config.save('config.yml')
trainer.train()
trainer.precision()