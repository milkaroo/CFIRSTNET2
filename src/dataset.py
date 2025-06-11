import glob
import os
import datasets
import pandas as pd
from dataclasses import dataclass
import cv2

from src.data_preprocess import ICCAD_Data

pd.options.mode.chained_assignment = None

#_REPO = 'https://huggingface.co/datasets/DaJhuan/ICCAD/resolve/main'
_REPO = 'https://huggingface.co/datasets/milkaroo0/CFIRSTNET_dataset/resolve/main'

_URLS = {
    'fake_data_url': f'{_REPO}/fake-circuit-data_20230623.zip',
    'real_data_url': f'{_REPO}/real-circuit-data_20230615.zip',
    'test_data_url': f'{_REPO}/hidden-real-circuit-data.zip',
    'BeGAN_01_data_url': f'{_REPO}/BeGAN-ver01_part.zip',
    'BeGAN_02_data_url': f'{_REPO}/BeGAN-ver02_half.zip',
}

@dataclass
class ICCAD_Config(datasets.BuilderConfig):
    test_mode: bool = False
    use_BeGAN: bool = False and not test_mode
    
    # transform
    img_size: int = 384
    interpolation: int = cv2.INTER_AREA

class ICCAD_Dataset(datasets.GeneratorBasedBuilder):
    DEFAULT_WRITER_data_SIZE = 10
    
    BUILDER_CONFIG_CLASS = ICCAD_Config
    BUILDER_CONFIGS = [
        ICCAD_Config(
            name='CFIRSTNET',
            version=datasets.Version('6.1.0', 'quick min dist version'),
            description='CFIRSTNET: Comprehensive Features for Static IR Drop Estimation with Neural Network',
        )
    ]
    
    def _info(self):
        #in_chans = 3 + 9 + 7 + 7
        in_chans = 9 + 7 + 7
        
        features = datasets.Features({
            'data_idx': datasets.Value('string'),
            'H': datasets.Value('int32'),
            'W': datasets.Value('int32'),
            'image': datasets.Array3D((in_chans, self.config.img_size, self.config.img_size), dtype='float32'),
            'ir_drop': datasets.Array2D((None, 1), dtype='float32'),
        })

        return datasets.DatasetInfo(
            features=features,
        )
       
    def _split_generators(self, dl_manager):
        test_idx = []
        test_ir_drop = []
        test_netlist = []
        
        real_idx = []
        real_ir_drop = []
        real_netlist = []
        
        fake_idx = []
        fake_ir_drop = []
        fake_netlist = []

        BeGAN_01_idx = []
        BeGAN_01_ir_drop = []
        BeGAN_01_netlist = []
        
        BeGAN_02_idx = []
        BeGAN_02_ir_drop = []
        BeGAN_02_netlist = []
        
        # Download images
        test_data_files = os.path.join(dl_manager.download_and_extract(_URLS['test_data_url']), 'hidden-real-circuit-data')
        test_path_files = sorted(glob.glob(os.path.join(test_data_files, '*')))
        
        if not self.config.test_mode:
            real_data_files = os.path.join(dl_manager.download_and_extract(_URLS['real_data_url']), 'real-circuit-data_20230615')
            real_path_files = sorted(glob.glob(os.path.join(real_data_files, '*')))
        
        if not self.config.test_mode:
            fake_data_files = os.path.join(dl_manager.download_and_extract(_URLS['fake_data_url']), 'fake-circuit-data_20230623')
            fake_path_files = sorted(glob.glob(os.path.join(fake_data_files, '*.sp')))
        
        if self.config.use_BeGAN and not self.config.test_mode:
            BeGAN_01_data_files = os.path.join(dl_manager.download_and_extract(_URLS['BeGAN_01_data_url']), 'BeGAN-ver01_part')
            BeGAN_01_path_files = sorted(glob.glob(os.path.join(BeGAN_01_data_files, '*.sp')))

            BeGAN_02_data_files = os.path.join(dl_manager.download_and_extract(_URLS['BeGAN_02_data_url']), 'BeGAN-ver02_half')
            BeGAN_02_path_files = sorted(glob.glob(os.path.join(BeGAN_02_data_files, '*.sp')))
            #BeGAN_02_path_files = sorted(glob.glob(os.path.join(BeGAN_02_data_files, '*.sp')))[:200]  # 최대 300개만 사용

        # for test
        for path in test_path_files:
            data_idx = os.path.basename(path)
            test_idx.append(data_idx)
            data_path = glob.glob(os.path.join(path, '*.*'))
            
            for data in data_path:
                if 'ir_drop_map.csv' in os.path.basename(data):
                    test_ir_drop.append(data)
                elif 'netlist.sp' in os.path.basename(data):
                    test_netlist.append(data)
                else:
                    raise AssertionError(os.path.basename(data), 'test data path error')
                
            assert len(test_idx) == len(test_ir_drop) == len(test_netlist), f'{(len(test_idx), len(test_ir_drop), len(test_netlist))} test data length not the same'
        
        # for real
        if not self.config.test_mode:
            for path in real_path_files:
                data_idx = os.path.basename(path)
                real_idx.append(data_idx)
                data_path = glob.glob(os.path.join(path, '*.*'))
                
                for data in data_path:
                    if 'ir_drop_map.csv' in os.path.basename(data):
                        real_ir_drop.append(data)
                    elif 'netlist.sp' in os.path.basename(data):
                        real_netlist.append(data)
                    else:
                        raise AssertionError(os.path.basename(data), 'real data path error')
                    
            assert len(real_idx) == len(real_ir_drop) == len(real_netlist), f'{(len(real_idx), len(real_ir_drop), len(real_netlist))} real data length not the same'
        
        # for fake
        if not self.config.test_mode:
            for path in fake_path_files:
                data_idx = os.path.basename(path).split('.')[0]
                fake_idx.append(data_idx)
                data_path = glob.glob(os.path.join(os.path.dirname(path), data_idx + '*.*'))

                for data in data_path:
                    if 'ir_drop.csv' in os.path.basename(data):
                        fake_ir_drop.append(data)
                    elif '.sp' in os.path.basename(data):
                        fake_netlist.append(data)
                    else:
                        raise AssertionError(os.path.basename(data), 'fake data path error')

            assert len(fake_idx) == len(fake_ir_drop) == len(fake_netlist), f'{(len(fake_idx), len(fake_ir_drop), len(fake_netlist))} fake data length not the same'

        if self.config.use_BeGAN and not self.config.test_mode:
            # for BeGAN-ver01
            for path in BeGAN_01_path_files:
                data_idx = os.path.basename(path).split('.')[0]
                BeGAN_01_idx.append(data_idx)
                data_path = glob.glob(os.path.join(os.path.dirname(path), data_idx + '*.*'))

                for data in data_path:
                    if 'ir_drop_map.csv' in os.path.basename(data):
                        BeGAN_01_ir_drop.append(data)
                    elif '.sp' in os.path.basename(data):
                        BeGAN_01_netlist.append(data)
                    else:
                        raise AssertionError(os.path.basename(data), 'BeGAN-ver01 data path error')

            assert len(BeGAN_01_idx) == len(BeGAN_01_ir_drop) == len(BeGAN_01_netlist), f'{(len(BeGAN_01_idx), len(BeGAN_01_ir_drop), len(BeGAN_01_netlist))} BeGAN-ver02 data length not the same'

            # for BeGAN-ver02
            for path in BeGAN_02_path_files:
                data_idx = os.path.basename(path).split('.')[0]
                BeGAN_02_idx.append(data_idx)
                data_path = glob.glob(os.path.join(os.path.dirname(path), data_idx + '*.*'))

                for data in data_path:
                    if 'voltage.csv' in os.path.basename(data):
                        BeGAN_02_ir_drop.append(data)
                    elif '.sp' in os.path.basename(data):
                        BeGAN_02_netlist.append(data)
                    else:
                        raise AssertionError(os.path.basename(data), 'BeGAN-ver01 data path error')

            assert len(BeGAN_02_idx) == len(BeGAN_02_ir_drop) == len(BeGAN_02_netlist), f'{(len(BeGAN_02_idx), len(BeGAN_02_ir_drop), len(BeGAN_02_netlist))} BeGAN-ver01 data length not the same'
            
        if self.config.test_mode:
            return [datasets.SplitGenerator(
                    name=datasets.Split('test'),
                    gen_kwargs={
                        'data_idx': test_idx,
                        'ir_drop': test_ir_drop,
                        'netlist': test_netlist,
                    })]
        else:
            return ([ datasets.SplitGenerator(
                    name=datasets.Split('BeGAN_01'),
                    gen_kwargs={
                        'data_idx': BeGAN_01_idx,
                        'ir_drop': BeGAN_01_ir_drop,
                        'netlist': BeGAN_01_netlist,
                    }),datasets.SplitGenerator(
                    name=datasets.Split('BeGAN_02'),
                    gen_kwargs={
                        'data_idx': BeGAN_02_idx,
                        'ir_drop': BeGAN_02_ir_drop,
                        'netlist': BeGAN_02_netlist,
                    })
                ] if self.config.use_BeGAN else []
                ) + [datasets.SplitGenerator(
                    name=datasets.Split('fake'),
                    gen_kwargs={
                        'data_idx': fake_idx,
                        'ir_drop': fake_ir_drop,
                        'netlist': fake_netlist,
                    })
                ] + [datasets.SplitGenerator(
                    name=datasets.Split('real'),
                    gen_kwargs={
                        'data_idx': real_idx,
                        'ir_drop': real_ir_drop,
                        'netlist': real_netlist,
                    })
                ] + [datasets.SplitGenerator(
                    name=datasets.Split('test'),
                    gen_kwargs={
                        'data_idx': test_idx,
                        'ir_drop': test_ir_drop,
                        'netlist': test_netlist,
                    })
                ]
    
    def _generate_examples(self, data_idx, ir_drop, netlist):
        self.preprocess = ICCAD_Data(
            img_size = self.config.img_size,
            interpolation = self.config.interpolation,
        )
        
        for idx, (_data_idx, _ir_drop, _netlist) in enumerate(zip(data_idx, ir_drop, netlist)):
            yield idx, self.preprocess.generate_example(_data_idx, _ir_drop, _netlist)
