import glob
import os
import datasets
import pandas as pd
from dataclasses import dataclass
import cv2

from src.data_preprocess import ICCAD_Data

pd.options.mode.chained_assignment = None

_REPO = 'https://huggingface.co/datasets/DaJhuan/ICCAD/resolve/main'

_URLS = {
    'fake_data_url': f'{_REPO}/fake-circuit-data_20230623.zip',
    'real_data_url': f'{_REPO}/real-circuit-data_20230615.zip',
    'test_data_url': f'{_REPO}/hidden-real-circuit-data.zip',
    #'BeGAN_01_data_url': f'{_REPO}/BeGAN-ver01.zip',
    #'BeGAN_02_data_url': f'{_REPO}/BeGAN-ver02.zip',
}

@dataclass
class ICCAD_Config(datasets.BuilderConfig):
    test_mode: bool = False
    use_BeGAN: bool = False
    
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
        in_chans = 3 + 9 + 7 + 7

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
        import random
        SAMPLE_PER_SPLIT = 100   # 각 split당 최대 샘플 개수
        SEED = 42                # reproducibility
        test_idx = []
        test_cur = []
        test_pdn = []
        test_dist = []
        test_ir_drop = []
        test_netlist = []
        
        real_idx = []
        real_cur = []
        real_pdn = []
        real_dist = []
        real_ir_drop = []
        real_netlist = []
        
        fake_idx = []
        fake_cur = []
        fake_pdn = []
        fake_dist = []
        fake_ir_drop = []
        fake_netlist = []
        
        
        # Download images
        test_data_files = os.path.join(dl_manager.download_and_extract(_URLS['test_data_url']), 'hidden-real-circuit-data')
        test_path_files = sorted(glob.glob(os.path.join(test_data_files, '*')))
        
        if not self.config.test_mode:
            real_data_files = os.path.join(dl_manager.download_and_extract(_URLS['real_data_url']), 'real-circuit-data_20230615')
            real_path_files = sorted(glob.glob(os.path.join(real_data_files, '*')))
        
        if not self.config.test_mode:
            fake_data_files = os.path.join(dl_manager.download_and_extract(_URLS['fake_data_url']), 'fake-circuit-data_20230623')
            fake_path_files = sorted(glob.glob(os.path.join(fake_data_files, '*.sp')))
        
      
        # for test
        for path in test_path_files:
            data_idx = os.path.basename(path)
            test_idx.append(data_idx)
            data_path = glob.glob(os.path.join(path, '*.*'))
            
            for data in data_path:
                if 'current_map.csv' in os.path.basename(data):
                    test_cur.append(data)
                elif 'eff_dist_map.csv' in os.path.basename(data):
                    test_dist.append(data)
                elif 'ir_drop_map.csv' in os.path.basename(data):
                    test_ir_drop.append(data)
                elif 'pdn_density.csv' in os.path.basename(data):
                    test_pdn.append(data)
                elif 'netlist.sp' in os.path.basename(data):
                    test_netlist.append(data)
                else:
                    raise AssertionError(os.path.basename(data), 'test data path error')
                
            assert len(test_idx) == len(test_cur) == len(test_dist) == len(test_ir_drop) == len(test_pdn) == len(test_netlist), f'{(len(test_idx), len(test_cur), len(test_dist), len(test_ir_drop), len(test_pdn), len(test_netlist))} test data length not the same'
        
        # ────────────── 샘플링 로직 ──────────────
        # helper: idx 리스트와 여러 feature 리스트를 같은 인덱스로 필터
        def sample_lists(idxs, *lists):
            total = len(idxs)
            k = min(SAMPLE_PER_SPLIT, total)
            rng = random.Random(SEED)
            indices = list(range(total))
            rng.shuffle(indices)
            sel = set(indices[:k])
            # filter each list
            return (
                [idxs[i]         for i in sorted(sel)],
                *[
                    [lst[i] for i in sorted(sel)]
                    for lst in lists
                ]
            )
        # test split 샘플링
        (test_idx,
         test_cur,
         test_pdn,
         test_dist,
         test_ir_drop,
         test_netlist) = sample_lists(
        test_idx, test_cur, test_pdn, test_dist, test_ir_drop, test_netlist
        )
        # for real
        if not self.config.test_mode:
            for path in real_path_files:
                data_idx = os.path.basename(path)
                real_idx.append(data_idx)
                data_path = glob.glob(os.path.join(path, '*.*'))
                
                for data in data_path:
                    if 'current_map.csv' in os.path.basename(data):
                        real_cur.append(data)
                    elif 'eff_dist_map.csv' in os.path.basename(data):
                        real_dist.append(data)
                    elif 'ir_drop_map.csv' in os.path.basename(data):
                        real_ir_drop.append(data)
                    elif 'pdn_density.csv' in os.path.basename(data):
                        real_pdn.append(data)
                    elif 'netlist.sp' in os.path.basename(data):
                        real_netlist.append(data)
                    else:
                        raise AssertionError(os.path.basename(data), 'real data path error')
                    
            assert len(real_idx) == len(real_cur) == len(real_dist) == len(real_ir_drop) == len(real_pdn) == len(real_netlist), f'{(len(real_idx), len(real_cur), len(real_dist), len(real_ir_drop), len(real_pdn), len(real_netlist))} real data length not the same'
        # real split 샘플링
            (real_idx,
             real_cur,
             real_pdn,
             real_dist,
             real_ir_drop,
             real_netlist) = sample_lists(
                 real_idx, real_cur, real_pdn, real_dist, real_ir_drop, real_netlist
            )
        # for fake
        if not self.config.test_mode:
            for path in fake_path_files:
                data_idx = os.path.basename(path).split('.')[0]
                fake_idx.append(data_idx)
                data_path = glob.glob(os.path.join(os.path.dirname(path), data_idx + '*.*'))

                for data in data_path:
                    if 'current.csv' in os.path.basename(data):
                        fake_cur.append(data)
                    elif 'eff_dist.csv' in os.path.basename(data):
                        fake_dist.append(data)
                    elif 'ir_drop.csv' in os.path.basename(data):
                        fake_ir_drop.append(data)
                    elif 'pdn_density.csv' in os.path.basename(data):
                        fake_pdn.append(data)
                    elif '.sp' in os.path.basename(data):
                        fake_netlist.append(data)
                    else:
                        raise AssertionError(os.path.basename(data), 'fake data path error')

            assert len(fake_idx) == len(fake_cur) == len(fake_dist) == len(fake_ir_drop) == len(fake_pdn) == len(fake_netlist), f'{(len(fake_idx), len(fake_cur), len(fake_dist), len(fake_ir_drop), len(fake_pdn), len(fake_netlist))} fake data length not the same'
            # fake split 샘플링
            (fake_idx,
             fake_cur,
             fake_pdn,
             fake_dist,
             fake_ir_drop,
             fake_netlist) = sample_lists(
                 fake_idx, fake_cur, fake_pdn, fake_dist, fake_ir_drop, fake_netlist
            )
        if self.config.test_mode:
            return [datasets.SplitGenerator(
                    name=datasets.Split('test'),
                    gen_kwargs={
                        'data_idx': test_idx,
                        'current': test_cur,
                        'pdn_density': test_pdn,
                        'eff_dist': test_dist,
                        'ir_drop': test_ir_drop,
                        'netlist': test_netlist,
                    })]
        else:
            return [datasets.SplitGenerator(
                    name=datasets.Split('fake'),
                    gen_kwargs={
                        'data_idx': fake_idx,
                        'current': fake_cur,
                        'pdn_density': fake_pdn,
                        'eff_dist': fake_dist,
                        'ir_drop': fake_ir_drop,
                        'netlist': fake_netlist,
                    })
                ] + [datasets.SplitGenerator(
                    name=datasets.Split('real'),
                    gen_kwargs={
                        'data_idx': real_idx,
                        'current': real_cur,
                        'pdn_density': real_pdn,
                        'eff_dist': real_dist,
                        'ir_drop': real_ir_drop,
                        'netlist': real_netlist,
                    })
                ] + [datasets.SplitGenerator(
                    name=datasets.Split('test'),
                    gen_kwargs={
                        'data_idx': test_idx,
                        'current': test_cur,
                        'pdn_density': test_pdn,
                        'eff_dist': test_dist,
                        'ir_drop': test_ir_drop,
                        'netlist': test_netlist,
                    })
                ]
    
    def _generate_examples(self, data_idx, current, pdn_density, eff_dist, ir_drop, netlist):
        self.preprocess = ICCAD_Data(
            img_size = self.config.img_size,
            interpolation = self.config.interpolation,
        )
        
        for idx, (_data_idx, _current, _pdn_density, _eff_dist, _ir_drop, _netlist) in enumerate(zip(data_idx, current, pdn_density, eff_dist, ir_drop, netlist)):
            yield idx, self.preprocess.generate_example(_data_idx, _current, _pdn_density, _eff_dist, _ir_drop, _netlist)
