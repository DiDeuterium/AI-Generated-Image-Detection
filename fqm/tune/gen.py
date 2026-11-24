"""生成调参用的配置与清单（每个配置一个 JSON，清单每行一个路径）。

用法（仓库根目录）：
    python fqm/tune/gen.py --wave 1          # 写入 fqm/tune/configs/w1_*.json 与 manifest_w1.txt
    然后：sbatch fqm/slurm/tune_efficientvim.sbatch fqm/tune/manifest_w1.txt
"""

import argparse
import copy
import json
from pathlib import Path

TUNE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = TUNE_DIR / 'configs'

BASE = {
    'data': {
        'splits': 'splits/train_val.csv',
        'image_dir': 'dataset',
        'input_size': 224,
        'resize_size': 256,
        'num_workers': 5,
        'norm_stats': 'fqm/configs/norm_stats.json',
        'jpeg_cache': 'fqm/cache/val_jpeg_q75',
        'jpeg_quality': 75,
        'jpeg_subsampling': 2,
        'aug': {'mode': 'protocol', 'hflip': 0.5},
    },
    'model': {'name': 'efficientvim_m1', 'num_classes': 2, 'drop_path_rate': 0.0},
    'train': {
        'seed': 42,
        'epochs': 30,
        'updates_per_epoch': 100,
        'micro_batch': 32,
        'accum_steps': 2,
        'lr': 0.0003,
        'lr_start': 3e-06,
        'lr_final': 3e-06,
        'warmup_updates': 200,
        'weight_decay': 0.01,
        'betas': [0.9, 0.999],
        'eps': 1e-08,
        'label_smoothing': 0.0,
        'loss': 'ce',
        'mixup_alpha': 0.0,
        'cutmix_alpha': 0.0,
        'mix_prob': 1.0,
        'ema_decay': 0.0,
        'amp': True,
        'amp_dtype': 'float16',
    },
    'output': {'root': 'fqm/runs/tune', 'run_name': 'unnamed'},
}


def variant(name, data=None, model=None, train=None, drop_data=()):
    cfg = copy.deepcopy(BASE)
    cfg['output']['run_name'] = name
    for key in drop_data:
        cfg['data'].pop(key, None)
    cfg['data'].update(data or {})
    cfg['model'].update(model or {})
    cfg['train'].update(train or {})
    return cfg


def wave1():
    return [
        variant('w1_control'),
        variant('w1_wd05_ls01', train={'weight_decay': 0.05, 'label_smoothing': 0.1}),
        variant('w1_dp015_wd05',
                model={'drop_path_rate': 0.15}, train={'weight_decay': 0.05}),
        variant('w1_rrc_jitter', data={'aug': {
            'mode': 'rrc', 'rrc_scale': [0.5, 1.0], 'hflip': 0.5,
            'color_jitter': {'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2}}},
            train={'weight_decay': 0.05}),
        variant('w1_long60', train={'epochs': 60}),
        variant('w1_ema', train={'ema_decay': 0.999, 'weight_decay': 0.05,
                                 'label_smoothing': 0.05}),
        variant('w1_m2', model={'embed_dim': [128, 256, 512],
                                'drop_path_rate': 0.1},
                train={'weight_decay': 0.05}),
        variant('w1_native', data={'aug': {'mode': 'native', 'hflip': 0.5}},
                train={'weight_decay': 0.05}),
    ]


def wave2():
    """第二轮：以首轮最优的 native（原尺度裁剪）为主，叠加正则化、预算与容量实验。"""
    native = {'mode': 'native', 'hflip': 0.5}
    base_train = {'weight_decay': 0.05, 'label_smoothing': 0.05, 'ema_decay': 0.999}
    return [
        variant('w2_native_ema', data={'aug': native}, train=base_train),
        variant('w2_native_long60', data={'aug': native},
                train={**base_train, 'epochs': 60}),
        variant('w2_native_256', data={'aug': native, 'input_size': 256,
                                       'resize_size': 288}, train=base_train),
        variant('w2_native_m2', data={'aug': native},
                model={'embed_dim': [128, 256, 512], 'drop_path_rate': 0.1},
                train=base_train),
        variant('w2_native_jpegaug', data={'aug': {**native, 'jpeg_aug': {
            'prob': 0.3, 'quality_min': 30, 'quality_max': 90}}}, train=base_train),
        variant('w2_native_mixup', data={'aug': native},
                train={**base_train, 'mixup_alpha': 0.2, 'epochs': 60}),
        variant('w2_native_lr1e3', data={'aug': native},
                train={**base_train, 'lr': 0.001}),
        variant('w2_native_ls0', data={'aug': native},
                train={**base_train, 'label_smoothing': 0.0}),
    ]


def wave3():
    """第三轮：在第二轮最优组合上继续加大输入尺寸、预算与容量（按前两轮结果更新）。"""
    native = {'mode': 'native', 'hflip': 0.5}
    base = {'weight_decay': 0.05, 'label_smoothing': 0.05, 'ema_decay': 0.999}
    return [
        variant('w3_native320', data={'aug': native, 'input_size': 320,
                                      'resize_size': 352}, train=base),
        variant('w3_native256_long', data={'aug': native, 'input_size': 256,
                                           'resize_size': 288},
                train={**base, 'epochs': 60}),
        variant('w3_native320_long', data={'aug': native, 'input_size': 320,
                                           'resize_size': 352},
                train={**base, 'epochs': 60}),
        variant('w3_native_m2_256', data={'aug': native, 'input_size': 256,
                                          'resize_size': 288},
                model={'embed_dim': [128, 256, 512], 'drop_path_rate': 0.1}, train=base),
        variant('w3_native_noflip', data={'aug': {'mode': 'native', 'hflip': 0.0}},
                train=base),
        variant('w3_native_lr5e4', data={'aug': native}, train={**base, 'lr': 0.0005}),
        variant('w3_native_wd02', data={'aug': native},
                train={**base, 'weight_decay': 0.02}),
        variant('w3_native_ls01', data={'aug': native},
                train={**base, 'label_smoothing': 0.1}),
    ]
def wave4():
    """第四轮：第三轮显示更高分辨率的原尺度裁剪最强，这里沿同一方向再做四组。"""
    native = {'mode': 'native', 'hflip': 0.5}
    base = {'weight_decay': 0.05, 'label_smoothing': 0.05, 'ema_decay': 0.999}
    big = {'aug': native, 'input_size': 320, 'resize_size': 352}
    return [
        variant('w4_native384', data={'aug': native, 'input_size': 384,
                                      'resize_size': 416}, train=base),
        variant('w4_native320_m2', data=big,
                model={'embed_dim': [128, 256, 512], 'drop_path_rate': 0.1}, train=base),
        variant('w4_native320_wd02', data=big,
                train={**base, 'weight_decay': 0.02}),
        variant('w4_native320_ls01', data=big,
                train={**base, 'label_smoothing': 0.1}),
    ]


def wave5():
    """第五轮：压缩/合成伪迹先验模块（CAP）及其消融，全部用原尺度裁剪 320。"""
    native = {'mode': 'native', 'hflip': 0.5}
    base = {'weight_decay': 0.05, 'label_smoothing': 0.05, 'ema_decay': 0.999}
    big = {'aug': native, 'input_size': 320, 'resize_size': 352}
    cap = {'prior': {'parts': ['block', 'highfreq']}}
    return [
        variant('w5_cap30', data=big, model=cap, train=base),
        variant('w5_hf30', data=big, model={'prior': {'parts': ['highfreq']}}, train=base),
        variant('w5_block30', data=big, model={'prior': {'parts': ['block']}}, train=base),
        variant('w5_cap60', data=big, model=cap, train={**base, 'epochs': 60}),
        variant('w5_cap_detach30', data=big,
                model={'prior': {'parts': ['block', 'highfreq'], 'detach': True}},
                train=base),
        variant('w5_cap_block7_30', data=big,
                model={'prior': {'parts': ['block'], 'block': 7}}, train=base),
        variant('w5_cap_ls01_30', data=big, model=cap,
                train={**base, 'label_smoothing': 0.1}),
        variant('w5_cap_m2_30', data=big,
                model={'prior': {'parts': ['block', 'highfreq']},
                       'embed_dim': [128, 256, 512], 'drop_path_rate': 0.1}, train=base),
    ]


def wave6():
    """第六轮：CAP 的完整消融（60 epoch）与压缩偏置缓解实验。"""
    native = {'mode': 'native', 'hflip': 0.5}
    base = {'weight_decay': 0.05, 'label_smoothing': 0.05, 'ema_decay': 0.999, 'epochs': 60}
    cap = {'prior': {'parts': ['block', 'highfreq']}}
    return [
        variant('w6_cap60_hfonly', data={'aug': native, 'input_size': 320,
                                         'resize_size': 352},
                model={'prior': {'parts': ['highfreq']}}, train=base),
        variant('w6_cap60_jpegaug', data={'aug': {**native, 'jpeg_aug': {
            'prob': 0.2, 'quality_min': 40, 'quality_max': 95}}, 'input_size': 320,
            'resize_size': 352}, model=cap, train=base),
        variant('w6_cap60_ls01', data={'aug': native, 'input_size': 320,
                                       'resize_size': 352}, model=cap,
                train={**base, 'label_smoothing': 0.1}),
        variant('w6_cap60_256', data={'aug': native, 'input_size': 256,
                                      'resize_size': 288}, model=cap, train=base),
    ]


def wave7():
    """第七轮：测试集几乎全为 JPEG，需削弱「编码格式」捷径，改学内容/平滑度线索。"""
    native = {'mode': 'native', 'hflip': 0.5}
    base = {'weight_decay': 0.05, 'label_smoothing': 0.05, 'ema_decay': 0.999, 'epochs': 60}
    cap = {'prior': {'parts': ['block', 'highfreq']}}
    j10 = {'prob': 1.0, 'quality_min': 40, 'quality_max': 95}
    j05 = {'prob': 0.5, 'quality_min': 40, 'quality_max': 95}
    return [
        variant('w7_cap_jpeg10', data={'aug': {**native, 'jpeg_aug': j10},
                                       'input_size': 320, 'resize_size': 352},
                model=cap, train=base),
        variant('w7_cap_jpeg05', data={'aug': {**native, 'jpeg_aug': j05},
                                       'input_size': 320, 'resize_size': 352},
                model=cap, train=base),
        variant('w7_nocap_jpeg10', data={'aug': {**native, 'jpeg_aug': j10},
                                         'input_size': 320, 'resize_size': 352},
                train=base),
        variant('w7_cap_jpeg10_ls01', data={'aug': {**native, 'jpeg_aug': j10},
                                            'input_size': 320, 'resize_size': 352},
                model=cap, train={**base, 'label_smoothing': 0.1}),
    ]


def wave8():
    """第八轮：测试集是「整批另存 q75、4:2:0」，因此把训练增强固定为 q75（p=1.0），
    使训练分布与测试构造一致；同时排 4 个种子用于集成。选模按 q75 验证集 AUROC。"""
    native = {'mode': 'native', 'hflip': 0.5}
    base = {'weight_decay': 0.05, 'label_smoothing': 0.05, 'ema_decay': 0.999,
            'epochs': 60, 'selection': 'jpeg_auroc'}
    cap = {'prior': {'parts': ['block', 'highfreq']}}
    fix75 = {'prob': 1.0, 'quality_min': 75, 'quality_max': 75}
    big = {'aug': {**native, 'jpeg_aug': fix75}, 'input_size': 320, 'resize_size': 352}
    return [
        variant('w8_fix75_320', data=big, model=cap, train=base),
        variant('w8_fix75_352', data={**big, 'input_size': 352, 'resize_size': 384},
                model=cap, train=base),
        variant('w8_fix75_320_ls01', data=big, model=cap,
                train={**base, 'label_smoothing': 0.1}),
        variant('w8_fix75_320_seed17', data=big, model=cap, train={**base, 'seed': 17}),
        variant('w8_fix75_320_seed43', data=big, model=cap, train={**base, 'seed': 43}),
        variant('w8_fix75_320_seed97', data=big, model=cap, train={**base, 'seed': 97}),
    ]


def wave9():
    """第九轮：固定 q75 增强下，检查更长预算与更大容量是否还有收益。"""
    fix75 = {'prob': 1.0, 'quality_min': 75, 'quality_max': 75}
    native = {'mode': 'native', 'hflip': 0.5}
    base = {'weight_decay': 0.05, 'label_smoothing': 0.05, 'ema_decay': 0.999,
            'epochs': 60, 'selection': 'jpeg_auroc'}
    cap = {'prior': {'parts': ['block', 'highfreq']}}
    big = {'aug': {**native, 'jpeg_aug': fix75}, 'input_size': 320, 'resize_size': 352}
    return [
        variant('w9_fix75_120ep', data=big, model=cap, train={**base, 'epochs': 120}),
        variant('w9_fix75_m2', data=big, model={**cap, 'embed_dim': [128, 256, 512],
                                                'drop_path_rate': 0.1}, train=base),
        variant('w9_fix75_90ep_352', data={**big, 'input_size': 352, 'resize_size': 384},
                model=cap, train={**base, 'epochs': 90}),
    ]


WAVES = {1: wave1, 2: wave2, 3: wave3, 4: wave4, 5: wave5, 6: wave6, 7: wave7,
         8: wave8, 9: wave9}




def main():
    parser = argparse.ArgumentParser(description='生成调参配置')
    parser.add_argument('--wave', type=int, default=1)
    args = parser.parse_args()
    configs = WAVES[args.wave]()
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = TUNE_DIR / f'manifest_w{args.wave}.txt'
    lines = []
    for cfg in configs:
        path = CONFIG_DIR / f"{cfg['output']['run_name']}.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
            f.write('\n')
        lines.append(str(path.relative_to(TUNE_DIR.parents[1])))
    with open(manifest, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'写入 {len(configs)} 个配置，清单 {manifest}')
    for line in lines:
        print(' ', line)


if __name__ == '__main__':
    main()
