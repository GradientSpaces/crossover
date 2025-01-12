import argparse
import os
import os.path as osp

import sys
# params
parser = argparse.ArgumentParser()
parser.add_argument('--scannet_path', required=True, help='path to scannet data')
parser.add_argument('--output_path', required=True, help='where to output 2d data')

opt = parser.parse_args()    
print(opt)

def main():
    if not osp.exists(opt.output_path):
        os.makedirs(opt.output_path)
    
    opt.scannet_path = osp.join(opt.scannet_path, 'scans')
    scenes = sorted([d for d in os.listdir(opt.scannet_path) if os.path.isdir(os.path.join(opt.scannet_path, d))])
    print('Found %d scenes' % len(scenes))
    
    for i in range(0,len(scenes)):
        scene_folder = osp.join(opt.scannet_path, scenes[i])
        
        zip_file_instance_seg = osp.join(scene_folder, scenes[i] + '_2d-instance-filt.zip')
        assert osp.exists(zip_file_instance_seg), 'Instance segmentation file does not exist!'
        output_path = os.path.join(opt.output_path, scenes[i], 'data')
        
        os.system('rm -rf {}'.format(osp.join(output_path, 'instance-filt')))
        sys.stdout.write('\r[ %d | %d ] %s\texporting...' % ((i + 1), len(scenes), scenes[i]))
        sys.stdout.flush()
        os.system('unzip -q {} -d {}'.format(zip_file_instance_seg, output_path))
        

if __name__ == '__main__':
    main()