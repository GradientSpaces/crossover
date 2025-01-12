import argparse
import os
import os.path as osp

import sys
# params
parser = argparse.ArgumentParser()
parser.add_argument('--scan3r_path', required=True, help='path to scan3r data')
parser.add_argument('--output_path', required=True, help='where to output scan3r data')

opt = parser.parse_args()    
print(opt)

def main():
    if not osp.exists(opt.output_path):
        os.makedirs(opt.output_path)
    
    opt.scan3r_path = osp.join(opt.scan3r_path, 'scans')
    scenes = sorted([d for d in os.listdir(opt.scan3r_path) if os.path.isdir(os.path.join(opt.scan3r_path, d))])
    print('Found %d scenes' % len(scenes))
    
    for i in range(0,len(scenes)):
        scene_folder = osp.join(opt.scan3r_path, scenes[i])
        
        zip_file_sequence = osp.join(scene_folder, scenes[i] + 'sequence.zip')
        if not osp.exists(zip_file_sequence): 
            print('No sequence.zip file found for scene {}!!'.format(scenes[i]))
        
        output_path = os.path.join(opt.output_path, scenes[i], 'sequence')
        
        os.system('rm -rf {}'.format(osp.join(output_path, 'sequence')))
        sys.stdout.write('\r[ %d | %d ] %s\texporting...' % ((i + 1), len(scenes), scenes[i]))
        sys.stdout.flush()
        os.system('unzip -q {} -d {}'.format(zip_file_sequence, output_path))
        

if __name__ == '__main__':
    main()