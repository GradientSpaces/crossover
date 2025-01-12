import argparse
import os
import sys
import math
import numpy as np

from scannet_sensordata import SensorData

def make_intrinsic(fx, fy, mx, my):
    '''Create camera intrinsics.'''

    intrinsic = np.eye(4)
    intrinsic[0][0] = fx
    intrinsic[1][1] = fy
    intrinsic[0][2] = mx
    intrinsic[1][2] = my
    return intrinsic

def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    '''Adjust camera intrinsics.'''

    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(image_dim[1] * float(
                    intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
    intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
    intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)
    return intrinsic

# params
parser = argparse.ArgumentParser()
parser.add_argument('--scannet_path', required=True, help='path to scannet data')
parser.add_argument('--output_path', required=True, help='where to output 2d data')
parser.add_argument('--export_label_images', dest='export_label_images', action='store_true')
parser.add_argument('--label_type', default='label-filt', help='which labels (label or label-filt)')
parser.add_argument('--frame_skip', type=int, default=1, help='export every nth frame')
parser.add_argument('--label_map_file', default='',
                    help='path to scannetv2-labels.combined.tsv (required for label export only)')
parser.add_argument('--output_image_width', type=int, default=640, help='export image width')
parser.add_argument('--output_image_height', type=int, default=480, help='export image height')

# parser.set_defaults(export_label_images=False)
opt = parser.parse_args()
# if opt.export_label_images:
#     assert opt.label_map_file != ''
    
print(opt)

def main():
    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)
    
    scenes = sorted([d for d in os.listdir(opt.scannet_path) if os.path.isdir(os.path.join(opt.scannet_path, d))])
    print('Found %d scenes' % len(scenes))
    
    img_dim = (640, 480)
    intrinsics = make_intrinsic(fx=577.870605, fy=577.870605, mx=319.5, my=239.5)
    intrinsics = adjust_intrinsic(intrinsics, img_dim, (opt.output_image_width, opt.output_image_height))
    
    for i in range(0,len(scenes)):
        sens_file = os.path.join(opt.scannet_path, scenes[i], scenes[i] + '.sens')
        
        # Save intrinsics
        np.savetxt(os.path.join(opt.output_path, scenes[i], 'intrinsics.txt'), intrinsics)
        
        output_path = os.path.join(opt.output_path, scenes[i], 'data')
        output_color_path = os.path.join(output_path, 'color')
        
        # if os.path.exists(os.path.join(opt.output_path, scenes[i])) and os.path.exists(output_color_path + '/0.jpg'):
        #     print(scenes[i] + ' already extracted!')
        #     continue
        
        if not os.path.isdir(output_color_path):
            os.makedirs(output_color_path)
        output_depth_path = os.path.join(output_path, 'depth')
        if not os.path.isdir(output_depth_path):
            os.makedirs(output_depth_path)
        output_pose_path = os.path.join(output_path, 'pose')
        if not os.path.isdir(output_pose_path):
            os.makedirs(output_pose_path)
        
        
        # read and export
        sys.stdout.write('\r[ %d | %d ] %s\tloading...' % ((i + 1), len(scenes), scenes[i]))
        sys.stdout.flush()
        if os.path.exists(sens_file):
            sd = SensorData(sens_file)    
        else:
            print(scenes[i] + " does not exist!")

        sys.stdout.write('\r[ %d | %d ] %s\texporting...' % ((i + 1), len(scenes), scenes[i]))
        sys.stdout.flush()
        
        sd.export_color_images(output_color_path, image_size=[opt.output_image_height, opt.output_image_width],
                               frame_skip=opt.frame_skip)
        sd.export_depth_images(output_depth_path, image_size=[opt.output_image_height, opt.output_image_width],
                               frame_skip=opt.frame_skip)
        sd.export_poses(output_pose_path, frame_skip=opt.frame_skip)
        

if __name__ == '__main__':
    main()