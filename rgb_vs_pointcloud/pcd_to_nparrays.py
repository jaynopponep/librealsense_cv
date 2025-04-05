import numpy as np
import os
import argparse
import struct

def read_pcd_header(file_path):
    header = {}
    with open(file_path, 'rb') as f:
        line = f.readline().strip().decode('utf-8')
        while line and not line.startswith('DATA'):
            key, value = line.split(' ', 1)
            header[key] = value
            line = f.readline().strip().decode('utf-8')
        if line:
            key, value = line.split(' ', 1)
            header[key] = value
    return header

def pcd_to_numpy(pcd_file, output_file=None):
    print(f"Loading PCD file: {pcd_file}")
    
    header = read_pcd_header(pcd_file)
    
    width = int(header.get('WIDTH', 0))
    height = int(header.get('HEIGHT', 1))
    n_points = width * height
    
    data_format = header.get('DATA', 'ascii')
    
    points = np.zeros((n_points, 3), dtype=np.float32)
    
    if data_format.lower() == 'ascii':
        with open(pcd_file, 'r') as f:
            line = f.readline()
            while line and not line.startswith('DATA'):
                line = f.readline()
            
            line = f.readline()
            
            point_idx = 0
            while line and point_idx < n_points:
                values = line.strip().split()
                if len(values) >= 3:  
                    points[point_idx, 0] = float(values[0])
                    points[point_idx, 1] = float(values[1])
                    points[point_idx, 2] = float(values[2])
                point_idx += 1
                line = f.readline()
                
    elif data_format.lower() == 'binary':
        with open(pcd_file, 'rb') as f:
            line = f.readline()
            while line and not b'DATA' in line:
                line = f.readline()
            
            if line:
                line = f.readline()
            
            for i in range(n_points):
                try:
                    x = struct.unpack('f', f.read(4))[0]
                    y = struct.unpack('f', f.read(4))[0]
                    z = struct.unpack('f', f.read(4))[0]
                    points[i] = [x, y, z]
                except:
                    break
                    
    print(f"Converted to numpy array with shape: {points.shape}")
    
    valid_mask = np.logical_and.reduce([
        ~np.isnan(points[:, 0]),
        ~np.isnan(points[:, 1]),
        ~np.isnan(points[:, 2]),
        np.abs(points[:, 0]) + np.abs(points[:, 1]) + np.abs(points[:, 2]) > 0
    ])
    points = points[valid_mask]
    
    print(f"After filtering invalid points: {points.shape}")
    
    if output_file:
        np.save(output_file, points)
        print(f"Saved numpy array to: {output_file}")
    
    return points

def process_directory(input_dir, output_dir=None, recursive=False):
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.pcd'):
                pcd_path = os.path.join(root, file)
                
                if output_dir:
                    rel_path = os.path.relpath(root, input_dir)
                    if rel_path == '.':
                        out_dir = output_dir
                    else:
                        out_dir = os.path.join(output_dir, rel_path)
                        if not os.path.exists(out_dir):
                            os.makedirs(out_dir)
                    
                    output_path = os.path.join(out_dir, file.replace('.pcd', '.npy'))
                else:
                    output_path = None
                
                try:
                    pcd_to_numpy(pcd_path, output_path)
                except Exception as e:
                    print(f"Error processing {pcd_path}: {e}")
        
        if not recursive:
            break

def sample_points(points, n_points):
    if len(points) >= n_points:
        indices = np.random.choice(len(points), n_points, replace=False)
    else:
        indices = np.random.choice(len(points), n_points, replace=True)
    
    return points[indices]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PCD files to NumPy arrays")
    parser.add_argument("input", help="Input PCD file or directory")
    parser.add_argument("-o", "--output", help="Output NPY file or directory (if not specified, arrays won't be saved)")
    parser.add_argument("-r", "--recursive", action="store_true", help="Process directories recursively")
    parser.add_argument("-n", "--num_points", type=int, help="Sample to a fixed number of points")
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        process_directory(args.input, args.output, args.recursive)
    else:
        points = pcd_to_numpy(args.input, args.output)
        
        if args.num_points and points is not None:
            sampled_points = sample_points(points, args.num_points)
            print(f"Sampled to {args.num_points} points")
            
            if args.output:
                output_sampled = args.output.replace('.npy', '_sampled.npy')
                np.save(output_sampled, sampled_points)
                print(f"Saved sampled points to: {output_sampled}")
