import Augmentor
import os
from datetime import datetime

def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def log_error(message, log_file="error_log.txt"):
    '''
    에러 발생 시 터미널 출력 및 로그 파일에 기록
    '''
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {message}"
    print(formatted_msg)
    with open(log_file, "a") as f:
        f.write(formatted_msg + "\n")

# datasets_root_dir = '/data/wangjiaqi/datasets/CUB_200_2011/CUB_200_2011/new_datasets/cub200_cropped/'
datasets_root_dir = '/nas/datasets/CUB_200_2011/cub200_cropped/'
dir = datasets_root_dir + 'train_cropped/'
target_dir = datasets_root_dir + 'train_cropped_augmented/'

makedir(target_dir)
folders = [os.path.join(dir, folder) for folder in next(os.walk(dir))[1]]
target_folders = [os.path.join(target_dir, folder) for folder in next(os.walk(dir))[1]]

total_folders = len(folders)
AUGMENT_FACTOR = 10

for i in range(len(folders)):
    fd = folders[i]
    tfd = target_folders[i]

    folder_name = os.path.basename(fd)

    try:
        # 진행 상황 표시
        print(f"[{i+1}/{total_folders}] Processing: {folder_name}")

        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.ppm')
        files = [f for f in os.listdir(fd) if f.lower().endswith(valid_extensions)]
        num_files = len(files)
        if num_files == 0:
            print(f"  -> [Skip] No images found in {folder_name}")
            continue
        num_samples = num_files * AUGMENT_FACTOR

        # rotation
        p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
        p.rotate(probability=1, max_left_rotation=15, max_right_rotation=15)
        p.flip_left_right(probability=0.5)
        p.sample(num_samples)

        # (2) Skew
        p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
        p.skew(probability=1, magnitude=0.2) 
        p.flip_left_right(probability=0.5)
        p.sample(num_samples)
        
        # (3) Shear
        p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
        p.shear(probability=1, max_shear_left=10, max_shear_right=10)
        p.flip_left_right(probability=0.5)
        p.sample(num_samples)
        
        # (4) Random Distortion
        p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
        p.random_distortion(probability=1.0, grid_width=10, grid_height=10, magnitude=5)
        p.flip_left_right(probability=0.5)
        p.sample(num_samples)

    except Exception as e:
        # 그 외 Augmentor 내부 에러나 알 수 없는 에러
        log_error(f"Unexpected Error in '{folder_name}': {e}")
        # 디버깅을 위해 상세 에러 내용 출력 (필요 시 주석 해제)
        # traceback.print_exc()