CUDA_VISIBLE_DEVICES=0 python train_fasterrcnn_clip.py --config-file configs/diverse_weather_fasterrcnn_clip.yaml \
OUTPUT_DIR "all_outs/train/log" SOLVER.MAX_ITER 240000 SOLVER.STEPS [160000,] \
DATASETS.TEST "('daytime_foggy_train','night_sunny_train','night_rainy_train','dusk_rainy_train',)"