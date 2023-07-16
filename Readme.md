# Self-Supervised Indoor 360-Degree Depth Estimation via Structural Regularization

# Train Command
python train_ud_pair.py --gpu 0 --batch_size 16 --epochs 30 --configuration ud --model resnet_coord --train_path YOUR_TRAIN_TXT_PATH --test_path YOUR_VAL_TXT_PATH --save_path YOUR_MODEL_SAVE_PATH --using_normloss --using_disp2seg --planar_thresh 200  --vps_path YOUR_VPS_PATH

# Test Command
python test_old.py -b 16 --test_path YOUR_VAL_TEST_PATH  --configuration ud --weights YOUR_MODEL_WEIGHT_PATH --model initial -g YOUR_GPU_ID --save_samples --save_path YOUR_IMG_RESULT_PATH

