python train.py --train_data_dir data/train --valid_data_dir data/valid --epochs 100 --batch_size 2
python predict.py --model_path checkpoints/best_model.pth --input_dir test --output_dir predictions