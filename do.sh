#/bin/bash
#python main.py ucf101 Flow train_list.txt test_list.txt --num_segments 3 --gd 20 --lr 0.001 --lr_steps 30 60 --epochs 80 -b 12 -j 4 --dropout 0.8
#python main.py ucf101 Flow train_list.txt test_list.txt --num_segments 3 --gd 20 --lr 0.001 --lr_steps 70 100 --epochs 120 -b 12 -j 4 --dropout 0.7 --eval-freq 5
#python main.py ucf101 RGB train_list.txt test_list.txt --num_segments 3 --gd 20 --lr 0.001 --lr_steps 30 60 --epochs 80 -b 12 -j 4 --dropout 0.8 --eval-freq 5
#python test_models.py ucf101 Flow test_list.txt flowmodel_best.pth  --save_scores result --test_segments 3
#python test_models.py ucf101 RGBDiff test_list.txt rgbdiffcheckpoint.pth  --save_scores result --test_segments 25
python main.py ucf101 Flow train_list.txt test_list.txt --num_segments 3  --gd 20 --lr 0.001 --lr_steps 190 300 --epochs 340 -b 12 -j 4 --dropout 0.6 --eval-freq 5