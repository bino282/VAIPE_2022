export NVCC=/usr/bin/nvcc
CUDA_VISIBLE_DEVICES=0 python run_class_finetuning.py \
    --model beit_large_patch16_224 --data_path ../datasets/VAIPE_PILL_CLS/train \
    --eval_data_path ../datasets/VAIPE_PILL_CLS/val \
    --data_set image_folder --nb_classes 108 \
    --input_size 224 \
    --finetune ../datasets/models/beit_large_patch16_224_pt22k_ft22k.pth \
    --output_dir outputs/VAIPE_PILL_108_recall_loss/beit_large_patch16_224_pt22k_ft22k --batch_size 8 --lr 2e-5 --update_freq 8 \
    --warmup_epochs 5 --epochs 100 --layer_decay 0.9 --drop_path 0.4 \
    --weight_decay 1e-8