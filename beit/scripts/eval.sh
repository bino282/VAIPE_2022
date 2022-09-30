export NVCC=/usr/bin/nvcc
python run_class_finetuning.py \
    --model beit_large_patch16_224 --data_path /datahdd/nhanv/git/Swin-Transformer/imageVAIPE_107/train \
    --eval_data_path /datahdd/nhanv/git/Swin-Transformer/imageVAIPE_107/val \
    --data_set image_folder  --nb_classes 100 \
    --eval \
    --auto_resume \
    --output_dir save_result --batch_size 32 --lr 2e-5 --update_freq 2 \
    --warmup_epochs 5 --epochs 30 --layer_decay 0.9 --drop_path 0.4 \
    --weight_decay 1e-8