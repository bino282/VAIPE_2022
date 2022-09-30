export NVCC=/usr/bin/nvcc
mkdir /app/output_results/beit
mkdir /app/tmp
python inference_new.py \
    --model beit_large_patch16_224 \
    --data_set image_folder --nb_classes 107 \
    --input_size 224 \
    --device cuda \
    --resume ../datasets/models/beit_107.pth \
    --batch_size 32