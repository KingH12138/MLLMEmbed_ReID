# 20250522：注意目前reid_augment需要手动在load_processor那儿设置一下，指定为True/False
# 注意每次换行符“\”后面不能有任何字符，不然会参数解析错误

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=22472 --max_restarts=0 train_ReIDMLLM.py \
  --model_name Qwen/Qwen2-VL-2B-Instruct \
  --output_dir /hongbojiang/workdirs/mllmreid/aio_lorasft_lora3\(lr_adjust\) \
  --bf16 --pooling last \
  --lora True \
  --lora_r 16 \
  --lora_alpha 64 \
  --lora_target_modules qkv,proj,mlp.0,mlp.2,q_proj,k_proj,v_proj,o_proj\
  --data_root /hongbojiang/datasets/aio_reid/aio \
  --dataset_name aio \
  --dataset_meta /hongbojiang/datasets/aio_reid/aio_train.csv \
  --max_len 165\
  --logging_steps 1 \
  --max_steps 50000 \
  --save_strategy epoch\
  --normalize True\
  --per_device_train_batch_size 32 \
  --save_safetensors False --label_smooth False \
  --report_to none \
  --num_classes 17278 \

# distill_svd
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=22477 --max_restarts=0 distill_mllm_v2_svd.py \
  --model_name Qwen/Qwen2-VL-2B-Instruct \
  --student_base_model_name "openai/clip-vit-large-patch14"\
  --output_dir "/hongbojiang/workdirs/mllmreid/distill_svd" \
  --bf16 --pooling last \
  --lora True \
  --dataset_meta /hongbojiang/datasets/aio_reid/aio_train.csv \
  --max_len 165 \
  --logging_steps 1 \
  --learning_rate 1e-5 \
  --per_device_train_batch_size 128 \
  --num_train_epochs 50 \
  --num_classes 17278 \
  --checkpoint_path "/hongbojiang/checkpoints/aio_lorasft_lora8_moreattn_plus_triplet_loss/checkpoint-122976"

# distill-basline
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=22481 --max_restarts=0 distill_mllm_v2.py \
  --model_name Qwen/Qwen2-VL-2B-Instruct \
  --student_base_model_name "openai/clip-vit-large-patch14"\
  --output_dir "/hongbojiang/workdirs/mllmreid/distill_baseline" \
  --bf16 --pooling last \
  --lora True \
  --dataset_meta /hongbojiang/datasets/aio_reid/aio_train.csv \
  --max_len 165 \
  --logging_steps 1 \
  --learning_rate 1e-5 \
  --per_device_train_batch_size 128 \
  --num_train_epochs 50 \
  --num_classes 17278 \
  --checkpoint_path "/hongbojiang/checkpoints/aio_lorasft_lora8_moreattn_plus_triplet_loss/checkpoint-122976"

# student training only
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=22481 --max_restarts=0 train_stu.py \
  --model_name Qwen/Qwen2-VL-2B-Instruct \
  --student_base_model_name "openai/clip-vit-large-patch14"\
  --output_dir "/hongbojiang/workdirs/mllmreid/stu_train_nopretrain" \
  --bf16 --pooling last \
  --lora True \
  --dataset_meta /hongbojiang/datasets/aio_reid/aio_train.csv \
  --max_len 165 \
  --logging_steps 1 \
  --learning_rate 5e-5 \
  --per_device_train_batch_size 128 \
  --num_train_epochs 50 \
  --num_classes 17278 \
  --stu_resume /hongbojiang/workdirs/mllmreid/stu_train_nopretrain/student_model_epoch8.pth

# distill-basline student finetune by teacher model
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=22483 --max_restarts=0 distill_stu_finetune.py \
  --model_name Qwen/Qwen2-VL-2B-Instruct \
  --student_base_model_name "openai/clip-vit-large-patch14"\
  --output_dir "/hongbojiang/workdirs/mllmreid/distill_stu_finetune_warmbackbone" \
  --bf16 --pooling last \
  --lora True \
  --dataset_meta /hongbojiang/datasets/aio_reid/aio_train.csv \
  --max_len 165 \
  --logging_steps 1 \
  --learning_rate 1e-5 \
  --per_device_train_batch_size 128 \
  --num_train_epochs 50 \
  --num_classes 17278 \
  --stu_resume "/hongbojiang/workdirs/mllmreid/distill_stu_finetune_warmbackbone/student_model_epoch9.pth" \
  --student_test_ckpt_path "/hongbojiang/workdirs/mllmreid/stu_train/student_model_epoch15.pth"

# distill-basline student finetune by teacher model with svd motivation
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=22480 --max_restarts=0 distill_stu_finetune_svd.py \
  --model_name Qwen/Qwen2-VL-2B-Instruct \
  --student_base_model_name "openai/clip-vit-large-patch14"\
  --output_dir "/hongbojiang/workdirs/mllmreid/distill_stu_finetune_svd_warmbackbone(nocosine)" \
  --bf16 --pooling last \
  --lora True \
  --max_len 165 \
  --logging_steps 1 \
  --learning_rate 1e-5 \
  --per_device_train_batch_size 128 \
  --num_train_epochs 50 \
  --num_classes 17278 \
  --student_test_ckpt_path "/hongbojiang/workdirs/mllmreid/stu_train/student_model_epoch15.pth"

########################################################################################################################
# test command

ckpt_id="122976"
dataname=icfgpedes
task_name="aio_lorasft_lora8_moreattn_plus_triplet_loss"
CUDA_VISIBLE_DEVICES=4 python test_ReIDMLLM_AIO.py \
 --model_name Qwen/Qwen2-VL-2B-Instruct \
 --output_dir /data/jhb_data/workdirs/mllmreid/${task_name}/checkpoint-${ckpt_id}\
 --bf16 --pooling last \
 --lora True \
 --data_root /data/jhb_data/datasets/aio_reid \
 --dataset_name aio \
 --dataset_meta /data/jhb_data/datasets/aio_reid/${dataname}_aio.csv \
 --max_len 165 \
 --logging_steps 1 \
 --num_train_epochs 50 \
 --warmup_ratio 0.3 \
 --save_steps 5000 --normalize False\
 --per_device_train_batch_size 32 \
 --save_safetensors False --label_smooth False \
 --report_to none \
 --num_classes 17278 \
 --test_batchsize 256 \
 --checkpoint_path  /data/jhb_data/workdirs/mllmreid/${task_name}/checkpoint-${ckpt_id}\
 --test_output_dir /data/jhb_data/workdirs/mllmreid/${task_name}/checkpoint-${ckpt_id}/${dataname}_testset

ckpt_id="122976"
dataname=rstpreid
task_name="aio_lorasft_lora8_moreattn_plus_triplet_loss"
CUDA_VISIBLE_DEVICES=6 python notes/20251009.py \
 --model_name Qwen/Qwen2-VL-2B-Instruct \
 --output_dir /hongbojiang/workdirs/mllmreid/${task_name}/checkpoint-${ckpt_id}\
 --bf16 --pooling last \
 --lora True \
 --data_root /hongbojiang/datasets/aio_reid \
 --dataset_name aio \
 --dataset_meta /hongbojiang/datasets/aio_reid/${dataname}_aio_train.csv \
 --max_len 165 \
 --logging_steps 1 \
 --num_train_epochs 50 \
 --warmup_ratio 0.3 \
 --save_steps 5000 --normalize False\
 --per_device_train_batch_size 32 \
 --save_safetensors False --label_smooth False \
 --report_to none \
 --num_classes 17278 \
 --test_batchsize 128 \
 --checkpoint_path  /hongbojiang/workdirs/mllmreid/${task_name}/checkpoint-${ckpt_id}\
 --test_output_dir /hongbojiang/workdirs/mllmreid/${task_name}/checkpoint-${ckpt_id}/${dataname}_testset
# nohup bash nohup.sh > output_aio_test.log 2>&1 &

###################################################################################################################################################
# distll_test

ckpt_id=48
student_task_name="distill_stu_finetune_svd_warmbackbone"
student_ckpt_path="/hongbojiang/workdirs/mllmreid/${student_task_name}/student_model_epoch${ckpt_id}.pth"
dataname=rstpreid
CUDA_VISIBLE_DEVICES=4 python distill_mllm_test_v2.py \
  --model_name Qwen/Qwen2-VL-2B-Instruct \
  --student_base_model_name ""openai/clip-vit-large-patch14"" \
  --output_dir "/hongbojiang/workdirs/mllmreid/${student_task_name}" \
  --pooling last \
  --lora True \
  --dataset_meta /hongbojiang/datasets/aio_reid/${dataname}_aio.csv \
  --max_len 165 \
  --logging_steps 1 \
  --per_device_train_batch_size 256 \
  --num_train_epochs 50 \
  --num_classes 17278 \
  --test_batchsize 128 \
  --checkpoint_path "/hongbojiang/checkpoints/aio_lorasft_lora8_moreattn_plus_triplet_loss/checkpoint-122976" \
  --student_test_ckpt_path ${student_ckpt_path} \
  --test_output_dir "/hongbojiang/workdirs/mllmreid/${student_task_name}/test_output/${dataname}_${ckpt_id}"

# cross cloud-edge test
ckpt_id=13
student_task_name="distill_stu_finetune_svd_warmbackbone"
student_ckpt_path="/hongbojiang/workdirs/mllmreid/${student_task_name}/student_model_epoch${ckpt_id}.pth"
dataname=icfgpedes
CUDA_VISIBLE_DEVICES=5 python distill_mllm_test_v2_stu2teacher.py \
  --model_name Qwen/Qwen2-VL-2B-Instruct \
  --student_base_model_name ""openai/clip-vit-large-patch14"" \
  --output_dir "/hongbojiang/workdirs/mllmreid/${student_task_name}" \
  --pooling last \
  --lora True \
  --dataset_meta /hongbojiang/datasets/aio_reid/${dataname}_aio.csv \
  --max_len 165 \
  --logging_steps 1 \
  --per_device_train_batch_size 256 \
  --num_train_epochs 50 \
  --num_classes 17278 \
  --test_batchsize 256 \
  --checkpoint_path "/hongbojiang/checkpoints/aio_lorasft_lora8_moreattn_plus_triplet_loss/checkpoint-122976" \
  --student_test_ckpt_path ${student_ckpt_path} \
  --test_output_dir "/hongbojiang/workdirs/mllmreid/${student_task_name}/test_output/${dataname}_${ckpt_id}"

# single student test
ckpt_id=48
student_task_name="distill_stu_finetune_svd_warmbackbone(nocosine)"
student_ckpt_path="/hongbojiang/workdirs/mllmreid/${student_task_name}/student_model_epoch${ckpt_id}.pth"
dataname=cuhkpedes
CUDA_VISIBLE_DEVICES=4 python test_stu.py \
  --model_name Qwen/Qwen2-VL-2B-Instruct \
  --student_base_model_name ""openai/clip-vit-large-patch14"" \
  --output_dir "/data/jhb_data/workdirs/mllmreid/${student_task_name}" \
  --pooling last \
  --lora True \
  --dataset_meta /hongbojiang/datasets/aio_reid/${dataname}_aio.csv \
  --max_len 165 \
  --logging_steps 1 \
  --per_device_train_batch_size 256 \
  --num_train_epochs 50 \
  --num_classes 17278 \
  --test_batchsize 256 \
  --student_test_ckpt_path ${student_ckpt_path} \
  --test_output_dir "/hongbojiang/workdirs/mllmreid/${student_task_name}/test_output/${dataname}_${ckpt_id}"

# single student test
ckpt_id=13
student_task_name="distill_stu_finetune_svd_warmbackbone"
student_ckpt_path="/hongbojiang/workdirs/mllmreid/${student_task_name}/student_model_epoch${ckpt_id}.pth"
dataname=icfgpedes
CUDA_VISIBLE_DEVICES=3 python test_stu.py \
  --model_name Qwen/Qwen2-VL-2B-Instruct \
  --student_base_model_name ""openai/clip-vit-large-patch14"" \
  --output_dir "/data/jhb_data/workdirs/mllmreid/${student_task_name}" \
  --pooling last \
  --lora True \
  --dataset_meta /hongbojiang/datasets/aio_reid/${dataname}_aio.csv \
  --max_len 165 \
  --logging_steps 1 \
  --per_device_train_batch_size 256 \
  --num_train_epochs 50 \
  --num_classes 17278 \
  --test_batchsize 256 \
  --student_test_ckpt_path ${student_ckpt_path} \
  --test_output_dir "/hongbojiang/workdirs/mllmreid/${student_task_name}/test_output/${dataname}_${ckpt_id}"


# batch student test(参数需要去脚本里面指定,这里修改测试集)
dataname=rstpreid
CUDA_VISIBLE_DEVICES=0 python test_stu_batch.py \
  --model_name Qwen/Qwen2-VL-2B-Instruct \
  --student_base_model_name ""openai/clip-vit-large-patch14"" \
  --pooling last \
  --lora True \
  --dataset_meta /hongbojiang/datasets/aio_reid/${dataname}_aio.csv \
  --max_len 165 \
  --logging_steps 1 \
  --per_device_train_batch_size 256 \
  --num_train_epochs 50 \
  --num_classes 17278 \
  --test_batchsize 128

# student svd anaylsis
ckpt_id=15
student_task_name="stu_train"
student_ckpt_path="/hongbojiang/workdirs/mllmreid/${student_task_name}/student_model_epoch${ckpt_id}.pth"
dataname=aio_test
CUDA_VISIBLE_DEVICES=0 python notes/20251029.py \
  --model_name Qwen/Qwen2-VL-2B-Instruct \
  --student_base_model_name "openai/clip-vit-large-patch14" \
  --output_dir "/data/jhb_data/workdirs/mllmreid/${student_task_name}" \
  --pooling last \
  --lora True \
  --dataset_meta /hongbojiang/datasets/aio_reid/${dataname}_aio.csv \
  --max_len 165 \
  --logging_steps 1 \
  --per_device_train_batch_size 256 \
  --num_train_epochs 50 \
  --num_classes 17278 \
  --test_batchsize 128 \
  --student_test_ckpt_path ${student_ckpt_path} \
  --test_output_dir "/hongbojiang/workdirs/mllmreid/${student_task_name}/test_output/${dataname}_${ckpt_id}"