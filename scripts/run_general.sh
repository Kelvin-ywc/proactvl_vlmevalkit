# videomme
# torchrun --nproc-per-node=2 run.py --data Video-MME_8frame --model Qwen3-VL-8B-Instruct --verbose
torchrun --nproc-per-node=2 run.py --data Video-MME_8frame --model Proact-VL --verbose

# torchrun --nproc-per-node=2 run.py --data Video-MME_8frame_subs --model Qwen3-VL-8B-Instruct --verbose
torchrun --nproc-per-node=2 run.py --data Video-MME_8frame_subs --model Proact-VL --verbose

torchrun --nproc-per-node=2 run.py --data Video-MME_64frame --model Qwen3-VL-8B-Instruct --verbose
torchrun --nproc-per-node=2 run.py --data Video-MME_64frame --model Proact-VL --verbose
# python run.py --data Video-MME_8frame --model Qwen2.5-VL-7B-Instruct --verbose

# torchrun --nproc-per-node=2 run.py run.py --data Video-MME_8frame_subs --model Proact-VL --verbose
# torchrun --nproc-per-node=1 run.py --data LongVideoBench_8frame --model Proact-VL --verbose

# # Qwen2.5-VL-7B-Instruct
# torchrun --nproc-per-node=2 run.py --data Video-MME_8frame --model Qwen2.5-VL-7B-Instruct --verbose