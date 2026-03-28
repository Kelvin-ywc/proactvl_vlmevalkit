# env setup
Create conda environment in Proact-VL project.
Then run:
```
pip install -e .
pip install vllm==v0.10.2 --extra-index-url https://download.pytorch.org/whl/cu128
```

Create a symbolic link to run.py from the project in the Proact-VL directory.
```
cd <Proact-VL>
ln -s <PATH to run.py> ./run.py
```
# eval
```
# Video-MME

export CUDA_VISIBLE_DEVICES=2,3,4,5 
torchrun --nproc-per-node=4 run.py --data Video-MME_8frame --model Proact-VL --verbose
torchrun --nproc-per-node=4 run.py run.py --data Video-MME_64frame --model Proact-VL --verbose
torchrun --nproc-per-node=4 run.py run.py --data Video-MME_8frame_subs --model Proact-VL --verbose

python run.py --data Video-MME_8frame --model Qwen3-VL-8B-Instruct --verbose
python run.py --data Video-MME_64frame --model Qwen3-VL-8B-Instruct --verbose
python run.py --data Video-MME_8frame_subs --model Qwen3-VL-8B-Instruct --verbose

# LongVideoBench
torchrun --nproc-per-node=8 run.py --data LongVideoBench_64frame --model Proact-VL --verbose
torchrun --nproc-per-node=8 run.py --data LongVideoBench_64frame --model Qwen3-VL-8B-Instruct --verbose

<!-- torchrun --nproc-per-node=8 run.py --data LongVideoBench_8frame --model Proact-VL --verbose
torchrun --nproc-per-node=8 run.py --data LongVideoBench_8frame_subs --model Proact-VL --verbose

python run.py --data LongVideoBench_8frame --model Qwen3-VL-8B-Instruct --verbose
python run.py --data LongVideoBench_8frame_subs --model Qwen3-VL-8B-Instruct --verbose -->

```