# env setup
Create conda environment in (Proact-VL project)[https://github.com/microsoft/AnthropomorphicIntelligence/tree/main/Proact-VL/proactvl].
Then run:
```
pip install -e .
pip install pysubs2
pip install vllm==v0.10.2 --extra-index-url https://download.pytorch.org/whl/cu128
```

Create a symbolic link from the (proact-vl project)[https://github.com/microsoft/AnthropomorphicIntelligence/tree/main/Proact-VL/proactvl] in the current directory.
```
cd <Proact-VL>
ln -s Proact-VL/proactvl ./
```
# eval
```
# Video-MME
torchrun --nproc-per-node=8 run.py --data Video-MME_8frame --model Proact-VL --verbose
torchrun --nproc-per-node=8 run.py run.py --data Video-MME_8frame_subs --model Proact-VL --verbose
torchrun --nproc-per-node=8 run.py --data Video-MME_8frame --model Qwen3-VL-8B-Instruct --verbose
torchrun --nproc-per-node=8 run.py --data Video-MME_8frame_subs --model Qwen3-VL-8B-Instruct --verbose

# LongVideoBench
torchrun --nproc-per-node=8 run.py --data LongVideoBench_64frame --model Proact-VL --verbose
torchrun --nproc-per-node=8 run.py --data LongVideoBench_8frame_subs --model Proact-VL --verbose
torchrun --nproc-per-node=8 run.py --data LongVideoBench_8frame --model Qwen3-VL-8B-Instruct --verbose
torchrun --nproc-per-node=8 run.py --data LongVideoBench_8frame_subs --model Qwen3-VL-8B-Instruct --verbose

```
