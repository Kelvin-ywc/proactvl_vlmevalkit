# env setup
Create conda environment in Proact-VL project.
Then run:
```
pip install -e .
```

Create a symbolic link to run.py from the project in the Proact-VL directory.
```
cd <Proact-VL>
ln -s <PATH to run.py> ./run.py
```
# eval
```
# Video-MME
python run.py --data Video-MME_8frame --model Proact-VL --verbose
export CUDA_VISIBLE_DEVICES=2,3,4,5 
torchrun --nproc-per-node=4 run.py --data Video-MME_8frame --model Proact-VL --verbose

python run.py --data Video-MME_64frame --model Proact-VL --verbose
python run.py --data Video-MME_8frame_subs --model Proact-VL --verbose

python run.py --data Video-MME_8frame --model Qwen3-VL-8B-Instruct --verbose
python run.py --data Video-MME_64frame --model Qwen3-VL-8B-Instruct --verbose
python run.py --data Video-MME_8frame_subs --model Qwen3-VL-8B-Instruct --verbose

# LongVideoBench
python run.py --data LongVideoBench_8frame --model Proact-VL --verbose
python run.py --data LongVideoBench_8frame_subs --model Proact-VL --verbose
python run.py --data LongVideoBench_64frame --model Proact-VL --verbose

python run.py --data LongVideoBench_8frame --model Qwen3-VL-8B-Instruct --verbose
python run.py --data LongVideoBench_8frame_subs --model Qwen3-VL-8B-Instruct --verbose
python run.py --data LongVideoBench_64frame --model Qwen3-VL-8B-Instruct --verbose
```