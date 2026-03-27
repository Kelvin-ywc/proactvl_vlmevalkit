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

python run.py --data Video-MME --model proact_vl --verbose
```