# klab_rotation

1. Classifier directly on pixels (SVM)
2. Copy architecture from Duncan (https://klab.tch.harvard.edu/publications/PDFs/gk7817.pdf) using backprop
3. Implement growth
4. Implement pruning

# Notes

**2024-10-15:**
- Set up conda environment and git repo.
- Implemented simple SVM for MNIST classification (01_Basic-SVM-MNIST-Classification.ipynb)
    - Took a while to run grid search locally. Reduced the training set to only 1k data points.
- Implemented simple MLP using keras and parameters mentioned in the thesis document. I was unsure if he used ReLU for the hidden layer and softmax for the output. 

**2024-12-6: O2 setup**

```
local> ssh username@o2.hms.harvard.edu
login04> tmux new -s launch_gpu # new screen
login04-launch_gpu> module load miniconda3/23.1.0
login04-launch_gpu> srun --pty -p interactive --mem 1G -t 0-06:00 /bin/bash
```
Get compute node name and add to local `~/.ssh/config` HostName under `o2job`.
```
# local ~/.ssh/config
# HMS O2
Host o2jump
  HostName o2.hms.harvard.edu
  User clc926
  ForwardAgent yes
  ForwardX11 yes
  ForwardX11Trusted yes
Host o2job
  HostName compute-g-16-176
  User clc926
  ProxyJump o2jump
  ForwardAgent yes
```
In VScode: `Show and Run Commands` > `Remote-SSH: Connect to Host` > `o2job`. Navigate to github repo.

Build conda environment on O2. In the terminal:
```
interactive-node> module load miniconda3/23.1.0 # not sure if this is necessary
interactive-node> conda env create --name klab_env --file=02_setup/environments.gpu_interactive.yml
```

Open jupyter notebook and select  `klab_env` as the kernel and `klab_env` python path as python interpreter.

**2024-12-11: O2 setup with GPU**

Login
```
local> ssh username@o2.hms.harvard.edu
login04> tmux new -s launch_gpu # new screen
login04-launch_gpu> module load miniconda3/23.1.0 gcc/9.2.0 cuda/12.1
login04-launch_gpu> srun -n 1 --pty -t 1:00:00 -p gpu --gres=gpu:1 --mem 5G --job-name="vscodetunnel" -x compute-g-16-177,compute-g-16-175 bash
```

Get compute node and add to local `/.ssh/config` file (see above). 

Open VScode and connect to `o2job`. Navigate to github repo. Open notebook. Set kernel to `klab_env(3.10.10)`

**2024-12-26 Training SBATCH script notes**

To run scripts in `sbatch_scripts/` directory:

```
$ cd klab_rotation # github repo
$ sbatch sbatch_scripts/<sbatch_script.sh>
Submitted batch job <job_id>
```

Outputs:
- stderr/stdout: `outputs/sbatch/<job_id>`. The `job_id` is produced when running above sbatch command
- Training outputs/model parameters: `outputs/<notebook analysis id>/<desc>/<training parameter summary>`
  - ex. `klab_rotation/outputs/11/no_prune_fast/sbatch-55637925_bs-32_sf-0.05_vr-6_id-0.5_nti-100_ugpp-False_lr-0.001_s-4`
  - `11/no_prune_fast` is specified as the output director in the python script run in the `sbatch_script.sh`
  - See python script's `parameters_abbr` variable to see how arguments are saved in the directory name (i.e. `bs` is batch size)

Pass the whole training output directory to `src.training_results.MLPUnsupervisedTrainingResults(...)` or `src.training_results.CNNUnsupervisedTrainingResults(...)` to automatically parse and reload model.

## Conda environment commands
This was trial and error to get this to work with the GPU. I may have lost a few steps along the way.

```
# check available cuda modules. On O2 the highest I saw was 12.1

gpu-node> module spider cuda

do_ypcall: clnt_call: RPC: Timed out

--------------------------------
  cuda:
--------------------------------
     Versions:
        cuda/8.0
        cuda/9.0
        cuda/10.0
        cuda/10.1
        cuda/10.2
        cuda/11.2
        cuda/11.7
        cuda/12.1

---------------------------------
  For detailed information about a specific "cuda" module (including how to load the modules) use the module's full name.
  For example:

     $ module spider cuda/12.1
--------------------------------

```

```
# check GPU
gpu-node> nvidia-smi

Wed Dec 11 15:47:31 2024       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.14              Driver Version: 550.54.14      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla M40                      On  |   00000000:02:00.0 Off |                    0 |
| N/A   40C    P0             66W /  250W |     104MiB /  11520MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A     24450      C   ...926/.conda/envs/klab_env/bin/python        101MiB |
+-----------------------------------------------------------------------------------------+
```

Note: it says CUDA version is 12.4, but this is the **highest** version this GPU can operate on. See https://pytorch.org/get-started/locally/ for other ways to install torch for different OS and compute packages.

```
login-node> module load miniconda3/23.1.0 gcc/9.2.0 cuda/12.1

# request and connect to gpu node
gpu-node> conda create -n "klab_env" python=3.10
gpu-node> conda activate klab_env
(klab_env) gpu-node> conda install ipykernel pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# extra confirm the cuda version is the same as the one I module loaded
(klab_env) gpu-node> conda install nvidia/label/cuda-12.1.0::cuda

# install jupyter notebook packages
(klab_env) gpu-node> conda install ipykernel jupyter

# confirm python path
(klab_env) gpu-node> which python 
~/.conda/envs/klab_env/bin/python

(klab_env) gpu-node> python -m ipykernel install --user --name=klab_env

# install other packages
conda install matplotlib seaborn numpy
```

Check that it works and then saved conda environment state to `.yml` file:
```
(klab_env) gpu-node> conda env export --name klab_env --file o2_setup/environment.gpu_interactive.yml
```
