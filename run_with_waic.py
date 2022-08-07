import json
import os
import shutil
import subprocess
from pathlib import Path
import argparse
import random
import string


def startup(args):
    # find available working dir
    v = 0
    while True:
        output_dir = os.path.abspath(os.path.join(args.working_dir_base, f'{args.tag}-v{v}'))
        if not os.path.isdir(output_dir):
            break
        v += 1
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(f'-startup- working directory is {output_dir}')

    # copy files
    for filename in os.listdir('.'):
        if filename.endswith('.py'):
            shutil.copy(filename, output_dir)
    return output_dir


def mirror_and_submit():
    parser = argparse.ArgumentParser('WAIC mirror and submit')
    parser.add_argument('--tag', type=str, help='tag for this run - experiment description')
    parser.add_argument('--working_dir_base', default='./experiments', type=str, help='master folder for all experiments')
    parser.add_argument('--cont', default=1, type=int, help='number of jobs in seq array')
    parser.add_argument('--ngpus', default=1, type=int, help='number of GPUs')
    parser.add_argument('--gpu_mem', default=1, type=int, help='minimal GPU memory in GB')
    parser.add_argument('--waic_q', default='waic-medium', type=str, help='LSF queue name')
    parser.add_argument('--bsub_args', default='', type=str, help='additinoal bsub arguments')
    parser.add_argument('--output_dir', default=None, type=str, help='Path to save logs and checkpoints.')

    args, main_script = parser.parse_known_args()

    # create new folder for this run
    if args.output_dir is None:
        output_dir = startup(args)
        job_name = os.path.basename(output_dir)
        if len(job_name) == 0:
            job_name = os.path.basename(output_dir[:-1])
        log = os.path.join(output_dir, 'log')
    else:
        output_dir = args.output_dir
        job_name = 'log_' + ''.join(random.choice(string.digits + string.ascii_letters) for _ in range(10))
        log = os.path.join(output_dir, job_name)
    main_script.extend(('--output_dir', output_dir))
    scipt_cmd = ' '.join(main_script)

    # ---------------------------------------------------------------------
    #
    # Need tpo change here for your modules/environment/conda
    #
    # ---------------------------------------------------------------------
    # build the command to run
    seq_cmd = [f'/home/projects/bagon/shared/seq_arr.sh', '-e', f'{args.cont:d}',
               '-c',
               f'"bsub -H -J \"{job_name}[1-{args.cont}]\" -o {log}.o '
               + f'-e {log}.e -R rusage[mem={48000*args.ngpus}] -R affinity[thread*{10*args.ngpus}] '
               + f'-gpu num={args.ngpus}:j_exclusive=yes:gmem={args.gpu_mem}G:aff=no -q {args.waic_q}'
               + ' ' + args.bsub_args + ' '
               + ' source /etc/profile.d/modules.sh;'
               + ' module unload cuda cudnn gcc;'
               + ' module load anaconda/2020.02/python/3.7 CUDA/11.3.1 gcc/8.3.0;'
               + ' module list;  '
               + f' source deactivate; source activate {os.environ["HOME"]}/.conda/envs/shai-py37; nvidia-smi; '
               + f'cd {output_dir}; python -u {scipt_cmd}"']
    print(seq_cmd)

    # submit
    subprocess.call(' '.join(seq_cmd), shell=True)


if __name__ == '__main__':
    mirror_and_submit()

