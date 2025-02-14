#!/bin/bash
#SBATCH --job-name=UDL_flow
#SBATCH --output=UDL_flow.out
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mem=12GB
#SBATCH --partition=gpu
#SBATCH --signal=B:12@600

mkdir -p $TMPDIR/results

# Compress and save the results if the timelimit is close
trap 'mkdir -p /home4/$USER/job_${SLURM_JOBID}; tar czvf /home4/$USER/job_${SLURM_JOBID}/results.tar.gz $TMPDIR/results' 12

# Copy to TPMDIR
cp -R /home4/$USER/UDL-Project $TMPDIR/UDL-Project
cd $TMPDIR/UDL-Project

# Run the training
/home4/$USER/venvs/udl_envs/bin/python3 $TMPDIR/UDL-Project/src/main_flow_training.py &
wait

mkdir -p /home4/$USER/job_${SLURM_JOBID}
tar czvf /home4/$USER/job_${SLURM_JOBID}/results.tar.gz $TMPDIR/results