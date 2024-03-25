#!/bin/bash
#PBS -N myFirstJob
#PBS -l select=1:ncpus=8:mem=64gb:scratch_local=20gb
#PBS -l walltime=150:00:00 
#PBS -m ae
# The 4 lines above are options for scheduling system: job will run 1 hour at maximum, 1 machine with 4 processors + 4gb RAM memory + 10gb scratch memory are requested, email notification will be sent when the job aborts (a) or ends (e)
export OMP_NUM_THREADS=$PBS_NUM_PPN
module add python/3.8.0-gcc-rab6t

export PATH="/storage/brno12-cerit/home/jakublaza/.local/bin:$PATH"

wget 'https://bootstrap.pypa.io/get-pip.py'
python get-pip.py




pip install pandas scikit-learn 

# define a DATADIR variable: directory where the input files are taken from and where output will be copied to
DATADIR=/storage/plzen1/home/jakublaza

# append a line to a file "jobs_info.txt" containing the ID of the job, the hostname of node it is run on and the path to a scratch directory
# this information helps to find a scratch directory in case the job fails and you need to remove the scratch directory manually 
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt



python3 $DATADIR/MICE_ridge.py
