#!/bin/bash
# Slurm template for ct120-gui. The page fills {{job_name}}, {{log}}, {{workdir}},
# {{env}} and {{command}} (written here with single braces); everything else
# is yours. Resource lines can be
# overridden per job from the form (they are passed to sbatch after this
# script, and sbatch's command line wins over #SBATCH directives).
# {{command}} expands to the driver plus a few lines that forward Slurm's
# advance warning to it, and it ends the script: keep it last. Unless a
# --signal line is set here, jobs are submitted with --signal=B:USR1@600, so
# a run that would hit the time limit stops ten minutes early, writes its
# checkpoint and exports what it has. {{raw_command}} is the bare command.
#SBATCH --job-name={job_name}
#SBATCH --output={log}
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
# On an Alliance cluster add:  #SBATCH --account=def-yourpi
set -euo pipefail

# Environment the job needs; module loads and venv activation go here.
# module load StdEnv/2023 gcc cuda/12 opencv
# source ~/venv/bin/activate

cd {workdir}
{env}
{command}
