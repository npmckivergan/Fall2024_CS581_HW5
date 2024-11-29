 
============================================================
=====     Queued job information at submit time        =====
============================================================
  The submitted file is: gpu_performance.sh
  The time limit is 01:00:00 HH:MM:SS.
  The target directory is: /home/ualclsd0176/Fall2024_CS581_HW5/scripts
  The memory limit is: 16gb
  The job will start running after: 202411282254.42
  Job Name: gpu_performance
  Queue: -q classgpu
  Constraints: 
  Command typed:
/scripts/run_gpu gpu_performance.sh     
  Queue submit command:
qsub -q classgpu -j oe -N gpu_performance -a 202411282254.42 -r n -M npmckivergan@crimson.ua.edu -l walltime=01:00:00 -l select=1:ngpus=1:ncpus=1:mpiprocs=1:mem=16000mb 
  Job number: 
 
