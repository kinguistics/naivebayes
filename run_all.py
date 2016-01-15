import os

os.system('qsub -cwd -l longq=1 -N nb_tests ./barley.submit.sh')
