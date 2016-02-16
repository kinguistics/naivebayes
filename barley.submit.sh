#!/bin/bash

# Name the job in Grid Engine
#$ -N <job-name>

#tell grid engine to use current directory
#$ -cwd

# Set Email Address where notifications are to be sent
#$ -M etking@stanford.edu

# Tell Grid Engine to notify job owner if job 'b'egins, 'e'nds, 's'uspended is 'a'borted, or 'n'o mail
#$ -m besan

# Tel Grid Engine to join normal output and error output into one file 
#$ -j y


## the "meat" of the script

# set ticket and get a token
kinit
aklog
tokens

# start the script and grab its pid
python test_high_low_freq_words.py $* &
pid=$!

# poll the pid every 12 hours, and renew the ticket/token if it's still running
while ps -p $pid >/dev/null
do
    sleep 12h
    kinit
    aklog
    tokens
done

