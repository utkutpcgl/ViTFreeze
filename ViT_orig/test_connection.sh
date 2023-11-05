#!/bin/bash

# Define the remote host, user, and the password
REMOTE_HOST="192.168.1.100"
REMOTE_USER="kuartis-dgx1"
PASSWORD="kuartis2012"
TARGET_PID=2659044

# Test SSH connection using sshpass
# sshpass -p "kuartis2012" ssh -o StrictHostKeyChecking=no kuartis-dgx1@192.168.1.100
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no ${REMOTE_USER}@${REMOTE_HOST} "echo 'SSH connection established!'"

# Check the exit status of the SSH command
if [ $? -eq 0 ]; then
    echo "SSH connection test succeeded."
else
    echo "SSH connection test failed."
fi
