#!/bin/bash

# Initialize default log file
LOG_FILE="cli_logs.txt"

# Initialize empty string for environment variables and command
ENV_VARS=""
CMD=""
exporting=true

# Parse arguments to find --output_dir and construct the command
for arg in "$@"; do
  if [ "$exporting" = true ] && [[ "$arg" == *=* ]]; then
    ENV_VARS+="export $arg; "
  elif [ "$arg" == "python3" ]; then
    exporting=false
  fi

  if [[ "$arg" == "--output_dir" ]]; then
    flag_found=true
  elif [[ "$flag_found" == true ]]; then
    LOG_FILE="$arg/cli_logs.txt"
    mkdir -p "$arg"  # Create the directory if it doesn't exist
    flag_found=false
  fi

  if [ "$exporting" = false ]; then
    CMD+="'$arg' "
  fi
done

# Run the provided command and capture logs
echo $ENV_VARS
eval "$ENV_VARS $CMD" 2>&1 | tee -a "$LOG_FILE"
