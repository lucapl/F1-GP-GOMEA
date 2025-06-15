#!/bin/bash

#source ./set_env.sh

echo "READ and ADJUST this script file"

# Adjust this based on the number of CPU cores
max_processes=4
echo "Max processes: $max_processes"
echo "Adjust this num ^ to the number of CPU cores"

# Define experiments
experiments=$(seq 1 6)
echo "Experiments: $experiments"
echo "IMPORTANT: adjust this ^ to not overwrite old results"

popsizes=50
echo "Popsizes are: ${popsizes[@]}"

# Output folder
out_folder="./out_mut/"

# Create output directory if it doesn't exist
mkdir -p "$out_folder"

# Function to run a single job
run_job() {
    local experiment=$1
	local popsize=$2
	local full_output="${out_folder}/pop_${popsize}/${experiment}/"
	mkdir -p "$full_output"
    local logfile="out.log"
	local cwd=$(pwd)
    #echo "Starting job for linkage: $linkage, experiment: $experiment"
    
    # Run the command and redirect output to the log file
    (cd "$full_output" && python3.12 "$cwd/run_gomea_f1.py" \
	-n 100 \
	-e 7 \
	-g 100 \
	-p "$popsize" \
	-v \
	--count_nevals \
	--fmut 4 \
	--pmut 0.8 \
	--sim_location "$cwd/framspy" \
	--framslib "$cwd/Framsticks52" \
	--sims "eval-allcriteria.sim" "eval-once.sim" "recording-body-coords.sim" \
	> "$logfile" 2>&1)
}

# Export function and variables for parallel execution
export -f run_job
export out_folder

# Run jobs in parallel
queue_jobs(){
	local experiments="$1"
	local -n popsize_refs="$2"
	local max_processes="$3"
	joblist=()
	for experiment in $experiments; do
		for popsize in "${popsize_refs[@]}"; do
			run_job "$experiment" "$popsize" &
			joblist+=($!)
			if (( ${#joblist[@]} >= max_processes )); then
				wait -n
				joblist=($(jobs -p))  # Update joblist
			fi
		done
	done
	wait  # Wait for all jobs to finish
}

queue_jobs "$experiments" popsizes "$max_processes" &
queue_pid=$!

disown

echo "Jobs are running in the background. Queue process ID: $queue_pid"
echo "Use 'ps' to monitor the jobs or check logs in $out_folder."