#!/bin/bash
# Monitor OSCAR training job in real-time
# Usage: ./monitor_oscar_training.sh [USERNAME]
#
# Run this from your local machine or another OSCAR terminal

USERNAME=${1:-$USER}
OSCAR_HOST="ssh.ccv.brown.edu"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=========================================="
echo "OSCAR Training Monitor"
echo "=========================================="
echo "Monitoring user: $USERNAME"
echo "Press Ctrl+C to exit"
echo ""

# Function to get latest output directory
get_latest_dir() {
    ssh ${USERNAME}@${OSCAR_HOST} "ls -td ~/urbanize/results/conditional_wealthy_* 2>/dev/null | head -1" 2>/dev/null
}

# Function to display job status
show_job_status() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}📋 JOB STATUS - $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Get job info
    JOB_INFO=$(ssh ${USERNAME}@${OSCAR_HOST} "squeue -u ${USERNAME} -o '%.18i %.9P %.30j %.8T %.10M %.6D %R'" 2>/dev/null)

    if [ -z "$JOB_INFO" ] || [ "$(echo "$JOB_INFO" | wc -l)" -eq 1 ]; then
        echo -e "${YELLOW}⚠️  No running jobs found${NC}"
        echo ""
        echo "Recent completed jobs:"
        ssh ${USERNAME}@${OSCAR_HOST} "sacct -u ${USERNAME} --format=JobID,JobName%30,State,Elapsed,End -n | head -5" 2>/dev/null
    else
        echo "$JOB_INFO"
    fi
    echo ""
}

# Function to display training progress
show_training_progress() {
    LATEST_DIR=$(get_latest_dir)

    if [ -z "$LATEST_DIR" ]; then
        echo -e "${YELLOW}⚠️  No training output directory found${NC}"
        return
    fi

    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}📊 TRAINING PROGRESS${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "Output directory: ${LATEST_DIR##*/}"
    echo ""

    # Check if metrics file exists
    METRICS_EXISTS=$(ssh ${USERNAME}@${OSCAR_HOST} "test -f ${LATEST_DIR}/metrics.csv && echo 'yes' || echo 'no'" 2>/dev/null)

    if [ "$METRICS_EXISTS" = "yes" ]; then
        # Get epoch count
        EPOCHS_DONE=$(ssh ${USERNAME}@${OSCAR_HOST} "tail -n +2 ${LATEST_DIR}/metrics.csv 2>/dev/null | wc -l | tr -d ' '" 2>/dev/null)

        if [ -n "$EPOCHS_DONE" ] && [ "$EPOCHS_DONE" -gt 0 ]; then
            PROGRESS_PCT=$((EPOCHS_DONE * 100 / 100))

            echo -e "${GREEN}Epochs completed: ${EPOCHS_DONE}/100 (${PROGRESS_PCT}%)${NC}"

            # Progress bar
            FILLED=$((PROGRESS_PCT / 2))
            BAR="["
            for ((i=0; i<50; i++)); do
                if [ $i -lt $FILLED ]; then
                    BAR="${BAR}█"
                else
                    BAR="${BAR}░"
                fi
            done
            BAR="${BAR}]"
            echo "$BAR"
            echo ""

            # Show last 5 epochs
            echo "Recent epochs:"
            ssh ${USERNAME}@${OSCAR_HOST} "tail -n 6 ${LATEST_DIR}/metrics.csv 2>/dev/null | tail -n 5 | awk -F',' '{printf \"  Epoch %3d: G_Loss=%.4f D_Loss=%.4f Time=%.1fs\\n\", \$1, \$2, \$3, \$4}'" 2>/dev/null
            echo ""

            # Calculate ETA
            AVG_TIME=$(ssh ${USERNAME}@${OSCAR_HOST} "tail -n +2 ${LATEST_DIR}/metrics.csv 2>/dev/null | awk -F',' '{sum+=\$4; count++} END {if(count>0) print sum/count; else print 0}'" 2>/dev/null)
            REMAINING=$((100 - EPOCHS_DONE))
            if [ -n "$AVG_TIME" ] && [ "$REMAINING" -gt 0 ]; then
                ETA_SECONDS=$(echo "$AVG_TIME * $REMAINING" | bc)
                ETA_MINS=$(echo "$ETA_SECONDS / 60" | bc)
                ETA_HOURS=$(echo "$ETA_MINS / 60" | bc)
                echo -e "${YELLOW}Estimated time remaining: ${ETA_MINS} minutes (~${ETA_HOURS} hours)${NC}"
            fi
        else
            echo "Waiting for training to start..."
        fi
    else
        echo "Metrics file not yet created. Training may be initializing..."
    fi
    echo ""
}

# Function to show recent log output
show_recent_logs() {
    LATEST_DIR=$(get_latest_dir)

    if [ -z "$LATEST_DIR" ]; then
        return
    fi

    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}📝 RECENT LOG OUTPUT${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Find the training log file
    LOG_FILE=$(ssh ${USERNAME}@${OSCAR_HOST} "ls -t ${LATEST_DIR}/training_log_*.txt 2>/dev/null | head -1" 2>/dev/null)

    if [ -n "$LOG_FILE" ]; then
        ssh ${USERNAME}@${OSCAR_HOST} "tail -n 15 ${LOG_FILE} 2>/dev/null" 2>/dev/null
    else
        echo "No log file found yet"
    fi
    echo ""
}

# Function to show GPU usage
show_gpu_usage() {
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}🖥️  GPU USAGE${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Get the node where job is running
    NODE=$(ssh ${USERNAME}@${OSCAR_HOST} "squeue -u ${USERNAME} -h -o '%N' | head -1" 2>/dev/null)

    if [ -n "$NODE" ] && [ "$NODE" != "" ]; then
        echo "Running on node: $NODE"
        # Try to get GPU info (may require special permissions)
        ssh ${USERNAME}@${OSCAR_HOST} "ssh $NODE 'nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader' 2>/dev/null || echo 'GPU info not accessible from login node'" 2>/dev/null
    else
        echo "No active job found"
    fi
    echo ""
}

# Main monitoring loop
while true; do
    clear
    show_job_status
    show_training_progress
    show_recent_logs
    show_gpu_usage

    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo "Auto-refreshing every 30 seconds..."
    echo "Press Ctrl+C to exit"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    sleep 30
done

