#!/bin/bash
# ==============================================================================
# Launch WandB Sweep Agents on Multiple GPUs
# ==============================================================================
#
# Usage:
#   ./launch_sweep_agents.sh <num_gpus> [sweep_id]
#
# Examples:
#   # Create new sweep and launch 4 agents
#   ./launch_sweep_agents.sh 4
#
#   # Launch 4 agents for existing sweep
#   ./launch_sweep_agents.sh 4 abc123xyz
#
# ==============================================================================

set -e

# Parse arguments
NUM_GPUS=${1:-4}
SWEEP_ID=${2:-""}
PROJECT=rdblearn-scripts
ENTITY=tgif

# Get script directory (for relative paths)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="${PROJECT_ROOT}/config"
SWEEP_CONFIG="${CONFIG_DIR}/sweep_config.yaml"

# Enable virtual environment if it exists
if [ -d "${PROJECT_ROOT}/.venv" ]; then
    echo "Activating virtual environment at ${PROJECT_ROOT}/.venv"
    source "${PROJECT_ROOT}/.venv/bin/activate"
else
    echo "Warning: No .venv found at ${PROJECT_ROOT}. Using system python."
fi

echo "============================================================"
echo "RDBLearn WandB Sweep Launcher"
echo "============================================================"
echo "  Project root: ${PROJECT_ROOT}"
echo "  Sweep config: ${SWEEP_CONFIG}"
echo "  Number of GPUs: ${NUM_GPUS}"
echo "  Project: ${PROJECT}"
echo "  Entity: ${ENTITY:-'(default)'}"
echo "============================================================"

# Check if wandb is installed
if ! command -v wandb &> /dev/null; then
    echo "Error: wandb is not installed. Install with: pip install wandb"
    exit 1
fi

# Check if logged in to wandb
if ! wandb login --verify &> /dev/null 2>&1; then
    echo "Warning: Not logged in to wandb. Run 'wandb login' first."
fi

# Create sweep if no sweep ID provided
if [ -z "$SWEEP_ID" ]; then
    echo ""
    echo "Creating new sweep from config: ${SWEEP_CONFIG}"
    
    if [ ! -f "$SWEEP_CONFIG" ]; then
        echo "Error: Sweep config not found at ${SWEEP_CONFIG}"
        exit 1
    fi
    
    # Create sweep and capture the sweep ID
    if [ -n "$ENTITY" ]; then
        SWEEP_OUTPUT=$(wandb sweep --project "$PROJECT" --entity "$ENTITY" "$SWEEP_CONFIG" 2>&1)
    else
        SWEEP_OUTPUT=$(wandb sweep --project "$PROJECT" "$SWEEP_CONFIG" 2>&1)
    fi
    
    echo "$SWEEP_OUTPUT"
    
    # Extract sweep ID from output (format varies, try multiple patterns)
    SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -oP '(?<=wandb agent )[^\s]+' | tail -1 | cut -d'/' -f3)
    
    if [ -z "$SWEEP_ID" ]; then
        # Alternative pattern
        SWEEP_ID=$(echo "$SWEEP_OUTPUT" | grep -oP '[a-z0-9]{8}' | tail -1)
    fi
    
    if [ -z "$SWEEP_ID" ]; then
        echo "Error: Could not extract sweep ID from output"
        echo "Please create the sweep manually and provide the sweep ID"
        exit 1
    fi
    
    echo ""
    echo "Created sweep with ID: ${SWEEP_ID}"
fi

echo ""
echo "============================================================"
echo "Launching ${NUM_GPUS} agents for sweep: ${SWEEP_ID}"
echo "============================================================"

# Build the agent path
if [ -n "$ENTITY" ]; then
    AGENT_PATH="${ENTITY}/${PROJECT}/${SWEEP_ID}"
    AGENT_URL="https://wandb.ai/${ENTITY}/${PROJECT}/sweeps/${SWEEP_ID}"
else
    AGENT_PATH="${PROJECT}/${SWEEP_ID}"
    AGENT_URL="https://wandb.ai/${PROJECT}/sweeps/${SWEEP_ID}"
fi

# Change to project root directory (required for relative imports in run_sweep_experiment.py)
cd "$PROJECT_ROOT"

# Array to store PIDs of spawned agents
AGENT_PIDS=()

# Trap to kill all agents on script exit
cleanup() {
    echo ""
    echo "Stopping all agents..."
    for pid in "${AGENT_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    echo "All agents stopped."
    exit 0
}

trap cleanup SIGINT SIGTERM

# Launch agents on each GPU
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    echo "Starting agent on GPU ${GPU_ID}..."
    
    # Launch agent with specific GPU, redirect output to log file
    LOG_FILE="/tmp/wandb_agent_gpu${GPU_ID}_$(date +%Y%m%d_%H%M%S).log"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID wandb agent "$AGENT_PATH" > "$LOG_FILE" 2>&1 &
    
    AGENT_PID=$!
    AGENT_PIDS+=($AGENT_PID)
    
    echo "  Agent PID: ${AGENT_PID}, Log: ${LOG_FILE}"
    
    # Small delay between launches to avoid race conditions
    sleep 2
done

echo ""
echo "============================================================"
echo "All ${NUM_GPUS} agents launched successfully!"
echo "============================================================"
echo ""
echo "Sweep URL: ${AGENT_URL}"
echo ""
echo "To monitor logs:"
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    echo "  GPU ${GPU_ID}: tail -f /tmp/wandb_agent_gpu${GPU_ID}_*.log"
done
echo ""
echo "Press Ctrl+C to stop all agents"
echo ""

# Wait for all agents to complete
wait

echo ""
echo "If you pressed Ctrl+C, please run 'wandb agent --cancel ${AGENT_PATH}' to stop the agents."