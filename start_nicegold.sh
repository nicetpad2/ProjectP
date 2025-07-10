#!/bin/bash
# NICEGOLD ProjectP Enterprise Production Launcher
# Ensures proper environment activation and dependency resolution

cd "$(dirname "$0")"

echo "🚀 NICEGOLD ProjectP - Enterprise Production Launcher"
echo "====================================================="

# Activate environment if script exists
if [ -f "./activate_nicegold_env.sh" ]; then
    echo "📦 Activating NICEGOLD environment..."
    source ./activate_nicegold_env.sh
    export NICEGOLD_ENV_ACTIVATED=1
else
    echo "⚠️ Environment script not found, using system Python"
    export NICEGOLD_ENV_ACTIVATED=0
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+"
    exit 1
fi

echo "� Python executable: $(which python3)"
echo "📍 Project directory: $(pwd)"

# Launch with comprehensive error handling
echo "🎯 Launching enterprise production system..."

if [ -f "./production_launcher.py" ]; then
    echo "✅ Using enterprise production launcher"
    python3 ./production_launcher.py
    LAUNCH_RESULT=$?
elif [ -f "./launch_nicegold.py" ]; then
    echo "⚠️ Using backup launcher"
    python3 ./launch_nicegold.py
    LAUNCH_RESULT=$?
elif [ -f "./ProjectP_optimized_final.py" ]; then
    echo "⚠️ Using direct system launcher"
    python3 ./ProjectP_optimized_final.py
    LAUNCH_RESULT=$?
else
    echo "❌ No launcher found. Critical system error."
    exit 1
fi

# Report results
if [ $LAUNCH_RESULT -eq 0 ]; then
    echo "✅ NICEGOLD ProjectP launched successfully"
else
    echo "❌ Launch failed with exit code: $LAUNCH_RESULT"
    echo "📋 Check logs for details:"
    echo "  - logs/production_launcher.log"
    echo "  - dependency_installation.log"
fi

echo "🏁 Launch sequence completed"
exit $LAUNCH_RESULT
        echo "🐍 Python: $(which python3)"
        
        # Launch optimized system
        echo "🚀 Launching NICEGOLD Enterprise System..."
        python3 ProjectP_optimized_final.py
    else
        echo "❌ Environment activation failed"
        echo "🔧 Trying direct launch..."
        python3 launch_nicegold.py
    fi
else
    echo "⚠️ Activation script not found"
    echo "🔧 Trying direct launch..."
    python3 launch_nicegold.py
fi

echo "✅ NICEGOLD Enterprise wrapper completed"
