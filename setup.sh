#!/bin/bash
# OpenHands Max Setup Script
# This script configures API keys from .env file

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🤖 OpenHands Max Setup"
echo "======================"
echo

# Check if .env exists
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${YELLOW}⚠️  .env file not found${NC}"
    echo "Creating from .env.example..."
    
    if [ -f "$SCRIPT_DIR/.env.example" ]; then
        cp "$SCRIPT_DIR/.env.example" "$ENV_FILE"
        echo -e "${GREEN}✅ Created .env from .env.example${NC}"
        echo -e "${YELLOW}📝 Please edit .env and add your API keys, then run this script again${NC}"
        exit 0
    else
        echo -e "${RED}❌ .env.example not found${NC}"
        exit 1
    fi
fi

# Load .env
echo "📁 Loading .env..."
export $(grep -v '^#' "$ENV_FILE" | xargs)

# Check required keys
missing_keys=()

if [ -z "$ANTHROPIC_API_KEY" ] || [ "$ANTHROPIC_API_KEY" = "sk-ant-api03-your-key-here" ]; then
    missing_keys+=("ANTHROPIC_API_KEY")
fi

if [ -z "$KIMI_API_KEY" ] || [ "$KIMI_API_KEY" = "sk-your-moonshot-key-here" ]; then
    missing_keys+=("KIMI_API_KEY")
fi

if [ -z "$TAVILY_API_KEY" ] || [ "$TAVILY_API_KEY" = "tvly-your-key-here" ]; then
    missing_keys+=("TAVILY_API_KEY")
fi

if [ ${#missing_keys[@]} -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Missing or placeholder API keys:${NC}"
    for key in "${missing_keys[@]}"; do
        echo "   - $key"
    done
    echo
    echo "Please edit .env and add your real API keys."
    echo "You can still use the tool with the models that have valid keys."
    echo
fi

# Substitute variables in template files
echo "🔧 Configuring templates..."

# Function to substitute env vars in file
substitute_env() {
    local file="$1"
    if [ -f "$file" ]; then
        # Create backup
        cp "$file" "${file}.bak"
        
        # Substitute variables
        envsubst < "${file}.bak" > "$file"
        rm "${file}.bak"
        
        echo "   ✅ $(basename $(dirname $file))/$(basename $file)"
    fi
}

# Process LLM templates
for dir in "$SCRIPT_DIR/templates/llm"/*/; do
    settings_file="${dir}agent_settings.json"
    if [ -f "$settings_file" ]; then
        substitute_env "$settings_file"
    fi
done

# Process MCP config
substitute_env "$SCRIPT_DIR/templates/mcp/mcp_servers.json"

echo
echo -e "${GREEN}✅ Setup complete!${NC}"
echo
echo "Next steps:"
echo "  1. Run: python openhands.py"
echo "  2. Create a new project or select existing"
echo "  3. Start coding with AI!"
echo
