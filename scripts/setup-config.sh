#!/bin/bash
# Clavi Agent Configuration Setup Script
# This script helps you set up Clavi Agent configuration files

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration directory
CONFIG_DIR="$HOME/.clavi-agent/config"

echo -e "${CYAN}╔════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   Clavi Agent Configuration Setup              ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Create config directory
echo -e "${BLUE}[1/3]${NC} Creating configuration directory..."
if [ -d "$CONFIG_DIR" ]; then
    # Auto backup existing config
    BACKUP_DIR="$HOME/.clavi-agent/config.backup.$(date +%Y%m%d_%H%M%S)"
    echo -e "${YELLOW}   Configuration directory exists, backing up to:${NC}"
    echo -e "${YELLOW}   $BACKUP_DIR${NC}"
    cp -r "$CONFIG_DIR" "$BACKUP_DIR"
    echo -e "${GREEN}   ✓ Backup created${NC}"
else
    mkdir -p "$CONFIG_DIR"
    echo -e "${GREEN}   ✓ Created: $CONFIG_DIR${NC}"
fi

# Step 2: Download configuration files from GitHub
echo -e "${BLUE}[2/3]${NC} Downloading configuration files..."

FILES_COPIED=0
GITHUB_RAW_URL="https://raw.githubusercontent.com/MiniMax-AI/clavi-agent/main/clavi_agent/config"

# Download config-example.yaml as config.yaml
if curl -fsSL "$GITHUB_RAW_URL/config-example.yaml" -o "$CONFIG_DIR/config.yaml" 2>/dev/null; then
    echo -e "${GREEN}   ✓ Downloaded: config.yaml${NC}"
    FILES_COPIED=$((FILES_COPIED + 1))
else
    echo -e "${RED}   ✗ Failed to download: config.yaml${NC}"
fi

# Download mcp.json (optional)
if curl -fsSL "$GITHUB_RAW_URL/mcp.json" -o "$CONFIG_DIR/mcp.json" 2>/dev/null; then
    echo -e "${GREEN}   ✓ Downloaded: mcp.json${NC}"
    FILES_COPIED=$((FILES_COPIED + 1))
fi

# Download system_prompt.md (optional)
if curl -fsSL "$GITHUB_RAW_URL/system_prompt.md" -o "$CONFIG_DIR/system_prompt.md" 2>/dev/null; then
    echo -e "${GREEN}   ✓ Downloaded: system_prompt.md${NC}"
    FILES_COPIED=$((FILES_COPIED + 1))
fi

if [ $FILES_COPIED -eq 0 ]; then
    echo -e "${RED}   ✗ Failed to download configuration files${NC}"
    echo -e "${YELLOW}   Please check your internet connection${NC}"
    exit 1
fi

echo -e "${GREEN}   ✓ Configuration files ready${NC}"

# Step 3: Install runtime dependencies and persist tool paths
echo -e "${BLUE}[3/3]${NC} Installing runtime dependencies..."
if command -v clavi-agent-setup-runtime >/dev/null 2>&1; then
    clavi-agent-setup-runtime --config "$CONFIG_DIR/config.yaml"
    echo -e "${GREEN}   ✓ Runtime dependencies ready${NC}"
else
    echo -e "${RED}   ✗ clavi-agent-setup-runtime command not found${NC}"
    echo -e "${YELLOW}   Please ensure Clavi Agent is installed before running this setup script${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Setup Complete! ✨                          ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Configuration files location:"
echo -e "  ${CYAN}$CONFIG_DIR${NC}"
echo ""
echo -e "Files:"
ls -1 "$CONFIG_DIR" 2>/dev/null | sed 's/^/  📄 /' || echo "  (no files yet)"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo ""
echo -e "${YELLOW}1. Configure your API Key:${NC}"
echo -e "   Edit config.yaml and add your MiniMax API Key:"
echo -e "   ${GREEN}nano $CONFIG_DIR/config.yaml${NC}"
echo -e "   ${GREEN}vim $CONFIG_DIR/config.yaml${NC}"
echo -e "   ${GREEN}code $CONFIG_DIR/config.yaml${NC}"
echo ""
echo -e "${YELLOW}2. Start using Clavi Agent:${NC}"
echo -e "   ${GREEN}clavi-agent${NC}                              # Use current directory"
echo -e "   ${GREEN}clavi-agent --workspace /path/to/project${NC} # Specify workspace"
echo -e "   ${GREEN}clavi-agent --help${NC}                      # Show help"
echo ""



