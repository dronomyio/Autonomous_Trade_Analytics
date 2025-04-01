#!/bin/bash
# Quick start script for Trade Analytics

# Set up color outputs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}"
echo "----------------------------------------"
echo "  Trade Analytics Platform Quick Start  "
echo "----------------------------------------"
echo -e "${NC}"

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}No .env file found. Creating from example...${NC}"
    if [ -f .env.example ]; then
        cp .env.example .env
        echo -e "${GREEN}.env file created!${NC}"
        echo -e "${YELLOW}Please edit .env file to add your ANTHROPIC_API_KEY${NC}"
        read -p "Do you want to edit the .env file now? (y/n): " edit_env
        if [[ $edit_env == "y" || $edit_env == "Y" ]]; then
            if command -v nano > /dev/null; then
                nano .env
            elif command -v vim > /dev/null; then
                vim .env
            else
                echo -e "${RED}No editor found. Please edit the .env file manually.${NC}"
            fi
        fi
    else
        echo -e "${RED}.env.example file not found!${NC}"
        echo "ANTHROPIC_API_KEY=your_anthropic_api_key_here" > .env
        echo -e "${GREEN}Created blank .env file. Please add your API key.${NC}"
    fi
fi

# Check if Docker is installed
if ! command -v docker > /dev/null; then
    echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
    echo "Visit https://docs.docker.com/get-docker/ for installation instructions."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose > /dev/null; then
    echo -e "${RED}Docker Compose is not installed. Please install Docker Compose first.${NC}"
    echo "Visit https://docs.docker.com/compose/install/ for installation instructions."
    exit 1
fi

# Ask if user wants to start fresh or resume
if [ "$(docker ps -a -q -f name=trade-agent)" ]; then
    echo -e "${YELLOW}Existing containers found.${NC}"
    read -p "Do you want to restart from scratch? (y/n): " restart
    if [[ $restart == "y" || $restart == "Y" ]]; then
        echo -e "${BLUE}Stopping and removing existing containers...${NC}"
        docker-compose down
    fi
fi

# Build and start containers
echo -e "${BLUE}Building and starting containers...${NC}"
docker-compose up -d

echo -e "${GREEN}Containers started!${NC}"
echo -e "${BLUE}Trade Agent API is running at:${NC} http://localhost:8000"
echo -e "${BLUE}Dashboard is running at:${NC} http://localhost:8080"
echo ""
echo -e "${YELLOW}To view logs:${NC} docker-compose logs -f"
echo -e "${YELLOW}To stop:${NC} docker-compose down"
echo ""
echo -e "${GREEN}Happy trading!${NC}"