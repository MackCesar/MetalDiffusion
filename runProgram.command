#!/bin/bash

# Define ANSI escape codes
GREEN='\033[1;32m'
BLUE='\033[1;34m'
RESET='\033[0m'
BOLD='\033[1m'
UNDERLINE='\033[4m'

# Define variables for the frame
FRAME_WIDTH=50
FRAME_CHAR="-"
TITLE="Stable Diffusion & TensorFlow Keras"
ENDINGTITLE="Ending Program"

cd "$(dirname "$0")"  # Change to the directory of the script

source venv/bin/activate  # Activate the virtual environment

# Print the title frame
echo
printf "${BLUE}%${FRAME_WIDTH}s\n" "" | tr " " "${FRAME_CHAR}"
printf "${BLUE}%*s\n" $(((${#TITLE}+$FRAME_WIDTH)/2)) "$TITLE"
printf "${BLUE}%${FRAME_WIDTH}s\n" "" | tr " " "${FRAME_CHAR}"
echo

# Print a message with green color and bold formatting
echo -e "${GREEN}${BOLD}Launching the program...${RESET}"  
echo

python dream.py  # Run the Python program

echo
printf "${BLUE}%${FRAME_WIDTH}s\n" "" | tr " " "${FRAME_CHAR}"
printf "${BLUE}%*s\n" $(((${#ENDINGTITLE}+$FRAME_WIDTH)/2)) "$ENDINGTITLE"
printf "${BLUE}%${FRAME_WIDTH}s\n" "" | tr " " "${FRAME_CHAR}"
echo

osascript -e 'tell application "Terminal" to quit' &> /dev/null