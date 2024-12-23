#!/bin/bash

# Define the base directory
BASE_DIR="."

# Define the directories to create
DIRECTORIES=(
  "$BASE_DIR/server/models"
  "$BASE_DIR/server/utils"
  "$BASE_DIR/client/workflows"
  "$BASE_DIR/client/utils"
  "$BASE_DIR/client/resources/example_images"
  "$BASE_DIR/env"
)

# Define the files to create
FILES=(
  "$BASE_DIR/README.md"
  "$BASE_DIR/env/environment.yml"
  "$BASE_DIR/server/main.py"
  "$BASE_DIR/server/requirements.txt"
  "$BASE_DIR/server/utils/image_processing.py"
  "$BASE_DIR/client/automation_client.py"
  "$BASE_DIR/client/requirements.txt"
  "$BASE_DIR/client/utils/api_client.py"
  "$BASE_DIR/client/workflows/screenshot_workflow.py"
  "$BASE_DIR/client/workflows/click_workflow.py"
)

# Create directories
echo "Creating directories..."
for DIR in "${DIRECTORIES[@]}"; do
  mkdir -p "$DIR"
  echo "Created: $DIR"
done

# Create files
echo "Creating files..."
for FILE in "${FILES[@]}"; do
  touch "$FILE"
  echo "Created: $FILE"
done

echo "File structure created successfully!"
