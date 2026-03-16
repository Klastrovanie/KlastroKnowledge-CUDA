#!/bin/bash

if [ -z "$1" ]; then
  echo "❌ Usage: ./make_package.sh cuda11|cuda12|cuda13"
  exit 1
fi

CUDA_VERSION=$1
RELEASE_DIR="klastroknowledge_${CUDA_VERSION}_release"
DEPLOY_DIR="deploy_exercise"
PKG_NAME="klastroknowledge"

# 1. building
echo "🔧 Building CUDA extension..."
python build.py build_ext --inplace

# 2. .so 탐색
SO_FILE=$(find . -name "${PKG_NAME}*.so" | head -n 1)
if [ ! -f "$SO_FILE" ]; then
  echo "❌ .so file not found. Build may have failed."
  exit 1
fi

# 3. making directory
mkdir -p "$DEPLOY_DIR/$PKG_NAME"
mkdir -p "$RELEASE_DIR/$PKG_NAME"

# 4. copying
cp "$SO_FILE" "$DEPLOY_DIR/$PKG_NAME/"
cp "$SO_FILE" "$RELEASE_DIR/$PKG_NAME/"

cp tools/__init__.py "$DEPLOY_DIR/$PKG_NAME/"
cp tools/__init__.py "$RELEASE_DIR/$PKG_NAME/"

cp tools/setup.py "$DEPLOY_DIR/"
cp tools/setup.py "$RELEASE_DIR/"

echo "✅ Build and packaging completed successfully."
