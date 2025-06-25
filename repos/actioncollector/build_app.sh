#!/bin/bash

# Build ActionCollector Mac App with DMG
# This script handles the complete build process including:
# 1. Installing dependencies
# 2. Downloading ffmpeg
# 3. Building the app with PyInstaller
# 4. Creating a DMG file

set -e  # Exit on any error

echo "🚀 Building ActionCollector Mac App..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if we're in the right directory
if [[ ! -f "pyproject.toml" ]]; then
    print_error "pyproject.toml not found. Please run this script from the actioncollector directory."
    exit 1
fi

# Clean previous builds
print_step "Cleaning previous builds..."
rm -rf dist/ build/ *.dmg
print_success "Cleaned previous builds"

# Install/upgrade PyInstaller
print_step "Installing PyInstaller..."
uv add pyinstaller --dev
print_success "PyInstaller installed"

# Download ffmpeg binary
print_step "Downloading ffmpeg binary..."
python download_ffmpeg.py
print_success "FFmpeg binary ready"

# Build the app
print_step "Building ActionCollector.app..."
uv run pyinstaller actioncollector.spec --clean --noconfirm
print_success "App built successfully"

# Check if app was created
if [[ ! -d "dist/ActionCollector.app" ]]; then
    print_error "ActionCollector.app was not created successfully"
    exit 1
fi

print_success "ActionCollector.app created at dist/ActionCollector.app"

# Create DMG
print_step "Creating DMG file..."

# Get version from pyproject.toml
VERSION=$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])" 2>/dev/null || echo "0.1.0")
DMG_NAME="ActionCollector-${VERSION}.dmg"

# Create temporary directory for DMG contents
TMP_DMG_DIR="tmp_dmg"
rm -rf "$TMP_DMG_DIR"
mkdir "$TMP_DMG_DIR"

# Copy app to temp directory
cp -R "dist/ActionCollector.app" "$TMP_DMG_DIR/"

# Create Applications symlink
ln -s /Applications "$TMP_DMG_DIR/Applications"

# Create the DMG
hdiutil create -volname "ActionCollector" \
    -srcfolder "$TMP_DMG_DIR" \
    -ov -format UDZO \
    "$DMG_NAME"

# Clean up temp directory
rm -rf "$TMP_DMG_DIR"

print_success "DMG created: $DMG_NAME"

# Final verification
print_step "Verifying build..."
if [[ -f "$DMG_NAME" ]] && [[ -d "dist/ActionCollector.app" ]]; then
    print_success "Build completed successfully!"
    echo ""
    echo "📦 Outputs:"
    echo "   • App Bundle: dist/ActionCollector.app"
    echo "   • DMG File: $DMG_NAME"
    echo ""
    echo "🔑 Service Account Setup:"
    echo "   • Replace service-account-key.json with your actual GCS credentials"
    echo "   • Rebuild the app to include updated credentials"
    echo ""
    echo "🎯 To test the app:"
    echo "   • Double-click the DMG to mount it"
    echo "   • Drag ActionCollector.app to Applications"
    echo "   • Or run directly: open dist/ActionCollector.app --args --username=default"
    echo ""
    echo "🔧 To run with arguments:"
    echo "   dist/ActionCollector.app/Contents/MacOS/ActionCollector --username=default"
else
    print_error "Build verification failed"
    exit 1
fi