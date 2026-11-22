#!/bin/bash
# ESP32-S3 Recovery Flash Script for GLaDOS Spoke
# Usage: ./flash_spoke_recovery.sh /dev/ttyUSB0
#
# This script helps recover a bricked XIAO ESP32-S3 by:
# 1. Erasing flash completely
# 2. Flashing the GLaDOS firmware
#
# Before running:
# - Hold BOOT button on XIAO ESP32-S3
# - While holding BOOT, plug in USB-C
# - Release BOOT after USB connected
# - Check dmesg | tail for the correct serial port

set -e

SERIAL_PORT="${1:-/dev/ttyACM0}"
BUILD_DIR="./.pio/build/xiao_esp32s3"

echo "============================================"
echo "GLaDOS ESP32-S3 Spoke Recovery Flasher"
echo "============================================"
echo "Serial port: $SERIAL_PORT"
echo "Build dir:   $BUILD_DIR"
echo ""

# Check if esptool is available
if ! command -v esptool.py &> /dev/null; then
    echo "[ERROR] esptool.py not found. Install it:"
    echo "  pip3 install esptool"
    exit 1
fi

# Step 1: Erase flash completely
echo "[1/3] Erasing flash..."
esptool.py --port "$SERIAL_PORT" erase_flash
echo "      Flash erased!"

# Step 2: Flash the application
echo "[2/3] Flashing application..."
esptool.py --port "$SERIAL_PORT" --baud 921600 write_flash 0x0 "$BUILD_DIR/GLaDOS_Spoke.ino.bin"
echo "      Application flashed!"

# Step 3: Verify
echo "[3/3] Reading chip info..."
esptool.py --port "$SERIAL_PORT" chip_info
echo ""
echo "============================================"
echo "Done! Unplug and replug the USB-C cable."
echo "The board should boot normally now."
echo "Open serial monitor at 115200 baud to see output."
echo "============================================"
