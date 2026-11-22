#!/bin/bash
# ============================================================
# XVF3800 I2S-Master Firmware Flash Script
# ============================================================
# Prerequisite: The XVF3800 must be flashed to I2S-master mode
# BEFORE the ESP32-S3 can communicate with it.
#
# This is a ONE-TIME step. After flashing, the XVF3800 will
# output I2S on the socket pins and accept I2C control.
#
# Firmware from Seeed Studio GitHub:
# https://github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY
# ============================================================

set -e

# Direct download URL for the I2S-master firmware
FIRMWARE_URL="https://github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY/raw/master/xmos_firmwares/i2s/respeaker_xvf3800_i2s_master_v1.0.9_48k.bin"
FIRMWARE_FILE="respeaker_xvf3800_i2s_master_v1.0.9_48k.bin"

echo "============================================"
echo "XVF3800 I2S-Master Firmware Flasher"
echo "============================================"
echo ""

# Check for dfu-util
if ! command -v dfu-util &> /dev/null; then
    echo "[ERROR] dfu-util not found. Install it first:"
    echo ""
    echo "  Ubuntu/Debian:"
    echo "    sudo apt install dfu-util"
    echo ""
    echo "  macOS:"
    echo "    brew install dfu-util"
    echo ""
    echo "  Then retry this script."
    exit 1
fi

echo "[1/4] Checking for dfu-util... OK"

# Download firmware
echo "[2/4] Downloading I2S-master firmware..."
echo "      URL: $FIRMWARE_URL"

if [ ! -f "$FIRMWARE_FILE" ]; then
    if curl -sL -o "$FIRMWARE_FILE" "$FIRMWARE_URL"; then
        echo "      Downloaded successfully!"
    else
        echo "      Download failed!"
        echo ""
        echo "Please download manually from:"
        echo "  $FIRMWARE_URL"
        echo ""
        echo "Then run this script again from the same directory."
        exit 1
    fi
else
    echo "      Firmware file already exists: $FIRMWARE_FILE"
fi

FIRMWARE_PATH="${1:-$FIRMWARE_FILE}"

if [ ! -f "$FIRMWARE_PATH" ]; then
    echo "[ERROR] Firmware file not found: $FIRMWARE_PATH"
    exit 1
fi

echo "[3/4] Enter DFU mode on your XVF3800 board:"
echo ""
echo "  1. Plug the XVF3800 into USB via its own USB-C port"
echo "  2. PRESS AND HOLD the MUTE button on the XVF3800"
echo "  3. WHILE HOLDING MUTE, plug in the USB cable"
echo "  4. KEEP HOLDING until you see the DFU device"
echo ""

echo "  Checking for DFU device..."
DFU_FOUND=false
for i in $(seq 1 15); do
    DFU_OUTPUT=$(dfu-util --list 2>/dev/null)
    if echo "$DFU_OUTPUT" | grep -q "2886\|20b1"; then
        echo "  [OK] DFU device found!"
        echo "$DFU_OUTPUT" | grep -E "Found DFU:.*alt=" | head -5
        DFU_FOUND=true
        break
    fi
    if [ "$i" -eq 15 ]; then
        echo "  [!] No DFU device found (VID 0x2886 or 0x20b1)."
        echo "  Make sure you're holding MUTE while plugging in USB."
        echo ""
        echo "  Press ENTER when the XVF3800 is in DFU mode, or Ctrl+C to cancel."
        read -r
        break
    fi
    sleep 1
done

if [ "$DFU_FOUND" != "true" ]; then
    echo ""
    echo "  Your DFU device is already detected. Proceeding with flash..."
fi

echo ""
echo "[4/4] Flashing firmware..."
echo ""
echo "  Note: Your board has 3 partitions:"
echo "    alt=0 = Factory  (read-only)"
echo "    alt=1 = Upgrade  (write firmware here)"
echo "    alt=2 = Data     (user data)"
echo ""

# The I2S-master firmware goes to the Upgrade partition (alt=1)
# Some older boards use alt=0, so we try alt=1 first, fall back to alt=0
echo "  Attempting to flash to alt=1 (Upgrade partition)..."

if dfu-util -d 0x2886: -a 1 -D "$FIRMWARE_PATH" -R 2>/dev/null; then
    echo "  [OK] Flashed successfully via alt=1 (device 0x2886)!"
elif dfu-util -d 0x20b1: -a 1 -D "$FIRMWARE_PATH" -R 2>/dev/null; then
    echo "  [OK] Flashed successfully via alt=1 (device 0x20b1)!"
else
    echo "  [!] alt=1 failed. Trying alt=0 (Factory partition)..."
    if dfu-util -d 0x2886: -a 0 -D "$FIRMWARE_PATH" -R 2>/dev/null; then
        echo "  [OK] Flashed successfully via alt=0 (device 0x2886)!"
    elif dfu-util -d 0x20b1: -a 0 -D "$FIRMWARE_PATH" -R 2>/dev/null; then
        echo "  [OK] Flashed successfully via alt=0 (device 0x20b1)!"
    else
        echo ""
        echo "  [!] Flash failed on both alt=0 and alt=1."
        echo ""
        echo "  Try these manually:"
        echo ""
        echo "    # Try alt=1 (Upgrade partition - RECOMMENDED):"
        echo "    sudo dfu-util -d 0x2886: -a 1 -D $FIRMWARE_PATH -R"
        echo ""
        echo "    # If that fails, try alt=0 (Factory partition):"
        echo "    sudo dfu-util -d 0x2886: -a 0 -D $FIRMWARE_PATH -R"
        echo ""
        echo "  Then unplug and replug the XVF3800 USB to exit DFU mode."
        exit 1
    fi
fi

echo ""
echo "============================================"
echo "Done! XVF3800 is now running I2S-master firmware."
echo ""
echo "Next steps:"
echo "  1. Unplug the XVF3800 USB cable"
echo "  2. Socket the XIAO ESP32-S3 onto the XVF3800"
echo "  3. Flash the ESP32-S3 with GLaDOS firmware"
echo "  4. Open serial monitor at 115200 baud"
echo "  5. Look for 'XVF3800 I2C bus up' in boot log"
echo "============================================"
