# Flash GLaDOS ESP32-S3 Spoke via Arduino IDE

## ⚠️ CRITICAL PREREQUISITE: Flash XVF3800 to I2S-Master Mode

**The XVF3800 ships from the factory with USB-audio firmware ONLY.**
You MUST flash the I2S-master variant before the ESP32-S3 can communicate with it.

### Step 0: Flash XVF3800 I2S-Master Firmware

```bash
# Install dfu-util
sudo apt install dfu-util        # Ubuntu/Debian
brew install dfu-util            # macOS

# Run the flash script (or see README.md for manual steps)
cd /opt/GLaDOS/spokes/esp32-spoke
chmod +x flash_xvf3800_i2s_master.sh
sudo ./flash_xvf3800_i2s_master.sh
```

**Manual DFU mode entry:**
1. Plug the XVF3800 USB-C into your PC
2. **Hold the MUTE button** on the XVF3800
3. **While holding MUTE**, plug in the USB cable
4. Keep holding until `dfu-util --list` shows VID `0x20b1`
5. Flash: `dfu-util -d 0x20b1: -a 0 -D <firmware.bin> -R`

### Verify XVF3800 is working (optional)

Before socketing the XIAO, test the XVF3800 on a Linux PC:
```bash
# Should show 4 channels of mic input
arecord -f S32_LE -r 48000 -c 4 test.wav -d 5
```

---

## Flashing the ESP32-S3

### On Your Laptop

### 1. Install ESP32 Board Package
- Open Arduino IDE
- Go to **File → Preferences**
- In **Additional Board Manager URLs**, add:
  ```
  https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
  ```
- Go to **Tools → Board → Board Manager**
- Search "ESP32" and install **ESP32 by Espressif**

### 2. Select Your Board
- **Tools → Board → Seeed Studio XIAO ESP32-S3**
- **Tools → Flash Size → 8MB** (XIAO ESP32-S3 has 8MB flash)
- **Tools → USB Mode → Hardware CDC and JTAG** (or "CDC only" — not on boot)
- **Tools → USB CDC On Boot → OFF** (critical: prevents boot blocking)

### 3. Open the Sketch
- Open `/opt/GLaDOS/spokes/esp32-spoke/src/GLaDOS_Spoke.ino` in Arduino IDE
- Or copy the file to your Arduino sketches folder

### 4. Edit WiFi Credentials (if needed)
At the top of the file, edit these lines:
```cpp
#define SSID            "1.21Jigowatts"
#define PASSWORD        "Jigowatts"
#define HUB_HOST        "192.168.1.58"
#define HUB_PORT        5051
```

### 5. Upload
- Socket the XIAO ESP32-S3 onto the XVF3800 board
- Connect via USB-C
- Select the correct COM port (**Tools → Port**)
- Click **Upload** (→ button)

### 6. Monitor
- Open **Serial Monitor** (115200 baud)
- You should see boot logs including:
  - `[DIAG] Scanning I2C bus for XVF3800... Found 0x2C [XVF3800]`
  - `[8c] I2S RX OK (48kHz, 32-bit, stereo, SLAVE mode)`
  - `[6] WiFi connected!`
  - `[9] All tasks started!`

## Pin Map (XIAO → XVF3800)

These are hardwired by the XIAO socket on the XVF3800 board:

| Signal | XIAO GPIO | XVF3800 Role |
|--------|-----------|--------------|
| I2C SDA | GPIO5 | I2C slave SDA, addr 0x2C |
| I2C SCL | GPIO6 | I2C slave SCL |
| I2S WS | GPIO7 | I2S WS (XVF master) |
| I2S BCLK | GPIO8 | I2S BCLK (XVF master) |
| I2S RX (DIN) | GPIO43 | Mic audio in from XVF |
| I2S TX (DOUT) | GPIO44 | TTS audio out to XVF |
| XVF Reset | GPIO9 | Hard reset line |

## Troubleshooting Boot Issues

### If the board won't boot normally:
1. **Hold the BOOT button** on the XIAO ESP32-S3
2. **While holding BOOT**, plug in the USB-C cable
3. **Release BOOT** after USB is connected
4. The board will enter download mode — upload your sketch

### Why does this happen?
- The ESP32-S3 strapping pins (GPIO0, GPIO45, GPIO46) control boot mode
- If USB CDC is enabled on boot and the application crashes during init,
  the USB Serial/JTAG controller disappears and the board becomes invisible
- Holding BOOT (GPIO0) forces download mode regardless of application state

### XVF3800 Not Detected on I2C?
- Verify you flashed the I2S-master firmware (Step 0 above)
- Check the USB cable is data-capable
- The XVF3800 must be DFU-flashed BEFORE the ESP32-S3 is socketed

### No Audio Data Sent?
- Verify I2S pins are correctly connected
- Check serial output for I2S errors
- The XVF3800 must be running I2S-master firmware (not USB-audio)
