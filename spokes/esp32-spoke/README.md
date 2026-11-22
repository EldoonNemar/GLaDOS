# GLaDOS ESP32-S3 Spoke

ESP32-S3 firmware for the reSpeaker XVF3800 4-Mic Array — WebSocket frontend
for the GLaDOS voice assistant hub.

## ⚠️ CRITICAL: XVF3800 I2S-Master Firmware (Do This First!)

**The XVF3800 ships from the factory with USB-audio firmware ONLY.**
It will NOT work with the ESP32-S3 via I2S until you flash the I2S-master variant.

### Step 1: Flash XVF3800 to I2S-Master Mode

```bash
# Install dfu-util
sudo apt install dfu-util        # Ubuntu/Debian
brew install dfu-util            # macOS

# Run the flash script
cd /opt/GLaDOS/spokes/esp32-spoke
chmod +x flash_xvf3800_i2s_master.sh
sudo ./flash_xvf3800_i2s_master.sh
```

**Or manually:**
1. Download firmware from: https://github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY/blob/master/xmos_firmwares/i2s
2. Look for: `respeaker_xvf3800_i2s_master_v1.0.9_48k.bin`
3. Enter DFU mode: **Hold MUTE button → plug in USB-C → keep holding**
4. Flash: `dfu-util -d 0x20b1: -a 0 -D <firmware.bin> -R`

After this, the XVF3800 outputs I2S on the socket pins and accepts I2C control.

### Step 2: Flash ESP32-S3 Firmware

See [ARDUINO_SETUP.md](ARDUINO_SETUP.md) for Arduino IDE instructions.

---

## Hardware

| Part | Role |
|------|------|
| Seeed XIAO ESP32-S3 | Runs GLaDOS firmware |
| ReSpeaker XVF3800 4-Mic Array (I2S-master variant) | Mic array, beamforming, AEC |
| 4-8 Ω speaker on XVF3800 JST connector | Audio output (future) |

## Pin Map (XIAO → XVF3800)

Based on the [OpenEnsemble](https://github.com/openensemble/voice-device-firmware) reference:

| Signal | XIAO GPIO | XVF3800 Role |
|--------|-----------|--------------|
| I2C SDA | GPIO5 (D4) | I2C slave SDA, addr 0x2C |
| I2C SCL | GPIO6 (D5) | I2C slave SCL |
| I2S WS | GPIO7 (D8) | I2S WS (XVF master) |
| I2S BCLK | GPIO8 (D9) | I2S BCLK (XVF master) |
| I2S RX (DIN) | GPIO43 (D6) | Mic audio in from XVF |
| I2S TX (DOUT) | GPIO44 (D7) | TTS audio out to XVF |
| XVF Reset | GPIO9 | Hard reset line |

## Build & Flash

### PlatformIO
```bash
cd /opt/GLaDOS/spokes/esp32-spoke
platformio run --environment xiao_esp32s3
platformio run --environment xiao_esp32s3 --target upload --upload-port /dev/ttyACM0
platformio run --environment xiao_esp32s3 --target monitor
```

### Arduino IDE
1. Board: **Seeed Studio XIAO ESP32-S3**
2. Flash Size: **8MB**
3. USB CDC On Boot: **OFF**
4. Upload

### Recovery Flash (if bricked)
```bash
chmod +x flash_spoke_recovery.sh
sudo ./flash_spoke_recovery.sh /dev/ttyACM0
```

## Day-One Bring-Up Order

1. DFU-flash the XVF3800 I2S-master image (see above)
2. Socket the XIAO ESP32-S3 onto the XVF3800
3. `platformio run --target upload --upload-port /dev/ttyACM0`
4. Open serial monitor at 115200 baud
5. Verify `XVF3800 I2C bus up` in boot log
6. Check WiFi connects and WebSocket establishes
7. Speak into the mic — verify audio frames are sent to the hub

