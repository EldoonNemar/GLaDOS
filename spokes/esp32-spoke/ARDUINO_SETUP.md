# Flash GLaDOS ESP32-C6 Spoke via Arduino IDE

## On Your Laptop

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
- **Tools → Board → Seeed Studio XIAO ESP32C6**
- **Tools → Flash Size → 4MB**
- **Tools → USB Mode → Hardware CDC and JTAG**

### 3. Open the Sketch
- Open `/opt/GLaDOS/spokes/esp32-spoke/src/main.cpp` in Arduino IDE
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
- Connect your XVF3800 board via USB-C
- Select the correct COM port (**Tools → Port**)
- Click **Upload** (→ button)

### 6. Monitor
- Open **Serial Monitor** (115200 baud)
- You should see boot logs

## Hardware
- ESP32-C6 on Seeed Studio XVF3800 4-Mic Array
- I2S Audio: RX GPIO4/5/6, TX GPIO10/11/12
- Connects to GLaDOS hub at 192.168.1.58:5051
