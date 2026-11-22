/**
 * GLaDOS ESP32-S3 Spoke — reSpeaker XVF3800 WebSocket frontend.
 *
 * Based on the OpenEnsemble voice-device-firmware reference:
 * https://github.com/openensemble/voice-device-firmware
 *
 * Hardware:
 *   ESP32-S3 (Seeed Studio XIAO ESP32-S3) socketed on XVF3800
 *   reSpeaker XVF3800 (I2S-master mode — must be DFU-flashed first!)
 *
 * I2S Configuration (XVF is MASTER, ESP32 is SLAVE):
 *   XVF WS (D8)  -> ESP32 GPIO7  (LRCLK input to ESP32)
 *   XVF BCLK(D9) -> ESP32 GPIO8  (BCLK input to ESP32)
 *   XVF DOUT(D6) -> ESP32 GPIO43 (DIN - mic audio in)
 *   XVF DIN(D7)  -> ESP32 GPIO44 (DOUT - TTS audio out, future)
 *
 * I2C Configuration (for XVF3800 control):
 *   XVF SDA (D4) -> ESP32 GPIO5
 *   XVF SCL (D5) -> ESP32 GPIO6
 *   XVF address:  0x2C
 *
 * Note: The XVF3800 MUST be flashed to I2S-master mode via dfu-util
 * before the ESP32-S3 can communicate with it. See README.md for details.
 */

#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <Arduino.h>
#include <WiFi.h>
#include <Wire.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/event_groups.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "nvs_flash.h"
#include "driver/i2s_std.h"
#include "driver/gpio.h"
#include "lwip/sockets.h"
#include "lwip/dns.h"
#include "lwip/netdb.h"

#define SSID            "1.21Jigowatts"
#define PASSWORD        "Jigowatts"
#define HUB_HOST        "192.168.1.58"
#define HUB_PORT        5051
#define HUB_PATH        "/ws/esp32-s3-001"
#define DEVICE_ID       "esp32-s3-001"

// XVF3800 I2S configuration (matches OpenEnsemble reference)
// XVF3800 is I2S MASTER — ESP32-S3 is I2S SLAVE
// Bus runs at 48kHz stereo 32-bit (XVF I2S-master firmware native rate)
// App layer resamples to 16kHz for STT/wake-word
#define SAMPLE_RATE         48000     // I2S bus sample rate
#define APP_SAMPLE_RATE     16000     // Application sample rate
#define I2S_GPIO_BCLK       ((gpio_num_t)8)    // XVF BCLK -> ESP32 GPIO8
#define I2S_GPIO_LRCLK      ((gpio_num_t)7)    // XVF WS  -> ESP32 GPIO7
#define I2S_GPIO_DATA       ((gpio_num_t)43)   // XVF DOUT -> ESP32 GPIO43 (DIN)
#define I2S_GPIO_TX_DATA    ((gpio_num_t)44)   // ESP32 GPIO44 -> XVF DIN  (speaker out)

// I2C pins for XVF3800 control
#define I2C_GPIO_SDA        ((gpio_num_t)5)
#define I2C_GPIO_SCL        ((gpio_num_t)6)
#define I2C_XVF_ADDRESS     0x2C

#define AUDIO_QUEUE_LEN 64
#define CONV_CHUNK      1024   // B109: TTS conversion chunk (samples/frame)
#define FRAMETYPE_AUDIO_RX  0x01
#define FRAMETYPE_WAKE_TX   0x02   // text to hub (also carries boot diagnostics)
#define FRAMETYPE_STATUS_RX 0x13
#define FRAMETYPE_AUDIO_TX  0x11
#define FRAMETYPE_SESSION   0x03

// ── Crash diagnostics ─────────────────────────────────────────────
// This board has no reliable runtime serial console, so every boot
// records how far init got and what was still alive (RTC memory
// survives resets) and reports the PREVIOUS boot's state to the hub
// as a 0x02 text frame on each successful WebSocket connect. The hub
// logs it — diagnosable mid-crash-loop, no console required.
enum {
    ST_BOOT = 0, ST_I2C_SCAN, ST_NVS, ST_WIFI_INIT, ST_WIFI_WAIT,
    ST_QUEUES, ST_I2S_INIT, ST_TASKS,          // init progress (last_stage)
    ST_WS_LOOP, ST_AUDIO, ST_RX, ST_PLAYBACK   // live activity (last_activity)
};
static const char *const STAGE_NAMES[] = {
    "boot", "i2c_scan", "nvs", "wifi_init", "wifi_wait", "queues",
    "i2s_init", "tasks", "ws_loop", "audio", "rx", "playback"
};

RTC_DATA_ATTR static struct {
    uint32_t boot_count;
    uint8_t  last_stage;      // how far the previous boot's init got
    uint8_t  last_activity;   // last live activity before death
    uint32_t activity_count;  // heartbeats at last_activity
    uint8_t  xvf_i2c_found;   // I2C scan saw the XVF3800 at 0x2C
    uint8_t  i2s_clocks;      // I2S reads returned data (XVF clocking master)
    int32_t  i2s_ret;         // i2s_channel_init_std_mode result (0=OK)
    int32_t  i2s_enable_ret;  // i2s_channel_enable result (0=OK)
    int32_t  i2s_read_ret;    // last i2s_channel_read result (0=OK)
    uint32_t i2s_frames;      // successful reads (bytes>0)
    uint32_t i2s_read_fail;   // reads that returned 0 bytes
    int32_t  i2s_tx_ret;      // last i2s_channel_write result (0=OK)
    uint32_t i2s_tx_frames;   // TTS frames written to the speaker
    uint8_t  xvf_ver[3];      // [major,minor,patch] from the DFU controller
    uint8_t  xvf_ver_ok;      // 1 = version read returned status 0
    uint32_t ws_edges;        // GPIO7 (I2S WS) transitions in 1 s
    uint32_t bclk_edges;      // GPIO8 (I2S BCLK) transitions in 200 ms
    uint32_t max_pin_edges;   // best of the other socket pins, 200 ms
    uint32_t ws_resync;         // B108: WS stream desyncs/recoveries
    uint32_t i2s_tx_timeouts;   // B110: I2S TX writes that timed out (re-enabled + retried)
    uint32_t tts_drop;          // B110: TTS frames dropped (alloc failure)
} diag;

#define STAGE(s)      do { diag.last_stage = (s); } while (0)
#define HEARTBEAT(s)  do { diag.last_activity = (s); diag.activity_count++; } while (0)

// This SDK ships no esp_reset_reason_str() — map it ourselves.
static const char *reset_reason_name(esp_reset_reason_t r)
{
    switch (r) {
        case ESP_RST_POWERON:    return "POWERON";
        case ESP_RST_EXT:        return "EXT";
        case ESP_RST_SW:         return "SW";
        case ESP_RST_PANIC:      return "PANIC";
        case ESP_RST_INT_WDT:    return "INT_WDT";
        case ESP_RST_TASK_WDT:   return "TASK_WDT";
        case ESP_RST_WDT:        return "WDT";
        case ESP_RST_DEEPSLEEP:  return "DEEPSLEEP";
        case ESP_RST_BROWNOUT:   return "BROWNOUT";
        case ESP_RST_SDIO:       return "SDIO";
        case ESP_RST_USB:        return "USB";
        case ESP_RST_JTAG:       return "JTAG";
        case ESP_RST_EFUSE:      return "EFUSE";
        case ESP_RST_PWR_GLITCH: return "PWR_GLITCH";
        case ESP_RST_CPU_LOCKUP: return "CPU_LOCKUP";
        default:                 return "UNKNOWN";
    }
}
#define I2S_BUFFER_SIZE     (SAMPLE_RATE / 50)  // 20ms buffer
#define WIFI_CONNECT_TIMEOUT_SECS  30
#define WIFI_RECONNECT_TIMEOUT_MS  5000

// The hub's ASR consumes 16 kHz mono int16 PCM. The XVF3800 I2S bus runs at
// 48 kHz, so capture decimates 3:1 (box filter on the left channel) and
// accumulates APP_FRAME_SAMPLES (100 ms) before sending one frame.
// Wire format (matches python-spoke and the hub): [0x01][raw PCM...] —
// NO length header; the WebSocket frame carries the length.
#define APP_FRAME_SAMPLES   (APP_SAMPLE_RATE / 10)   // 1600 samples = 100 ms
#define MIC_GAIN            1    // integer gain before 32->16-bit (1..3 safe)

typedef struct {
    uint8_t *data;   // frame[0] = 0x01, followed by raw 16 kHz PCM
    size_t   len;
} audio_frame_t;

static EventGroupHandle_t wifi_event_group;
static const int WIFI_CONNECTED_BIT = BIT0;
static int ws_fd = -1;
static QueueHandle_t audio_tx_queue = NULL;
static QueueHandle_t audio_rx_queue = NULL;
static i2s_chan_handle_t i2s_rx_handle = NULL;
static i2s_chan_handle_t i2s_tx_handle = NULL;

// Playback volume (0-100, hub frame 0x16). Written by the WS task, read by
// the playback task; a plain int16 is atomic on Xtensa, volatile keeps the
// compiler from hoisting the load out of the per-sample loop.
// 100 = full (unchanged behavior), 0 = digital silence.
static volatile int16_t playback_volume = 100;

// Forward declarations
static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                                int32_t event_id, void *event_data);
static esp_err_t i2s_init(void);
static void i2s_capture_task(void *arg);
static void audio_playback_task(void *arg);
static int ws_connect(void);
static void ws_client_task(void *arg);
static void ws_send_frame(int fd, uint8_t frame_type, const uint8_t *data, size_t len);

// ── I2C diagnostic ────────────────────────────────────────────────
static void i2c_scan(void)
{
    Serial.println("[DIAG] Scanning I2C bus for XVF3800...");
    // Try the XIAO->XVF socket pins: GPIO5 (SDA), GPIO6 (SCL)
    // Also try GPIO21/20 (common ESP32-S3 defaults)
    uint8_t try_pins[][2] = {{5, 6}, {21, 20}, {8, 9}};
    
    for (int p = 0; p < 3; p++) {
        TwoWire i2c = TwoWire(0);
        i2c.begin(try_pins[p][0], try_pins[p][1], 400000);
        Serial.printf("  Pins SDA=%d SCL=%d: ", try_pins[p][0], try_pins[p][1]);
        int found = 0;
        for (uint8_t addr = 1; addr < 127; addr++) {
            i2c.beginTransmission(addr);
            if (i2c.endTransmission() == 0) {
                Serial.printf("Found 0x%02X ", addr);
                found++;
                if (addr == 0x2C || addr == 0x31 || addr == 0x32 || addr == 0x40) {
                    Serial.print("[XVF3800?]");
                    if (addr == 0x2C) diag.xvf_i2c_found = 1;
                }
            }
        }
        if (!found) Serial.print("No devices");
        Serial.println();
        i2c.end();
    }
    Serial.println("[DIAG] If no 0x2C above: XVF3800 has no power (board USB-C "
                   "plugged in?) or is not running I2S-master firmware.");
}

// Count digital transitions on a pin for ~ms. A toggling I2S clock shows
// up as a large count even when undersampled; a dead pin reads ~0.
static uint32_t count_edges(int pin, uint32_t ms)
{
    pinMode(pin, INPUT);
    uint32_t edges = 0;
    int last = digitalRead(pin);
    uint32_t t0 = millis();
    while (millis() - t0 < ms) {
        int v = digitalRead(pin);
        if (v != last) { edges++; last = v; }
    }
    return edges;
}

// XMOS XCORE-VOICE protocol (see OpenEnsemble xvf3800_ctrl):
//   read = write [resid, cmd|0x80, len+1], wait 5 ms, read [status, data...]
// resid 240 = DFU controller, cmd 88 = GETVERSION -> [major,minor,patch].
// Only the I2S-family firmware implements this resource; the factory
// USB-audio firmware will NACK or return a non-zero status.
static void xvf_version_probe(void)
{
    TwoWire wire(0);
    wire.begin(I2C_GPIO_SDA, I2C_GPIO_SCL, 400000);
    wire.beginTransmission(I2C_XVF_ADDRESS);
    wire.write(240);
    wire.write(88 | 0x80);
    wire.write(4);
    bool ack = (wire.endTransmission() == 0);
    if (ack) {
        delay(5);
        if (wire.requestFrom(I2C_XVF_ADDRESS, (int)4) == 4) {
            uint8_t st = wire.read();
            diag.xvf_ver[0] = wire.read();
            diag.xvf_ver[1] = wire.read();
            diag.xvf_ver[2] = wire.read();
            diag.xvf_ver_ok = (st == 0) ? 1 : 0;
        }
    }
    wire.end();
    Serial.printf("[DIAG] XVF version: %d.%d.%d (ok=%d ack=%d)\n",
                  diag.xvf_ver[0], diag.xvf_ver[1], diag.xvf_ver[2],
                  diag.xvf_ver_ok, (int)ack);
}

// ── XVF3800 LED ring (XMOS XCORE-VOICE protocol, resid 20 = GPO) ─────
// The I2S-master firmware dropped the factory LED animation, so the hub
// drives the ring via 0x15 frames (one byte). Each state is a few short
// I2C writes (≤7 bytes each at 400 kHz ≈ 200 µs) over the same bus the
// version probe used. Colors are {B, G, R, 0} — the firmware expects BGR.
#define XVF_LED_OFF     0
#define XVF_LED_LISTEN  1   // blue breath
#define XVF_LED_THINK   2   // amber breath
#define XVF_LED_SPEAK   3   // green steady
#define XVF_LED_ALARM   4   // rainbow
#define XVF_RESID_GPO        20
#define XVF_CMD_LED_EFFECT     12
#define XVF_CMD_LED_BRIGHTNESS 13
#define XVF_CMD_LED_SPEED      15
#define XVF_CMD_LED_COLOR      16

// The I2C bus is opened LAZILY — on the first LED frame, in the WS task,
// which only runs after WiFi is up. This is deliberate: an I2C transaction
// at boot (before WiFi) can block the whole boot if the XVF holds SDA low
// or NACKs, which would leave the device unable to ever connect. Post-WiFi
// it's harmless: the device is already connected and reporting.
static bool s_led_i2c_open = false;
static void xvf_led_i2c_open(void)
{
    if (!s_led_i2c_open) {
        Wire.begin(I2C_GPIO_SDA, I2C_GPIO_SCL, 400000);
        s_led_i2c_open = true;
    }
}

static void xvf_led_write(uint8_t cmd, const uint8_t *data, uint8_t len)
{
    Wire.beginTransmission(I2C_XVF_ADDRESS);
    Wire.write(XVF_RESID_GPO);
    Wire.write(cmd);
    Wire.write(len);
    if (len) Wire.write(data, len);
    Wire.endTransmission();
}

static void xvf_led_set_state(uint8_t state)
{
    xvf_led_i2c_open();  // lazy: first call is post-WiFi (WS task)
    uint8_t b, s, e, color[4];
    switch (state) {
    case XVF_LED_LISTEN:
        color[0] = 0xFF; color[1] = 0x40; color[2] = 0x20; color[3] = 0;  // blue
        b = 40; s = 50; e = 1;                                            // BREATH
        break;
    case XVF_LED_THINK:
        color[0] = 0x00; color[1] = 0xA0; color[2] = 0xFF; color[3] = 0;  // amber
        b = 60; s = 90; e = 1;                                            // BREATH
        break;
    case XVF_LED_SPEAK:
        color[0] = 0x20; color[1] = 0xFF; color[2] = 0x00; color[3] = 0;  // green
        b = 80; s = 90; e = 3;                                            // SINGLE
        break;
    case XVF_LED_ALARM:
        b = 100; s = 120; e = 2;                                          // RAINBOW
        xvf_led_write(XVF_CMD_LED_BRIGHTNESS, &b, 1);
        xvf_led_write(XVF_CMD_LED_SPEED, &s, 1);
        xvf_led_write(XVF_CMD_LED_EFFECT, &e, 1);
        return;
    default:  // OFF (and any unknown state)
        e = 0;
        xvf_led_write(XVF_CMD_LED_EFFECT, &e, 1);
        return;
    }
    xvf_led_write(XVF_CMD_LED_COLOR, color, 4);
    xvf_led_write(XVF_CMD_LED_BRIGHTNESS, &b, 1);
    xvf_led_write(XVF_CMD_LED_SPEED, &s, 1);
    xvf_led_write(XVF_CMD_LED_EFFECT, &e, 1);
}

// Which socket pins is the XVF actually driving? Expected: BCLK on GPIO8,
// WS on GPIO7. Every other socket pin is counted too, so a pin-map
// surprise (clocks on the wrong pads) is caught instead of guessed.
static void i2s_pin_probe(void)
{
    Serial.println("[DIAG] I2S pin probe...");
    int other_pins[] = {43, 44, 1, 3, 12, 14, 15, 16, 17, 18, 21};
    uint32_t best = 0;
    for (unsigned i = 0; i < sizeof(other_pins) / sizeof(other_pins[0]); i++) {
        uint32_t e = count_edges(other_pins[i], 200);
        if (e > best) best = e;
        if (e) Serial.printf("  GPIO%d: %lu edges\n", other_pins[i], (unsigned long)e);
    }
    diag.bclk_edges = count_edges(8, 200);
    diag.ws_edges = count_edges(7, 1000);
    diag.max_pin_edges = best;
    Serial.printf("[DIAG] probe: bclk(GPIO8)=%lu ws(GPIO7)=%lu max_other=%lu\n",
                  (unsigned long)diag.bclk_edges, (unsigned long)diag.ws_edges,
                  (unsigned long)best);
}

void glados_main(void)
{
    Serial.println("[1] Starting GLaDOS Spoke...");

    // NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        Serial.println("[2] Erasing NVS...");
        nvs_flash_erase();
        ret = nvs_flash_init();
    }
    if (ret != ESP_OK) {
        Serial.print("[2] NVS fail: "); Serial.println(ret);
        return;
    }
    Serial.println("[3] NVS OK");
    STAGE(ST_NVS);

    // Network
    Serial.println("[4] Init network...");
    esp_netif_init();
    esp_event_loop_create_default();
    esp_netif_create_default_wifi_sta();
    wifi_event_group = xEventGroupCreate();
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&cfg);
    esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                         &wifi_event_handler, NULL, NULL);
    esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP,
                                         &wifi_event_handler, NULL, NULL);

    // WiFi config
    wifi_config_t wifi_config = {};
    strncpy((char *)wifi_config.sta.ssid, SSID, sizeof(wifi_config.sta.ssid));
    strncpy((char *)wifi_config.sta.password, PASSWORD, sizeof(wifi_config.sta.password));
    wifi_config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;

    esp_wifi_set_mode(WIFI_MODE_STA);
    esp_wifi_set_config(WIFI_IF_STA, &wifi_config);
    esp_wifi_start();
    STAGE(ST_WIFI_INIT);
    Serial.print("[5] Connecting to WiFi: "); Serial.println(SSID);
    STAGE(ST_WIFI_WAIT);
    xEventGroupWaitBits(wifi_event_group, WIFI_CONNECTED_BIT, false, true,
                         pdMS_TO_TICKS(WIFI_CONNECT_TIMEOUT_SECS * 1000));

    if (!(xEventGroupGetBits(wifi_event_group) & WIFI_CONNECTED_BIT)) {
        Serial.println("[5] WiFi FAILED! Rebooting...");
        delay(2000);
        esp_restart();
    }
    Serial.println("[6] WiFi connected!");

    // Queues
    audio_tx_queue = xQueueCreate(AUDIO_QUEUE_LEN, sizeof(audio_frame_t));
    audio_rx_queue = xQueueCreate(AUDIO_QUEUE_LEN, sizeof(audio_frame_t));
    Serial.println("[7] Queues created");
    STAGE(ST_QUEUES);

    // I2S - with error handling
    Serial.println("[8] Initializing I2S...");
    esp_err_t i2s_ret = i2s_init();
    if (i2s_ret != ESP_OK) {
        Serial.printf("[8] I2S init FAILED (0x%x), continuing without audio\n", i2s_ret);
    } else {
        Serial.println("[8] I2S RX OK");
    }
    STAGE(ST_I2S_INIT);

    // Tasks - with larger stacks to prevent overflow
    Serial.println("[9] Creating tasks...");
    if (i2s_ret == ESP_OK) {
        // B110: audio tasks own core 1 (reference driver runs them at 10/9).
        // At priority 5 they lost core time to the mic path and the DMA
        // underran under load. Playback above capture: feeding the TX DMA
        // is the hard real-time deadline.
        xTaskCreatePinnedToCore(i2s_capture_task, "i2s_capture", 16384, NULL, 8, NULL, 1);
        Serial.println("[9a] I2S capture task started");
    } else {
        Serial.println("[9a] I2S capture task SKIPPED (I2S init failed)");
    }
    xTaskCreatePinnedToCore(audio_playback_task, "audio_playback", 16384, NULL, 9, NULL, 1);
    Serial.println("[9b] Audio playback task started");
    xTaskCreatePinnedToCore(ws_client_task, "ws_client", 24576, NULL, 6, NULL, 0);
    Serial.println("[9c] WebSocket client task started");
    Serial.println("[9] All tasks started!");
    STAGE(ST_TASKS);
}

static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                                int32_t event_id, void *event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        Serial.println("WiFi disconnected, reconnecting...");
        esp_wifi_connect();
        xEventGroupClearBits(wifi_event_group, WIFI_CONNECTED_BIT);
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        Serial.print("Got IP: "); Serial.println(ip4addr_ntoa((const ip4_addr_t *)&event->ip_info.ip));
        xEventGroupSetBits(wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

static esp_err_t i2s_init(void)
{
    Serial.println("[8a] Creating I2S TX+RX channels (XVF master, ESP32 slave)...");
    // XVF3800 is I2S MASTER — ESP32-S3 is SLAVE (follows XVF clocks)
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_SLAVE);
    // Slave-mode requirement (verified against the openensemble reference
    // driver for this exact board): DMA descriptors must auto-clear, or
    // blocking reads return 0 bytes forever despite live BCLK/WS.
    chan_cfg.auto_clear = true;
    // Reference driver's DMA sizing: 4 descriptors x 480 frames x 16 B =
    // 30 KB ~ 64 ms of absorption. The driver default (4 x 240 = 15 KB,
    // 40 ms) underruns under the WiFi+TTS-burst cache pressure, and a
    // slave-mode TX underrun stalls the engine (B110 also fixes the stall
    // recovery; bigger DMA is the first line of defense).
    chan_cfg.dma_desc_num = 4;
    chan_cfg.dma_frame_num = 480;
    // Full duplex, like the reference driver: one i2s_new_channel call
    // creates both directions on the same port, so mic RX keeps flowing
    // while TTS TX plays (this is what makes barge-in possible).
    esp_err_t ret = i2s_new_channel(&chan_cfg, &i2s_tx_handle, &i2s_rx_handle);
    if (ret != ESP_OK) {
        Serial.print("[8a] I2S channel fail: "); Serial.println(esp_err_to_name(ret));
        return ret;
    }
    Serial.println("[8b] Configuring I2S...");
    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        // XVF3800 I2S-master firmware outputs: 48kHz, 32-bit, stereo (Philips mode)
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,   // XVF generates MCLK internally
            .bclk = I2S_GPIO_BCLK,     // GPIO8  (XVF BCLK -> ESP32)
            .ws = I2S_GPIO_LRCLK,      // GPIO7  (XVF WS   -> ESP32)
            .dout = I2S_GPIO_TX_DATA,  // GPIO44 (ESP32 -> XVF speaker path)
            .din = I2S_GPIO_DATA,      // GPIO43 (XVF DOUT -> ESP32)
        },
    };
    ret = i2s_channel_init_std_mode(i2s_rx_handle, &std_cfg);
    if (ret == ESP_OK) ret = i2s_channel_init_std_mode(i2s_tx_handle, &std_cfg);
    diag.i2s_ret = (int32_t)ret;
    if (ret != ESP_OK) {
        Serial.print("[8b] I2S init fail: "); Serial.println(esp_err_to_name(ret));
        return ret;
    }
    // Required before the channel moves any data — the reference driver for
    // this board (openensemble) calls enable on both channels after init.
    esp_err_t en = i2s_channel_enable(i2s_rx_handle);
    if (en == ESP_OK) en = i2s_channel_enable(i2s_tx_handle);
    diag.i2s_enable_ret = (int32_t)en;
    if (en != ESP_OK) {
        Serial.printf("[8c] I2S enable fail: %s\n", esp_err_to_name(en));
        return en;
    }
    Serial.println("[8c] I2S RX+TX OK (48kHz, 32-bit, stereo, SLAVE, enabled)");
    return ESP_OK;
}

static void i2s_capture_task(void *arg)
{
    size_t bytes_read;
    // Buffer for 32-bit stereo samples from XVF3800 I2S bus (48kHz)
    int32_t *raw_buf = (int32_t *)heap_caps_malloc(I2S_BUFFER_SIZE * 4, MALLOC_CAP_DMA);
    if (!raw_buf) { Serial.println("[I2S] DMA buf alloc fail"); vTaskDelete(NULL); return; }

    // Accumulator: 48 kHz -> 16 kHz (3:1 box filter on the left channel),
    // then emit 100 ms frames of 16-bit mono for the hub's ASR.
    static int16_t acc[APP_FRAME_SAMPLES];
    static size_t acc_count = 0;

    uint32_t frame_count = 0;
    uint32_t no_data = 0;
    int32_t last_peak = 0;

    while (1) {
        // 1 s timeout, NOT portMAX_DELAY: if the XVF3800 isn't running the
        // I2S-master firmware (or the board is unpowered) there are no BCLK/
        // WS clocks, a blocking read hangs the task, and the task WDT reboots
        // the whole device in a 13 s loop.
        esp_err_t rret = i2s_channel_read(i2s_rx_handle, raw_buf, I2S_BUFFER_SIZE * 4,
                                         &bytes_read, pdMS_TO_TICKS(1000));
        diag.i2s_read_ret = (int32_t)rret;
        HEARTBEAT(ST_AUDIO);  // capture task alive (1 Hz w/o clocks, 50 Hz w/ data)
        if (bytes_read <= 0) {
            diag.i2s_read_fail++;
            if (++no_data == 30) {
                Serial.println("[I2S] WARNING: no I2S clocks from XVF3800 for 30 s — "
                               "is the board powered and flashed to I2S-master mode?");
                no_data = 0;
            }
            continue;
        }
        no_data = 0;
        diag.i2s_clocks = 1;
        diag.i2s_frames++;
        frame_count++;

        size_t stereo_frames = bytes_read / 8;   // 8 bytes per L/R pair
        size_t out = stereo_frames / 3;          // 48k -> 16k decimation
        for (size_t i = 0; i < out; i++) {
            size_t base = i * 3;                 // 3 consecutive stereo frames
            int32_t s0 = raw_buf[base * 2];      // left channel (even indices)
            int32_t s1 = raw_buf[base * 2 + 2];
            int32_t s2 = raw_buf[base * 2 + 4];
            int32_t avg = (s0 + s1 + s2) / 3;
            int16_t s = (int16_t)(((int32_t)avg * MIC_GAIN) >> 16);
            int32_t a = s < 0 ? -(int32_t)s : (int32_t)s;
            if (a > last_peak) last_peak = a;
            acc[acc_count++] = s;
            if (acc_count >= APP_FRAME_SAMPLES) {
                size_t pcm_len = APP_FRAME_SAMPLES * 2;
                uint8_t *frame = (uint8_t *)malloc(1 + pcm_len);
                if (frame) {
                    frame[0] = FRAMETYPE_AUDIO_RX;
                    memcpy(&frame[1], acc, pcm_len);
                    audio_frame_t af = { frame, 1 + pcm_len };
                    xQueueSend(audio_tx_queue, &af, portMAX_DELAY);
                }
                acc_count = 0;
            }
        }
        if (frame_count % 100 == 0) {
            // ~2 s of audio: report level so gain can be tuned (the looped
            // reference recording should show a stable, non-zero peak)
            Serial.printf("[I2S] %lu frames, peak %ld\n", frame_count, last_peak);
            last_peak = 0;
        }
    }
    heap_caps_free(raw_buf);
    vTaskDelete(NULL);
}

static void audio_playback_task(void *arg)
{
    // Hub streams TTS as 24 kHz, 16-bit, mono PCM (0x11 frames). The XVF I2S
    // bus is 48 kHz, 32-bit, stereo. Per input sample:
    //   s (int16) -> two int32 stereo pairs (s<<16) on each channel,
    // which is 2x linear upsample + 16->32 expand + mono duplicate.
    // The XVF firmware does AEC, so mic RX keeps working while we talk
    // (barge-in still works; the hub also cuts us with 0x12).
    //
    // B110: fixed 16 KB conversion buffer, STATIC INTERNAL RAM (BSS).
    // It was a heap malloc: under the TTS burst the heap was down to 7 KB,
    // so the alloc raced the crash, and a PSRAM return (malloc's default
    // when internal RAM is exhausted) put the I2S write source in slow
    // memory. 1024-sample chunks: starts writing ~20 ms into each frame.
    static int32_t conv_buf[CONV_CHUNK * 4];

    HEARTBEAT(ST_PLAYBACK);  // playback task alive at boot

    while (1) {
        audio_frame_t af;
        if (xQueueReceive(audio_rx_queue, &af, portMAX_DELAY) != pdTRUE) continue;
        if (!af.data) continue;
        HEARTBEAT(ST_PLAYBACK);  // forensics: a death while speaking says "playback"
        uint8_t *pcm = af.data;
        size_t n_in = af.len / 2;  // int16 samples (little-endian)

        size_t i = 0;
        while (i < n_in) {
            size_t n = (n_in - i) < CONV_CHUNK ? (n_in - i) : CONV_CHUNK;
            for (size_t j = 0; j < n; j++) {
                int32_t s = (int32_t)(int16_t)((uint16_t)pcm[(i + j) * 2] | ((uint16_t)pcm[(i + j) * 2 + 1] << 8));
                int16_t vol = playback_volume;  // hub 0x16 gain (per chunk, not per sample)
                if (vol < 100) s = s * vol / 100;  // int16*100 fits int32; no clamp needed
                s <<= 16;  // 16-bit -> 32-bit, upper-aligned (Philips)
                conv_buf[j * 4 + 0] = s;  // L @ t
                conv_buf[j * 4 + 1] = s;  // R @ t
                conv_buf[j * 4 + 2] = s;  // L @ t+1 (2x upsample)
                conv_buf[j * 4 + 3] = s;  // R @ t+1
            }
            size_t to_write = n * 16;   // n * 4 int32 * 4 bytes
            size_t written = 0;
            uint32_t stuck = 0;
            // B110: never drop audio on a stalled write. A 0-byte timeout
            // means the slave TX engine locked up after a DMA underrun —
            // re-enable the TX channel (RX keeps running) and retry the
            // SAME data. Only a bus that stays dead for ~2 s loses a chunk.
            while (written < to_write) {
                size_t w = 0;
                esp_err_t wret = i2s_channel_write(i2s_tx_handle, conv_buf + (written / 4),
                                                   to_write - written, &w, pdMS_TO_TICKS(100));
                diag.i2s_tx_ret = (int32_t)wret;
                if (w > 0) { written += w; stuck = 0; continue; }
                diag.i2s_tx_timeouts++;
                i2s_channel_disable(i2s_tx_handle);
                i2s_channel_enable(i2s_tx_handle);
                vTaskDelay(pdMS_TO_TICKS(5));
                if (++stuck >= 200) {
                    Serial.printf("[TX] chunk dropped: %lu timeouts in a row\n",
                                  (unsigned long)stuck);
                    written = to_write;  // advance past the lost chunk
                    break;
                }
            }
            if (written > 0) diag.i2s_tx_frames++;
            i += n;
        }
        free(pcm);
    }
    vTaskDelete(NULL);
}

static int resolve_host(const char *host, struct sockaddr_in *addr)
{
    struct hostent *he = gethostbyname(host);
    if (!he) { Serial.print("DNS fail: "); Serial.println(host); return -1; }
    addr->sin_family = AF_INET;
    addr->sin_port = htons(HUB_PORT);
    addr->sin_addr = *(struct in_addr *)he->h_addr;
    memset(&addr->sin_zero, 0, sizeof(addr->sin_zero));
    return 0;
}

static int ws_connect(void)
{
    struct sockaddr_in addr;
    if (resolve_host(HUB_HOST, &addr) != 0) return -1;
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { Serial.println("Socket fail"); return -1; }
    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        Serial.println("Connect fail"); close(fd); return -1;
    }
    char req[512];
    snprintf(req, sizeof(req),
        "GET %s HTTP/1.1\r\nHost: %s:%d\r\n"\
        "Upgrade: websocket\r\nConnection: Upgrade\r\n"\
        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"\
        "Sec-WebSocket-Version: 13\r\n\r\n",
        HUB_PATH, HUB_HOST, HUB_PORT);
    send(fd, req, strlen(req), 0);
    char resp[256];
    if (recv(fd, resp, sizeof(resp), 0) <= 0) { close(fd); return -1; }
    if (strstr(resp, "101") == NULL) { close(fd); return -1; }
    Serial.println("WebSocket connected!");
    return fd;
}

static void ws_send_frame(int fd, uint8_t frame_type, const uint8_t *data, size_t len)
{
    uint8_t frame[4096 + 14];  // 14 = 2 header + 4 mask + 8 extended length
    size_t offset = 0;
    uint8_t mask_key[4];
    // Generate random mask key
    for (int i = 0; i < 4; i++) mask_key[i] = (uint8_t)(esp_random() & 0xFF);

    // First frame header: FIN=1, opcode=BINARY(2), masked=1
    frame[offset++] = 0x82;
    // Second byte: masked (0x80) + length
    if (len < 125) {
        frame[offset++] = 0x80 | len;
    } else if (len < 65535) {
        frame[offset++] = 0x80 | 126;
        frame[offset++] = (len >> 8) & 0xFF;
        frame[offset++] = len & 0xFF;
    }
    // Mask key
    memcpy(&frame[offset], mask_key, 4);
    offset += 4;

    // Masked payload
    for (size_t i = 0; i < len; i++) {
        frame[offset++] = data[i] ^ mask_key[i % 4];
    }

    send(fd, frame, offset, 0);
}

// WebSocket frame handler — one COMPLETE frame at a time. Returns an action
// for the caller. Split out so the reassembly loop in ws_client_task can
// hold partial frames across recv()s and process several per recv().
enum { WS_ACT_OK = 0, WS_ACT_RECONNECT };

static int ws_handle_frame(uint8_t opcode, const uint8_t *payload, size_t payload_len)
{
    if (payload_len < 1) return WS_ACT_OK;  // no type byte — nothing to do

    if (opcode == 0x08) {  // Close frame
        Serial.println("WS closed by server");
        return WS_ACT_RECONNECT;
    }

    if (opcode == 0x09) {  // PING — respond with a masked PONG echoing payload
        if (ws_fd >= 0 && payload_len < 126) {
            uint8_t pong[14 + 125];  // header(2) + mask(4) + payload(125) max
            pong[0] = 0x8A;  // FIN + PONG
            pong[1] = 0x80 | (uint8_t)payload_len;
            uint8_t mask_key[4] = {0xDE, 0xAD, 0xBE, 0xEF};
            memcpy(&pong[2], mask_key, 4);
            for (size_t i = 0; i < payload_len; i++) {
                pong[6 + i] = payload[i] ^ mask_key[i % 4];
            }
            send(ws_fd, pong, 6 + payload_len, 0);
        }
        return WS_ACT_OK;
    }

    uint8_t frame_type = payload[0];
    uint8_t *body = (uint8_t *)&payload[1];
    size_t body_len = payload_len - 1;

    if (frame_type == 0x11) {  // Audio response (TTS: 24 kHz mono int16)
        // B110: THE TTS KILLER. This malloc'd from the 75 KB internal heap:
        // a TTS burst queues ~8 frames (9.6 KB each), the heap runs dry,
        // and every later frame died here — silently. That is the "5 words,
        // garble, silence." Copy into PSRAM instead (8 MB, no pressure);
        // the blocking xQueueSend then paces the hub via TCP backpressure,
        // exactly like the reference driver's ring buffer.
        // PSRAM first (8 MB on this board — a full reply bursts ~576 KB).
        // Fallback to the internal heap so a PSRAM-less build degrades to
        // "first few seconds of audio" instead of total silence.
        void *copy = heap_caps_malloc(body_len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!copy) copy = malloc(body_len);
        if (copy) {
            memcpy(copy, body, body_len);
            audio_frame_t af = { (uint8_t *)copy, body_len };
            xQueueSend(audio_rx_queue, &af, portMAX_DELAY);
        } else {
            diag.tts_drop++;  // no more silent drops
            Serial.println("[TTS] frame dropped (alloc fail)");
        }
    } else if (frame_type == 0x12) {  // Stop playback (barge-in / GLaDOS interrupted)
        Serial.println("STOP playback");
        audio_frame_t dropped;
        while (xQueueReceive(audio_rx_queue, &dropped, 0) == pdTRUE) {
            free(dropped.data);
        }
    } else if (frame_type == 0x13) {  // Status (not NUL-terminated — use length)
        Serial.print("STATUS: "); Serial.write(body, body_len); Serial.println();
    } else if (frame_type == 0x15) {  // XVF3800 LED ring state (1 byte)
        if (body_len >= 1) {
            xvf_led_set_state(body[0]);
            Serial.printf("LED -> %d\n", body[0]);
        }
    } else if (frame_type == 0x16) {  // Playback volume (int16 LE, 0-100)
        if (body_len >= 2) {
            int16_t v = (int16_t)((uint16_t)body[0] | ((uint16_t)body[1] << 8));
            if (v < 0) v = 0;
            if (v > 100) v = 100;
            playback_volume = v;
            Serial.printf("Volume -> %d%%\n", (int)v);
        }
    }
    return WS_ACT_OK;
}

static void ws_client_task(void *arg)
{
    // Reassembly buffer for inbound frames. Hub TTS frames are ~9.2 KB
    // (190 ms @ 24 kHz mono int16 + type byte) — bigger than any single
    // recv() — and several small frames can arrive in one. Static (BSS),
    // not stack. 32 KB covers frames up to ~660 ms of audio.
    static uint8_t ws_stream_buf[32768 + 16];
    static size_t ws_stream_have = 0;
    int reconnect_delay = 1000;
    while (1) {
        HEARTBEAT(ST_WS_LOOP);
        if (ws_fd < 0) {
            ws_fd = ws_connect();
            if (ws_fd >= 0) {
                reconnect_delay = 1000;
                ws_stream_have = 0;  // fresh connection: no leftover stream bytes
                uint8_t start_frame[] = {FRAMETYPE_SESSION};
                ws_send_frame(ws_fd, 0, start_frame, sizeof(start_frame));
                // Report the PREVIOUS boot's forensics as a 0x02 text frame —
                // the wake-word type the hub already decodes and logs. This
                // board has no reliable runtime console, so the hub log is the
                // only channel to see how far the last boot got and what it
                // was doing when it died.
                {
                    char diag_buf[192];
                    if (diag.boot_count > 1) {
                        uint8_t st = diag.last_stage > 7 ? 7 : (uint8_t)diag.last_stage;
                        uint8_t ac = diag.last_activity > 11 ? 11 : (uint8_t)diag.last_activity;
                        snprintf(diag_buf, sizeof(diag_buf),
                                 "boot #%lu, prev: init->%s, last=%s x%lu, xvf_i2c=%d, i2s_clocks=%d",
                                 (unsigned long)diag.boot_count, STAGE_NAMES[st], STAGE_NAMES[ac],
                                 (unsigned long)diag.activity_count, (int)diag.xvf_i2c_found, (int)diag.i2s_clocks);
                    } else {
                        snprintf(diag_buf, sizeof(diag_buf), "boot #1 (first boot)");
                    }
                    // How the previous boot died + which XVF firmware is
                    // running (v=X.Y.Z + 1/0 = version read ok) + I2S pin
                    // probe (w=WS edges 1 s, c=BCLK edges 200 ms, o=best
                    // of the other socket pins):
                    snprintf(diag_buf + strlen(diag_buf),
                             sizeof(diag_buf) - strlen(diag_buf),
                             " rst=%s v=%d.%d.%d%d w=%lu c=%lu o=%lu",
                             reset_reason_name(esp_reset_reason()),
                             diag.xvf_ver[0], diag.xvf_ver[1], diag.xvf_ver[2],
                             (int)diag.xvf_ver_ok, (unsigned long)diag.ws_edges,
                             (unsigned long)diag.bclk_edges, (unsigned long)diag.max_pin_edges);
                    uint8_t diag_frame[200];
                    diag_frame[0] = FRAMETYPE_WAKE_TX;
                    memcpy(&diag_frame[1], diag_buf, strlen(diag_buf));
                    ws_send_frame(ws_fd, 0, diag_frame, strlen(diag_buf) + 1);
                }
            } else {
                vTaskDelay(pdMS_TO_TICKS(reconnect_delay));
                reconnect_delay = reconnect_delay < 30000 ? reconnect_delay * 2 : reconnect_delay;
                continue;
            }
        }

        fd_set readfds;
        struct timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = 100000;  // 100ms timeout
        FD_ZERO(&readfds);
        FD_SET(ws_fd, &readfds);

        int ret = select(ws_fd + 1, &readfds, NULL, NULL, &tv);

        // Send audio from queue (non-blocking)
        audio_frame_t af;
        int sent = 0;
        while (xQueuePeek(audio_tx_queue, &af, 0) == pdTRUE) {
            if (xQueueReceive(audio_tx_queue, &af, 0) == pdTRUE) {
                // af.data[0] is 0x01; the WebSocket frame carries the length
                ws_send_frame(ws_fd, 0, af.data, af.len);
                free(af.data);
                sent++;
            } else {
                break;
            }
        }
        if (sent) Serial.printf("[SEND] %d frames\n", sent);

        // Periodic diagnostic heartbeat to the hub (~10 s): uptime, whether
        // the XVF3800 answered on I2C, whether I2S clocks are flowing, and
        // free heap. The hub logs it — the last line before a crash names
        // the death state without needing the runtime console.
        {
            static uint32_t diag_iter = 0;
            if (++diag_iter >= 100) {
                diag_iter = 0;
                char dbg[240];
                snprintf(dbg, sizeof(dbg),
                         "DIAG B114 t=%lus xvf=%d iclk=%d heap=%d pmb=%d v=%d.%d.%d%d "
                         "i2s r=%d e=%d rd=%d fr=%lu rf=%lu tx=%d tf=%lu txto=%lu tdr=%lu",
                         (unsigned long)(millis() / 1000), (int)diag.xvf_i2c_found,
                         (int)diag.i2s_clocks, (int)ESP.getFreeHeap(),
                         (int)(ESP.getPsramSize() / 1048576),  // PSRAM MB (8 = enabled+alive)
                         diag.xvf_ver[0], diag.xvf_ver[1], diag.xvf_ver[2],
                         (int)diag.xvf_ver_ok, (int)diag.i2s_ret,
                         (int)diag.i2s_enable_ret, (int)diag.i2s_read_ret,
                         (unsigned long)diag.i2s_frames,
                         (unsigned long)diag.i2s_read_fail,
                         (int)diag.i2s_tx_ret, (unsigned long)diag.i2s_tx_frames,
                         (unsigned long)diag.i2s_tx_timeouts, (unsigned long)diag.tts_drop);
                uint8_t df[244];
                df[0] = FRAMETYPE_WAKE_TX;
                memcpy(&df[1], dbg, strlen(dbg));
                ws_send_frame(ws_fd, 0, df, strlen(dbg) + 1);
            }
        }

        if (ret < 0) continue;  // Error

        // Receive from the hub. TTS frames (~9.2 KB) are bigger than a single
        // recv() and several small frames can arrive in one, so accumulate
        // bytes and parse only COMPLETE frames (see ws_handle_frame). The old
        // parser assumed one frame per recv() into a 4 KB buffer: a partial
        // TTS frame was mis-parsed, its leftover bytes were re-read as a new
        // frame, and a garbage 0x08 opcode made the device close its own
        // socket mid-TTS — the "crash" that was never a crash.
        if (ret > 0 && ws_stream_have < sizeof(ws_stream_buf)) {
            int bytes = recv(ws_fd, ws_stream_buf + ws_stream_have,
                             sizeof(ws_stream_buf) - ws_stream_have, 0);
            if (bytes <= 0) {
                Serial.println("WS lost, reconnecting...");
                close(ws_fd); ws_fd = -1;
                ws_stream_have = 0;
                vTaskDelay(pdMS_TO_TICKS(reconnect_delay));
                reconnect_delay = reconnect_delay < 30000 ? reconnect_delay * 2 : reconnect_delay;
                continue;
            }
            ws_stream_have += (size_t)bytes;
            HEARTBEAT(ST_RX);
        }

        // Process every complete frame currently buffered
        while (ws_stream_have >= 2) {
            uint8_t opcode = ws_stream_buf[0] & 0x0F;
            uint8_t mask_flag = ws_stream_buf[1] & 0x80;
            size_t payload_len = ws_stream_buf[1] & 0x7F;
            size_t header = 2;

            if (payload_len == 126) {  // 16-bit extended length
                if (ws_stream_have < 4) break;
                payload_len = ((size_t)ws_stream_buf[2] << 8) | ws_stream_buf[3];
                header = 4;
            } else if (payload_len == 127) {  // 64-bit extended length
                if (ws_stream_have < 10) break;
                if (ws_stream_buf[2] != 0) {  // >4 GB is nonsense — resync
                    Serial.println("WS frame absurd, resync");
                    diag.ws_resync++;
                    ws_stream_have = 0;
                    break;
                }
                payload_len = 0;
                for (int i = 3; i < 10; i++) payload_len = (payload_len << 8) | ws_stream_buf[i];
                header = 10;
            }

            size_t total = header + (mask_flag ? 4 : 0) + payload_len;
            if (total > sizeof(ws_stream_buf)) {  // bigger than buffer — drop stream
                Serial.println("WS frame too big, resync");
                diag.ws_resync++;
                ws_stream_have = 0;
                break;
            }
            if (ws_stream_have < total) break;  // frame incomplete — wait for more

            // Complete frame: unmask in place, then dispatch
            uint8_t *payload = ws_stream_buf + header;
            if (mask_flag) {
                uint32_t mask_key;
                memcpy(&mask_key, payload, 4);
                payload += 4;
                for (size_t i = 0; i < payload_len; i++) {
                    payload[i] ^= (mask_key >> ((i % 4) * 8)) & 0xFF;
                }
            }
            int act = ws_handle_frame(opcode, payload, payload_len);

            // Consume the frame
            if (ws_stream_have > total) {
                memmove(ws_stream_buf, ws_stream_buf + total, ws_stream_have - total);
                ws_stream_have -= total;
            } else {
                ws_stream_have = 0;
            }

            if (act == WS_ACT_RECONNECT) {
                close(ws_fd); ws_fd = -1;
                ws_stream_have = 0;
                vTaskDelay(pdMS_TO_TICKS(reconnect_delay));
                reconnect_delay = reconnect_delay < 30000 ? reconnect_delay * 2 : reconnect_delay;
                break;
            }
        }
    }
}

// ── Arduino entry point ──────────────────────────────────────────
void setup()
{
    Serial.begin(115200);
    Serial.println("=============================");
    Serial.println("GLaDOS ESP32-S3 Spoke v1.4");
    Serial.println("=============================");
    Serial.printf("Free heap: %d bytes\n", ESP.getFreeHeap());
    // How did the PREVIOUS boot die? The ROM records this in RTC registers
    // at reset time — it survives even when the RTC data bss is wiped:
    //   BROWNOUT  -> power sag (weak cable/port/rail)
    //   PANIC     -> software bug (backtrace is printed just above this)
    //   WATCHDOG  -> a task hung
    //   POWERON   -> full power loss
    Serial.printf("Reset reason: %s\n", reset_reason_name(esp_reset_reason()));
    STAGE(ST_BOOT);
    diag.boot_count++;

    // Quick I2C scan to find the XVF3800
    delay(200);
    i2c_scan();
    // NOTE: the LED ring's I2C bus is opened LAZILY on the first 0x15 frame
    // (in the WS task, post-WiFi) — see xvf_led_i2c_open(). An I2C
    // transaction here, before WiFi, can block boot and leave the device
    // unable to ever connect.
    STAGE(ST_I2C_SCAN);
    // Which firmware is actually running, and is it driving I2S clocks?
    xvf_version_probe();
    i2s_pin_probe();

    // Main initialization. glados_main() blocks until wifi is up, creates
    // the audio/websocket tasks, and RETURNS — that is the normal path, not
    // a fault. loop() then parks the core task. (An earlier build treated
    // the return as fatal and esp_restart()'d in a 16 s loop.)
    glados_main();
}

void loop()
{
    // All real work happens in the dedicated FreeRTOS tasks created by
    // glados_main(). This task (Arduino's loopTask) stays alive — the core
    // registers it with the task watchdog, and deleting it from loop()
    // produced delayed, load-timed resets.
    vTaskDelay(pdMS_TO_TICKS(10000));
}
