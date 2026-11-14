/**
 * GLaDOS ESP32-C6 Spoke — reSpeaker XVF3800 WebSocket frontend.
 * Hardware:
 *   ESP32-C6 (Seeed Studio XIAO ESP32C6 on reSpeaker XVF3800)
 *   reSpeaker XVF3800 (I2S mode)
 *     - SCK -> GPIO4 (I2S BCLK)
 *     - WS  -> GPIO5 (I2S LRCLK)
 *     - SD  -> GPIO6 (I2S DIN)
 *   Audio output: GPIO10 (BCLK), GPIO11 (WS), GPIO12 (DOUT)
 */

#include <string.h>
#include <stdlib.h>
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
#define HUB_PATH        "/ws/esp32-c6-001"
#define DEVICE_ID       "esp32-c6-001"
#define SAMPLE_RATE     16000
#define I2S_GPIO_BCLK   ((gpio_num_t)4)
#define I2S_GPIO_LRCLK  ((gpio_num_t)5)
#define I2S_GPIO_DATA   ((gpio_num_t)6)
#define AUDIO_QUEUE_LEN 64
#define FRAMETYPE_AUDIO_RX  0x01
#define FRAMETYPE_STATUS_RX 0x13
#define FRAMETYPE_AUDIO_TX  0x11
#define FRAMETYPE_SESSION   0x03
#define I2S_BUFFER_SIZE     (SAMPLE_RATE / 10)
#define WIFI_CONNECT_TIMEOUT_SECS  30
#define WIFI_RECONNECT_TIMEOUT_MS  5000

static EventGroupHandle_t wifi_event_group;
static const int WIFI_CONNECTED_BIT = BIT0;
static int ws_fd = -1;
static QueueHandle_t audio_tx_queue = NULL;
static QueueHandle_t audio_rx_queue = NULL;
static i2s_chan_handle_t i2s_rx_handle = NULL;
static i2s_chan_handle_t i2s_tx_handle = NULL;

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
    Serial.println("[DIAG] Scanning I2C bus...");
    // Try XIAO-C6 default I2C pins first (17/18), then 21/20
    uint8_t try_pins[][2] = {{17, 18}, {21, 20}, {8, 9}};
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
            }
        }
        if (!found) Serial.print("No devices");
        Serial.println();
        i2c.end();
        if (found > 0) break;
    }
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
    Serial.print("[5] Connecting to WiFi: "); Serial.println(SSID);
    xEventGroupWaitBits(wifi_event_group, WIFI_CONNECTED_BIT, false, true,
                         pdMS_TO_TICKS(WIFI_CONNECT_TIMEOUT_SECS * 1000));

    if (!(xEventGroupGetBits(wifi_event_group) & WIFI_CONNECTED_BIT)) {
        Serial.println("[5] WiFi FAILED! Rebooting...");
        delay(2000);
        esp_restart();
    }
    Serial.println("[6] WiFi connected!");

    // Queues
    audio_tx_queue = xQueueCreate(AUDIO_QUEUE_LEN, sizeof(void *));
    audio_rx_queue = xQueueCreate(AUDIO_QUEUE_LEN, sizeof(void *));
    Serial.println("[7] Queues created");

    // I2S
    i2s_init();
    Serial.println("[8] I2S RX OK (TX skipped)");

    // Tasks
    xTaskCreate(i2s_capture_task, "i2s_capture", 8192, NULL, 5, NULL);
    xTaskCreate(audio_playback_task, "audio_playback", 8192, NULL, 5, NULL);
    xTaskCreate(ws_client_task, "ws_client", 16384, NULL, 6, NULL);
    Serial.println("[9] All tasks started!");
    Serial.flush();
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
    Serial.println("[8a] Creating I2S RX channel...");
    Serial.flush();
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
    esp_err_t ret = i2s_new_channel(&chan_cfg, NULL, &i2s_rx_handle);
    if (ret != ESP_OK) {
        Serial.print("[8a] I2S RX channel fail: "); Serial.println(esp_err_to_name(ret));
        return ret;
    }
    Serial.println("[8b] Configuring I2S RX...");
    Serial.flush();
    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_STD_MSB_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .bclk = I2S_GPIO_BCLK,
            .ws = I2S_GPIO_LRCLK,
            .din = I2S_GPIO_DATA,
        },
    };
    ret = i2s_channel_init_std_mode(i2s_rx_handle, &std_cfg);
    if (ret != ESP_OK) {
        Serial.print("[8b] I2S init fail: "); Serial.println(esp_err_to_name(ret));
        return ret;
    }
    Serial.println("[8c] I2S RX OK");
    Serial.flush();
    // Skip TX for now - XVF3800 handles audio output
    return ESP_OK;
}

static void i2s_capture_task(void *arg)
{
    size_t bytes_read;
    int16_t *buffer = (int16_t *)heap_caps_malloc(I2S_BUFFER_SIZE * 2, MALLOC_CAP_DMA);
    if (!buffer) { Serial.println("Buf alloc fail"); vTaskDelete(NULL); return; }

    uint32_t frame_count = 0;
    uint32_t last_print = 0;
    while (1) {
        i2s_channel_read(i2s_rx_handle, buffer, I2S_BUFFER_SIZE * 2, &bytes_read, portMAX_DELAY);
        if (bytes_read > 0) {
            frame_count++;
            // Debug: print frame count every 10 seconds
            if (frame_count % 100 == 0 && millis() - last_print > 10000) {
                last_print = millis();
                Serial.printf("[I2S] %lu frames, %lu samples read\n", frame_count, bytes_read / 2);
            }
            size_t total_len = 3 + bytes_read;
            uint8_t *frame = (uint8_t *)malloc(total_len);
            if (!frame) { Serial.println("Frame alloc fail"); continue; }
            frame[0] = FRAMETYPE_AUDIO_RX;
            frame[1] = (bytes_read >> 8) & 0xFF;
            frame[2] = bytes_read & 0xFF;
            memcpy(&frame[3], buffer, bytes_read);
            xQueueSend(audio_tx_queue, &frame, portMAX_DELAY);
        }
    }
    heap_caps_free(buffer);
    vTaskDelete(NULL);
}

static void audio_playback_task(void *arg)
{
    // TX I2S not initialized — skip playback until reSpeaker output is wired up
    // When enabled: i2s_channel_write(i2s_tx_handle, ...)
    while (1) {
        void *frame_ptr;
        if (xQueueReceive(audio_rx_queue, &frame_ptr, portMAX_DELAY) == pdTRUE) {
            // free allocated audio data
            free(frame_ptr);
            // TODO: when TX I2S is re-enabled:
            // i2s_channel_write(i2s_tx_handle, frame_ptr, payload_size, &bytes_written, portMAX_DELAY);
        }
    }
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

static void ws_client_task(void *arg)
{
    uint8_t rx_buf[4096];
    int reconnect_delay = 1000;
    while (1) {
        if (ws_fd < 0) {
            ws_fd = ws_connect();
            if (ws_fd >= 0) {
                reconnect_delay = 1000;
                uint8_t start_frame[] = {FRAMETYPE_SESSION};
                ws_send_frame(ws_fd, 0, start_frame, sizeof(start_frame));
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
        void *frame_ptr;
        int sent = 0;
        while (xQueuePeek(audio_tx_queue, &frame_ptr, 0) == pdTRUE) {
            if (xQueueReceive(audio_tx_queue, &frame_ptr, 0) == pdTRUE) {
                uint8_t *frame = (uint8_t *)frame_ptr;
                size_t audio_len = ((size_t)frame[1] << 8) | frame[2];
                size_t total_payload = 3 + audio_len;  // 1 frame_type + 2 length + audio
                ws_send_frame(ws_fd, 0, frame, total_payload);
                free(frame);
                sent++;
            } else {
                break;
            }
        }
        if (sent) Serial.printf("[SEND] %d frames\n", sent);

        if (ret < 0) continue;  // Error
        if (ret == 0) continue;  // Timeout, check queue

        // Received data from Hub
        int bytes = recv(ws_fd, rx_buf, sizeof(rx_buf), 0);
        if (bytes <= 0) {
            Serial.println("WS lost, reconnecting...");
            close(ws_fd); ws_fd = -1;
            vTaskDelay(pdMS_TO_TICKS(reconnect_delay));
            reconnect_delay = reconnect_delay < 30000 ? reconnect_delay * 2 : reconnect_delay;
            continue;
        }
        if (bytes < 4) continue;  // minimum WebSocket frame

        uint8_t mask_flag = rx_buf[1] & 0x80;
        size_t payload_len = rx_buf[1] & 0x7F;
        size_t header_offset = 2;

        // Extended payload length
        if (payload_len == 126) {
            if (bytes < 4) continue;
            payload_len = ((uint16_t)rx_buf[2] << 8) | rx_buf[3];
            header_offset = 4;
        } else if (payload_len == 127) {
            if (bytes < 10) continue;
            uint64_t plen = ((uint64_t)rx_buf[2] << 56) | ((uint64_t)rx_buf[3] << 48) |
                            ((uint64_t)rx_buf[4] << 40) | ((uint64_t)rx_buf[5] << 32) |
                            ((uint64_t)rx_buf[6] << 24) | ((uint64_t)rx_buf[7] << 16) |
                            ((uint64_t)rx_buf[8] << 8) | (uint64_t)rx_buf[9];
            if (plen > sizeof(rx_buf)) continue;  // reject oversized
            payload_len = (size_t)plen;
            header_offset = 10;
        }

        // Unmask if needed
        uint8_t *payload_start = NULL;
        if (mask_flag) {
            if (header_offset + 4 > (size_t)bytes) continue;  // not enough for mask key
            uint32_t mask_key;
            memcpy(&mask_key, &rx_buf[header_offset], 4);
            header_offset += 4;
            if (header_offset + payload_len > (size_t)bytes) continue;  // partial frame
            for (size_t i = 0; i < payload_len; i++) {
                rx_buf[header_offset + i] ^= ((mask_key >> ((i % 4) * 8)) & 0xFF);
            }
            payload_start = &rx_buf[header_offset];
        } else {
            payload_start = &rx_buf[header_offset];
        }

        uint8_t opcode = rx_buf[0] & 0x0F;
        uint8_t frame_type = payload_start[0];
        uint8_t *payload = &payload_start[1];
        size_t payload_size = payload_len - 1;

        if (opcode == 0x08) {  // Close frame
            Serial.println("WS closed by server");
            close(ws_fd); ws_fd = -1;
            vTaskDelay(pdMS_TO_TICKS(reconnect_delay));
            continue;
        }

        if (opcode == 0x09) {  // PING - respond with PONG
            uint8_t pong[14 + 126];  // header(2) + mask(4) + payload(126) max
            pong[0] = 0x8A;  // FIN + PONG
            pong[1] = 0x80 | payload_len;  // masked + length
            uint8_t mask_key[4] = {0xDE, 0xAD, 0xBE, 0xEF};
            memcpy(&pong[2], mask_key, 4);
            for (size_t i = 0; i < payload_len; i++) {
                pong[6 + i] = payload_start[i] ^ mask_key[i % 4];
            }
            send(ws_fd, pong, 6 + payload_len, 0);
            continue;
        }

        if (frame_type == 0x11) {  // Audio response
            void *copy = malloc(payload_size);
            if (copy) {
                memcpy(copy, payload, payload_size);
                xQueueSend(audio_rx_queue, &copy, portMAX_DELAY);
            }
        } else if (frame_type == 0x13) {  // Status
            Serial.print("STATUS: "); Serial.println((char *)payload);
        }
    }
}

// ── Arduino entry point ──────────────────────────────────────────
void setup()
{
    Serial.begin(115200);
    while (!Serial) { delay(10); }
    Serial.println("GLaDOS Spoke starting...");
    delay(100);
    i2c_scan();
    glados_main();
}

void loop()
{
    vTaskDelete(NULL);
}
