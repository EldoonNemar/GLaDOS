/**
 * GLaDOS ESP32-C6 Spoke — reSpeaker XVF3800 WebSocket frontend.
 *
 * Hardware:
 *   ESP32-C6 DevKitC-1
 *   reSpeaker XVF3800 (I2S mode — XMOS firmware configured for I2S output)
 *     - SCK -> GPIO4 (I2S BCLK)
 *     - WS  -> GPIO5 (I2S LRCLK)
 *     - SD  -> GPIO6 (I2S DIN)
 *   Built-in DAC or external amplifier for audio output
 *
 * The XVF3800 must be flashed with an I2S-output firmware.
 * See: https://github.com/reSpeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY
 */

#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/event_groups.h"
#include "freertos/semphr.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "nvs_flash.h"
#include "driver/i2s_std.h"
#include "driver/i2s_tdm.h"
#include "driver/i2s_pdm.h"
#include "driver/gpio.h"
#include "lwip/sockets.h"
#include "lwip/dns.h"
#include "lwip/netdb.h"

static const char *TAG = "glados_spoke";

// ── Configuration ──────────────────────────────────────────────────
#define SSID            CONFIG_WIFI_SSID
#define PASSWORD        CONFIG_WIFI_PASSWORD
#define HUB_HOST        CONFIG_HUB_HOST
#define HUB_PORT        CONFIG_HUB_PORT
#define HUB_PATH        "/ws/esp32-c6-001"
#define DEVICE_ID       "esp32-c6-001"
#define SAMPLE_RATE     16000
#define I2S_GPIO_BCLK   4
#define I2S_GPIO_LRCLK  5
#define I2S_GPIO_DATA   6
#define AUDIO_QUEUE_LEN 64

// WebSocket frame types
#define FRAMETYPE_AUDIO_RX  0x01
#define FRAMETYPE_STATUS_RX 0x13
#define FRAMETYPE_AUDIO_TX  0x11
#define FRAMETYPE_SESSION   0x03

// I2S buffer settings
#define I2S_BUFFER_SIZE     (SAMPLE_RATE / 10)  // 100ms chunks = 1600 samples = 3200 bytes
#define PLAYBACK_BUF_COUNT  8

// WiFi
#define WIFI_CONNECT_TIMEOUT_SECS  30
#define WIFI_RECONNECT_TIMEOUT_MS  5000

// Globals
static EventGroupHandle_t wifi_event_group;
static const int WIFI_CONNECTED_BIT = BIT0;
static int ws_fd = -1;
static QueueHandle_t audio_tx_queue = NULL;
static QueueHandle_t audio_rx_queue = NULL;

// ── WiFi callbacks ─────────────────────────────────────────────────
static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                                int32_t event_id, void *event_data);
static void wifi_connect(void);

// ── I2S capture ───────────────────────────────────────────────────
static esp_err_t i2s_init(void);
static void i2s_capture_task(void *arg);

// ── Audio playback ────────────────────────────────────────────────
static void audio_playback_task(void *arg);

// ── WebSocket ─────────────────────────────────────────────────────
static int ws_connect(void);
static int ws_send(int fd, const uint8_t *data, size_t len);
static int ws_receive_frame(int fd, uint8_t *out_buf, size_t out_len);

// ── Main ──────────────────────────────────────────────────────────
void app_main(void)
{
    ESP_LOGI(TAG, "GLaDOS ESP32-C6 Spoke starting...");

    // Initialize NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase();
        nvs_flash_init();
    }

    // Initialize network
    esp_netif_init();
    esp_event_loop_create_default();
    esp_netif_create_default_wifi_sta();

    // Initialize WiFi
    wifi_event_group = xEventGroupCreate();
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    esp_wifi_init(&cfg);

    esp_event_handler_instance_t instance_any_id;
    esp_event_handler_instance_t instance_got_ip;
    esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                         &wifi_event_handler, NULL, &instance_any_id);
    esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP,
                                         &wifi_event_handler, NULL, &instance_got_ip);

    wifi_config_t wifi_config = {
        .sta = {
            .ssid = SSID,
            .password = PASSWORD,
            .threshold.authmode = WIFI_AUTH_WPA2_PSK,
        },
    };

    esp_wifi_set_mode(WIFI_MODE_STA);
    esp_wifi_set_config(WIFI_IF_STA, &wifi_config);
    esp_wifi_start();

    ESP_LOGI(TAG, "Connecting to WiFi: %s", SSID);
    xEventGroupWaitBits(wifi_event_group, WIFI_CONNECTED_BIT, false, true,
                         pdMS_TO_TICKS(WIFI_CONNECT_TIMEOUT_SECS * 1000));

    if (!(xEventGroupGetBits(wifi_event_group) & WIFI_CONNECTED_BIT)) {
        ESP_LOGE(TAG, "WiFi connection failed! Rebooting...");
        vTaskDelay(pdMS_TO_TICKS(2000));
        esp_restart();
    }


// ── WiFi handlers ──────────────────────────────────────────────────
static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                                int32_t event_id, void *event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGW(TAG, "WiFi disconnected, reconnecting...");
        esp_wifi_connect();
        xEventGroupClearBits(wifi_event_group, WIFI_CONNECTED_BIT);
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        ESP_LOGI(TAG, "Got IP: " IPSTR, IP2STR(&event->ip_info.ip));
        xEventGroupSetBits(wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

// ── I2S init ───────────────────────────────────────────────────────
static esp_err_t i2s_init(void)
{
    i2s_chan_handle_t rx_handle = NULL;
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_AUTO, I2S_ROLE_MASTER);
    chan_cfg.buffer_queue_len = 5;
    esp_err_t ret = i2s_new_channel(&chan_cfg, NULL, &rx_handle);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "I2S channel creation failed: %s", esp_err_to_name(ret));
        return ret;
    }

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),
        .slot_cfg = I2S_STD_MSB_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .bclk = I2S_GPIO_BCLK,
            .ws = I2S_GPIO_LRCLK,
            .din = I2S_GPIO_DATA,
            .out_nlpm = GPIO_NUM_NC,
            .out_bclk_not_mclk = false,
            .out_ws_not_mclk = false,
        },
    };

    ret = i2s_channel_init_std_mode(rx_handle, &std_cfg);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "I2S std mode init failed: %s", esp_err_to_name(ret));

// ── I2S capture task ──────────────────────────────────────────────
static void i2s_capture_task(void *arg)
{
    static i2s_chan_handle_t rx_handle = NULL;
    size_t bytes_read;
    int16_t *buffer = malloc(I2S_BUFFER_SIZE * 2);  // 16-bit samples

    if (!buffer) {
        ESP_LOGE(TAG, "Failed to alloc audio buffer");
        vTaskDelete(NULL);
        return;
    }

    // Get the rx handle (set during i2s_init)
    i2s_get_channel_handle(I2S_NUM_AUTO, NULL, &rx_handle);

    while (1) {
        i2s_channel_read(rx_handle, buffer, I2S_BUFFER_SIZE * 2, &bytes_read, portMAX_DELAY);

        if (bytes_read > 0) {
            // Pack as WebSocket frame: [0x01 (type)] [2 bytes length] [audio data]
            uint8_t frame[4096];
            frame[0] = FRAMETYPE_AUDIO_RX;
            frame[1] = (bytes_read >> 8) & 0xFF;
            frame[2] = bytes_read & 0xFF;
            memcpy(&frame[3], buffer, bytes_read);

            // Send to TX queue for WebSocket task
            xQueueSend(audio_tx_queue, &frame, portMAX_DELAY);
        }
    }

    free(buffer);
}

// ── Audio playback task ──────────────────────────────────────────
static void audio_playback_task(void *arg)
{
    static int16_t play_buf[I2S_BUFFER_SIZE];
    i2s_chan_handle_t tx_handle = NULL;

    // Simple I2S TX for playback
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    i2s_channel_init_std_mode(tx_handle, &(i2s_std_config_t){
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(22050),
        .slot_cfg = I2S_STD_MSB_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {.bclk = GPIO_NUM_10, .ws = GPIO_NUM_11, .out = GPIO_NUM_12},
    });

    while (1) {
        void *frame_ptr;
        if (xQueueReceive(audio_rx_queue, &frame_ptr, portMAX_DELAY) == pdTRUE) {
            // Decode WebSocket frame: [0x11] [audio data...]
            uint8_t *frame = (uint8_t *)frame_ptr;
            size_t audio_len = strlen((char *)frame + 3);  // approximate
            memcpy(play_buf, frame + 3, audio_len);
            size_t bytes_written;
            i2s_channel_write(tx_handle, play_buf, audio_len, &bytes_written, portMAX_DELAY);
            free(frame);
        }
    }
}

        return ret;
    }

    ESP_LOGI(TAG, "I2S initialized: %d Hz, 16-bit mono, BCLK=%d, LRCLK=%d, DIN=%d",
             SAMPLE_RATE, I2S_GPIO_BCLK, I2S_GPIO_LRCLK, I2S_GPIO_DATA);
    return ESP_OK;
}

    ESP_LOGI(TAG, "WiFi connected! IP: %s", "");

    // Initialize audio queues
    audio_tx_queue = xQueueCreate(AUDIO_QUEUE_LEN, sizeof(void *));
    audio_rx_queue = xQueueCreate(AUDIO_QUEUE_LEN, sizeof(void *));

    // Initialize I2S
    i2s_init();

    // Create tasks
    xTaskCreate(i2s_capture_task, "i2s_capture", 4096, NULL, 5, NULL);
    xTaskCreate(audio_playback_task, "audio_playback", 4096, NULL, 5, NULL);
    xTaskCreate(ws_client_task, "ws_client", 8192, NULL, 6, NULL);

    ESP_LOGI(TAG, "All tasks started!");
}

// ── WebSocket helpers ─────────────────────────────────────────────
static int resolve_host(const char *host, struct sockaddr_in *addr)
{
    struct hostent *he = gethostbyname(host);
    if (!he) { ESP_LOGE(TAG, "Failed to resolve host: %s", host); return -1; }
    addr->sin_family = AF_INET;
    addr->sin_port = htons(HUB_PORT);
    addr->sin_addr = *((struct in_addr *)he->h_addr);
    return 0;
}

static int ws_connect(void)
{
    struct sockaddr_in addr;
    if (resolve_host(HUB_HOST, &addr) != 0) return -1;
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { ESP_LOGE(TAG, "Socket creation failed"); return -1; }
    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
        ESP_LOGE(TAG, "Connection failed"); close(fd); return -1;
    }
    char req[512];
    snprintf(req, sizeof(req),
        "GET %s HTTP/1.1\r\nHost: %s:%d\r\nUpgrade: websocket\r\n"
        "Connection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n", HUB_PATH, HUB_HOST, HUB_PORT);
    send(fd, req, strlen(req), 0);
    char buf[256];
    int n = recv(fd, buf, sizeof(buf) - 1, 1000);
    if (n < 0) { ESP_LOGE(TAG, "Failed to read response"); close(fd); return -1; }
    buf[n] = '\0';
    if (strstr(buf, "101") == NULL) {
        ESP_LOGE(TAG, "WS handshake failed"); close(fd); return -1;
    }
    ESP_LOGI(TAG, "WebSocket connected!");
    return fd;
}

static void ws_send_frame(int fd, uint8_t frame_type, const uint8_t *data, size_t len)
{
    uint8_t payload[4096 + 10];  // header overhead
    size_t offset = 0;
    // FIN bit + text/binary frame type
    payload[offset++] = 0x82;  // FIN=1, opcode=2 (binary)
    if (len < 125) {
        payload[offset++] = len;
    } else if (len < 65535) {
        payload[offset++] = 126;
        payload[offset++] = (len >> 8) & 0xFF;
        payload[offset++] = len & 0xFF;
    }
    memcpy(&payload[offset], data, len);
    offset += len;
    send(fd, payload, offset, 0);

// ── WebSocket client task ────────────────────────────────────────
static void ws_client_task(void *arg)
{
    uint8_t rx_buf[4096];
    int reconnect_delay = 1000;  // ms

    while (1) {
        if (ws_fd < 0) {
            ws_fd = ws_connect();
            if (ws_fd >= 0) {
                reconnect_delay = 1000;
                // Send session start
                uint8_t start_frame[] = {FRAMETYPE_SESSION};
                ws_send_frame(ws_fd, 0, start_frame, sizeof(start_frame));
            } else {
                vTaskDelay(pdMS_TO_TICKS(reconnect_delay));
                reconnect_delay = reconnect_delay < 30000 ? reconnect_delay * 2 : reconnect_delay;
                continue;
            }
        }

        // Try to receive a WebSocket frame from the hub
        int bytes = recv(ws_fd, rx_buf, sizeof(rx_buf), 0);
        if (bytes <= 0) {
            ESP_LOGW(TAG, "WS connection lost, reconnecting...");
            close(ws_fd);
            ws_fd = -1;
            vTaskDelay(pdMS_TO_TICKS(reconnect_delay));
            reconnect_delay = reconnect_delay < 30000 ? reconnect_delay * 2 : reconnect_delay;
            continue;
        }

        // Parse WebSocket frame
        if (bytes < 2) continue;
        uint8_t opcode = rx_buf[0] & 0x0F;
        uint8_t mask_flag = rx_buf[1] & 0x80;
        size_t payload_len = rx_buf[1] & 0x7F;
        size_t header_offset = 2;

        if (payload_len == 126) {
            payload_len = (rx_buf[2] << 8) | rx_buf[3];
            header_offset = 4;
        } else if (payload_len == 127) {
            payload_len = ((uint64_t)rx_buf[2] << 56) | ... ; // simplified
            header_offset = 10;
        }

        if (mask_flag) {
            // Client sends masked data, server should NOT mask
            // If masked, extract mask key
            uint32_t mask_key;
            memcpy(&mask_key, &rx_buf[header_offset], 4);
            header_offset += 4;
            for (size_t i = 0; i < payload_len && (header_offset + i) < (size_t)bytes; i++) {
                rx_buf[header_offset + i] ^= ((mask_key >> (i % 4) * 8) & 0xFF);
            }
            payload_len = bytes - header_offset;
        }

        uint8_t frame_type = rx_buf[header_offset];
        uint8_t *payload = &rx_buf[header_offset + 1];
        size_t payload_size = bytes - header_offset - 1;

        if (frame_type == 0x11) {
            // Audio chunk - send to playback queue
            void *copy = malloc(payload_size);
            if (copy) {
                memcpy(copy, payload, payload_size);
                xQueueSend(audio_rx_queue, &copy, portMAX_DELAY);
            }
        } else if (frame_type == 0x13) {
            // Status text
            ESP_LOGI(TAG, "STATUS: %.*s", (int)payload_size, payload);
        }
    }
}

}

