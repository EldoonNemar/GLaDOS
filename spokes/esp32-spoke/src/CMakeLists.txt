idf_component_register(
    SRC_DIRS "."
    INCLUDE_DIRS "."
    REQUIRES driver nvs_flash esp_wifi esp_event esp_netif lwip
)

# KConfig defaults
target_compile_options(${COMPONENT_TARGET} PRIVATE
    "-DCONFIG_WIFI_SSID=\"1.21Jigowatts\""
    "-DCONFIG_WIFI_PASSWORD=\"Jigowatts\""
    "-DCONFIG_HUB_HOST=\"192.168.1.58\""
    "-DCONFIG_HUB_PORT=5051"
)
