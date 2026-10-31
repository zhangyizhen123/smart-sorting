#include "sorting_protocol.h"

#include <stdio.h>
#include <string.h>

int main(void) {
    uint8_t encoded[SS_MAX_FRAME_SIZE];
    const uint8_t python_status_frame[] = {
        0x53u, 0x53u, 0x01u, 0x02u, 0x04u, 0x00u,
        0x00u, 0x00u, 0x01u, 0x00u, 0x9eu, 0x23u
    };
    uint8_t payload[2] = {SS_STATUS_OK, 0u};
    ss_frame_t decoded;
    size_t length;
    if (ss_crc16((const uint8_t *)"123456789", 9u) != 0x29b1u) return 1;
    length = ss_encode_frame(encoded, sizeof(encoded), SS_MSG_ACK, payload, sizeof(payload));
    if (length != 10u) return 2;
    if (ss_decode_frame(encoded, length, &decoded) != 0) return 3;
    if (decoded.type != SS_MSG_ACK || decoded.payload_len != sizeof(payload)) return 4;
    if (decoded.payload[0] != SS_STATUS_OK) return 5;
    encoded[length - 1u] ^= 0x01u;
    if (ss_decode_frame(encoded, length, &decoded) != -3) return 6;
    encoded[length - 1u] ^= 0x01u;
    encoded[2] = (uint8_t)(SS_PROTOCOL_VERSION + 1u);
    {
        uint16_t crc = ss_crc16(&encoded[2], 6u);
        ss_write_u16_le(&encoded[length - 2u], crc);
    }
    if (ss_decode_frame(encoded, length, &decoded) != -4) return 7;
    length = ss_encode_status_frame(encoded, sizeof(encoded), SS_MSG_ACK,
                                    SS_STATUS_OK, 1u);
    if (length != sizeof(python_status_frame)) return 8;
    if (memcmp(encoded, python_status_frame, sizeof(python_status_frame)) != 0) return 9;
    puts("STM32 UART protocol tests passed");
    return 0;
}
