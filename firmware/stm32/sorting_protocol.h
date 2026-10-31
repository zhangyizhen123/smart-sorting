#ifndef SORTING_PROTOCOL_H
#define SORTING_PROTOCOL_H

#include <stddef.h>
#include <stdint.h>

/* Wire format: 'S','S', version, type, payload_len_le16, payload, crc_le16.
 * CRC-16/CCITT-FALSE covers version through the final payload byte.
 */
#define SS_PROTOCOL_VERSION 1u
#define SS_MAX_PAYLOAD 128u
#define SS_MAX_FRAME_SIZE (8u + SS_MAX_PAYLOAD)
#define SS_MAGIC_0 ((uint8_t)'S')
#define SS_MAGIC_1 ((uint8_t)'S')

typedef enum {
    SS_MSG_VISION_RESULT = 0x01,
    SS_MSG_ACK = 0x02,
    SS_MSG_ACTION_STARTED = 0x03,
    SS_MSG_DROP_SUCCESS = 0x04,
    SS_MSG_DROP_FAILED = 0x05,
    SS_MSG_MOTION_ERROR = 0x06,
    SS_MSG_BIN_POSITION_ERROR = 0x07,
    SS_MSG_EMERGENCY_STOP = 0x08,
    SS_MSG_RESET_ROUND = 0x09
} ss_message_type_t;

typedef enum {
    SS_STATUS_OK = 0,
    SS_STATUS_BAD_CRC = 1,
    SS_STATUS_BAD_LENGTH = 2,
    SS_STATUS_BUSY = 3,
    SS_STATUS_LOW_CONFIDENCE = 4,
    SS_STATUS_NO_OBJECT = 5,
    SS_STATUS_SENSOR_TIMEOUT = 6,
    SS_STATUS_LIMIT_ERROR = 7
} ss_status_code_t;

typedef struct {
    uint8_t version;
    uint8_t type;
    uint16_t payload_len;
    uint8_t payload[SS_MAX_PAYLOAD];
} ss_frame_t;

static inline uint16_t ss_read_u16_le(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static inline void ss_write_u16_le(uint8_t *p, uint16_t value) {
    p[0] = (uint8_t)(value & 0xffu);
    p[1] = (uint8_t)(value >> 8);
}

static inline uint16_t ss_crc16(const uint8_t *data, size_t length) {
    uint16_t crc = 0xffffu;
    size_t i;
    for (i = 0; i < length; ++i) {
        uint8_t bit;
        crc ^= (uint16_t)data[i] << 8;
        for (bit = 0; bit < 8u; ++bit) {
            crc = (crc & 0x8000u) ? (uint16_t)((crc << 1) ^ 0x1021u) : (uint16_t)(crc << 1);
        }
    }
    return crc;
}

static inline size_t ss_encode_frame(uint8_t *out, size_t capacity, uint8_t type,
                              const uint8_t *payload, uint16_t payload_len) {
    size_t body_len = 4u + payload_len;
    size_t frame_len = 2u + body_len + 2u;
    uint16_t crc;
    if (out == NULL || payload_len > SS_MAX_PAYLOAD || capacity < frame_len ||
        (payload_len > 0u && payload == NULL)) return 0u;
    out[0] = SS_MAGIC_0;
    out[1] = SS_MAGIC_1;
    out[2] = SS_PROTOCOL_VERSION;
    out[3] = type;
    ss_write_u16_le(&out[4], payload_len);
    if (payload_len > 0u) {
        size_t i;
        for (i = 0; i < payload_len; ++i) out[6u + i] = payload[i];
    }
    crc = ss_crc16(&out[2], body_len);
    ss_write_u16_le(&out[6u + payload_len], crc);
    return frame_len;
}

static inline int ss_decode_frame(const uint8_t *raw, size_t length, ss_frame_t *out) {
    uint16_t payload_len;
    uint16_t received_crc;
    size_t expected;
    size_t i;
    if (raw == NULL || out == NULL || length < 8u || raw[0] != SS_MAGIC_0 || raw[1] != SS_MAGIC_1) return -1;
    payload_len = ss_read_u16_le(&raw[4]);
    if (payload_len > SS_MAX_PAYLOAD) return -2;
    expected = 8u + payload_len;
    if (length != expected) return -2;
    if (raw[2] != SS_PROTOCOL_VERSION) return -4;
    received_crc = ss_read_u16_le(&raw[6u + payload_len]);
    if (ss_crc16(&raw[2], 4u + payload_len) != received_crc) return -3;
    out->version = raw[2];
    out->type = raw[3];
    out->payload_len = payload_len;
    for (i = 0; i < payload_len; ++i) out->payload[i] = raw[6u + i];
    return 0;
}

/* VISION_RESULT payload, packed little endian, 18 bytes:
 * shape_id:u8, color_id:u8, bin_id:u8, reserved:u8,
 * x_mm:f32, y_mm:f32, confidence:f32, frame_id:u16.
 * Floats are IEEE-754 binary32 little endian.
 */
#define SS_VISION_RESULT_PAYLOAD_SIZE 18u
#define SS_STATUS_PAYLOAD_SIZE 4u

static inline size_t ss_encode_status_frame(uint8_t *out, size_t capacity,
                                            uint8_t type, uint8_t status,
                                            uint16_t sequence) {
    uint8_t payload[SS_STATUS_PAYLOAD_SIZE] = {
        status, 0u, (uint8_t)(sequence & 0xffu), (uint8_t)(sequence >> 8)
    };
    if (type == SS_MSG_VISION_RESULT) return 0u;
    return ss_encode_frame(out, capacity, type, payload, SS_STATUS_PAYLOAD_SIZE);
}

#endif
