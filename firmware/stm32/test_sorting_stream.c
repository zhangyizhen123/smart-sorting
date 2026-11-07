#include "sorting_controller.h"
#include "sorting_stream.h"

#include <stdio.h>
#include <string.h>

#define MAX_RECORDED 256u
#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
            return 1;                                                       \
        }                                                                   \
    } while (0)

typedef struct {
    unsigned count;
    uint16_t length[MAX_RECORDED];
    uint8_t bytes[MAX_RECORDED][SS_MAX_FRAME_SIZE];
} recorder_t;

static void record(void *ctx, const uint8_t *raw, uint16_t length) {
    recorder_t *rec = (recorder_t *)ctx;
    if (rec->count < MAX_RECORDED && length <= SS_MAX_FRAME_SIZE) {
        rec->length[rec->count] = length;
        memcpy(rec->bytes[rec->count], raw, length);
    }
    rec->count++;
}

static int same_frame(const recorder_t *rec, unsigned index, const uint8_t *frame, size_t length) {
    return index < rec->count && index < MAX_RECORDED && rec->length[index] == length &&
           memcmp(rec->bytes[index], frame, length) == 0;
}

static uint32_t lcg_state = 0x5eed1234u;
static uint32_t next_random(void) {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return lcg_state >> 8;
}

static size_t make_frame(uint8_t *out, uint8_t type, uint16_t payload_len, uint8_t seed) {
    uint8_t payload[SS_MAX_PAYLOAD];
    uint16_t i;
    for (i = 0; i < payload_len; ++i) payload[i] = (uint8_t)(seed + i * 7u);
    return ss_encode_frame(out, SS_MAX_FRAME_SIZE, type, payload, payload_len);
}

static int test_every_chunk_size(void) {
    uint8_t frame[SS_MAX_FRAME_SIZE];
    size_t length = make_frame(frame, SS_MSG_VISION_RESULT, SS_VISION_RESULT_PAYLOAD_SIZE, 3u);
    size_t chunk;
    CHECK(length == 26u);
    for (chunk = 1u; chunk <= length; ++chunk) {
        ss_stream_t stream;
        recorder_t rec;
        size_t offset;
        ss_stream_init(&stream);
        memset(&rec, 0, sizeof(rec));
        for (offset = 0u; offset < length; offset += chunk) {
            size_t n = length - offset < chunk ? length - offset : chunk;
            ss_stream_feed(&stream, &frame[offset], n, record, &rec);
            if (offset + n < length) CHECK(rec.count == 0u);
        }
        CHECK(rec.count == 1u);
        CHECK(same_frame(&rec, 0u, frame, length));
        CHECK(stream.length == 0u);
        CHECK(stream.stats.discarded_bytes == 0u);
    }
    return 0;
}

static int test_back_to_back_frames(void) {
    uint8_t a[SS_MAX_FRAME_SIZE], b[SS_MAX_FRAME_SIZE], c[SS_MAX_FRAME_SIZE];
    uint8_t joined[3u * SS_MAX_FRAME_SIZE];
    size_t la = make_frame(a, SS_MSG_ACK, SS_STATUS_PAYLOAD_SIZE, 1u);
    size_t lb = make_frame(b, SS_MSG_VISION_RESULT, SS_VISION_RESULT_PAYLOAD_SIZE, 2u);
    size_t lc = make_frame(c, SS_MSG_RESET_ROUND, SS_MAX_PAYLOAD, 0x53u);
    ss_stream_t stream;
    recorder_t rec;
    CHECK(lc == SS_MAX_FRAME_SIZE);
    memcpy(joined, a, la);
    memcpy(&joined[la], b, lb);
    memcpy(&joined[la + lb], c, lc);
    ss_stream_init(&stream);
    memset(&rec, 0, sizeof(rec));
    CHECK(ss_stream_feed(&stream, joined, la + lb + lc, record, &rec) == 3u);
    CHECK(rec.count == 3u);
    CHECK(same_frame(&rec, 0u, a, la));
    CHECK(same_frame(&rec, 1u, b, lb));
    CHECK(same_frame(&rec, 2u, c, lc));
    CHECK(stream.length == 0u);
    return 0;
}

static int test_noise_and_bad_headers(void) {
    uint8_t frame[SS_MAX_FRAME_SIZE];
    size_t length = make_frame(frame, SS_MSG_VISION_RESULT, SS_VISION_RESULT_PAYLOAD_SIZE, 9u);
    const uint8_t noise[] = {
        0x00u, 0xffu, 'S', 0x01u, 'S', 'S', 'S', 0x7eu,     /* lone/odd magic */
        'S', 'S', 0x02u,                                     /* wrong version */
        'S', 'S', 0x01u, 0x02u, 0xffu, 0xffu,                /* oversized length */
        'S', 'S', 0x01u, 0x02u, 0x81u, 0x00u                 /* 129 > max payload */
    };
    ss_stream_t stream;
    recorder_t rec;
    ss_stream_init(&stream);
    memset(&rec, 0, sizeof(rec));
    ss_stream_feed(&stream, noise, sizeof(noise), record, &rec);
    CHECK(rec.count == 0u);
    CHECK(stream.length < 6u);
    ss_stream_feed(&stream, frame, length, record, &rec);
    CHECK(rec.count == 1u);
    CHECK(same_frame(&rec, 0u, frame, length));
    /* "SSS", "SS\x7e" and "SS\x02" each fail the version byte. */
    CHECK(stream.stats.version_errors == 3u);
    CHECK(stream.stats.length_errors == 2u);
    CHECK(stream.stats.discarded_bytes == sizeof(noise));
    CHECK(stream.length == 0u);
    return 0;
}

static int test_crc_error_then_valid(void) {
    uint8_t bad[SS_MAX_FRAME_SIZE], good[SS_MAX_FRAME_SIZE];
    size_t lbad = make_frame(bad, SS_MSG_VISION_RESULT, SS_VISION_RESULT_PAYLOAD_SIZE, 4u);
    size_t lgood = make_frame(good, SS_MSG_VISION_RESULT, SS_VISION_RESULT_PAYLOAD_SIZE, 5u);
    ss_stream_t stream;
    recorder_t rec;
    bad[10] ^= 0x10u; /* corrupt one payload bit */
    ss_stream_init(&stream);
    memset(&rec, 0, sizeof(rec));
    ss_stream_feed(&stream, bad, lbad, record, &rec);
    ss_stream_feed(&stream, good, lgood, record, &rec);
    CHECK(rec.count == 1u);
    CHECK(same_frame(&rec, 0u, good, lgood));
    CHECK(stream.stats.crc_errors == 1u);
    CHECK(stream.stats.discarded_bytes == lbad);
    return 0;
}

static int test_false_header_hides_real_frame(void) {
    /* A false header claiming a 128-byte payload swallows the real frame that
     * follows. It is recovered once the candidate completes, or immediately
     * when the line goes idle. */
    const uint8_t false_header[] = {'S', 'S', 0x01u, 0x01u, 0x80u, 0x00u};
    uint8_t frame[SS_MAX_FRAME_SIZE];
    uint8_t padding[SS_MAX_FRAME_SIZE];
    size_t length = make_frame(frame, SS_MSG_VISION_RESULT, SS_VISION_RESULT_PAYLOAD_SIZE, 6u);
    ss_stream_t stream;
    recorder_t rec;

    ss_stream_init(&stream);
    memset(&rec, 0, sizeof(rec));
    ss_stream_feed(&stream, false_header, sizeof(false_header), record, &rec);
    ss_stream_feed(&stream, frame, length, record, &rec);
    CHECK(rec.count == 0u);
    CHECK(ss_stream_idle(&stream, record, &rec) == 1u);
    CHECK(rec.count == 1u);
    CHECK(same_frame(&rec, 0u, frame, length));
    CHECK(stream.length == 0u);
    CHECK(stream.stats.discarded_bytes == sizeof(false_header));

    memset(padding, 0, sizeof(padding));
    ss_stream_init(&stream);
    memset(&rec, 0, sizeof(rec));
    ss_stream_feed(&stream, false_header, sizeof(false_header), record, &rec);
    ss_stream_feed(&stream, frame, length, record, &rec);
    ss_stream_feed(&stream, padding, sizeof(padding), record, &rec);
    CHECK(rec.count == 1u);
    CHECK(same_frame(&rec, 0u, frame, length));
    CHECK(stream.stats.crc_errors == 1u);
    return 0;
}

static int test_reset_and_null_arguments(void) {
    uint8_t frame[SS_MAX_FRAME_SIZE];
    size_t length = make_frame(frame, SS_MSG_ACK, SS_STATUS_PAYLOAD_SIZE, 7u);
    ss_stream_t stream;
    recorder_t rec;
    ss_stream_init(&stream);
    memset(&rec, 0, sizeof(rec));
    ss_stream_feed(&stream, frame, 5u, record, &rec);
    CHECK(stream.length == 5u);
    ss_stream_reset(&stream);
    CHECK(stream.length == 0u);
    CHECK(stream.stats.discarded_bytes == 5u);
    ss_stream_feed(&stream, frame, length, record, &rec);
    CHECK(rec.count == 1u);
    CHECK(ss_stream_feed(NULL, frame, length, record, &rec) == 0u);
    CHECK(ss_stream_feed(&stream, NULL, 3u, record, &rec) == 0u);
    CHECK(ss_stream_feed(&stream, NULL, 0u, record, &rec) == 0u);
    CHECK(ss_stream_idle(NULL, record, &rec) == 0u);
    /* A NULL handler still consumes frames. */
    CHECK(ss_stream_feed(&stream, frame, length, NULL, NULL) == 1u);
    CHECK(stream.stats.frames_ok == 2u);
    return 0;
}

static int test_randomized_noise_and_chunking(void) {
    static uint8_t wire[200u * (SS_MAX_FRAME_SIZE + 24u)];
    static uint8_t expected[200u][SS_MAX_FRAME_SIZE];
    static uint16_t expected_length[200u];
    static recorder_t rec;
    const uint8_t noise_alphabet[] = {'S', 'S', 'S', 0x01u, 0x00u, 0x80u, 0xffu, 0x12u};
    ss_stream_t stream;
    size_t wire_len = 0u;
    size_t offset = 0u;
    unsigned i;
    for (i = 0; i < 200u; ++i) {
        unsigned noise = next_random() % 24u;
        unsigned j;
        uint8_t type = (uint8_t)(1u + next_random() % 9u);
        uint16_t payload_len = (uint16_t)(next_random() % (SS_MAX_PAYLOAD + 1u));
        for (j = 0; j < noise; ++j) {
            wire[wire_len++] = noise_alphabet[next_random() % sizeof(noise_alphabet)];
        }
        expected_length[i] = (uint16_t)make_frame(expected[i], type, payload_len,
                                                  (uint8_t)next_random());
        memcpy(&wire[wire_len], expected[i], expected_length[i]);
        wire_len += expected_length[i];
    }
    ss_stream_init(&stream);
    memset(&rec, 0, sizeof(rec));
    while (offset < wire_len) {
        size_t n = 1u + next_random() % 64u;
        if (n > wire_len - offset) n = wire_len - offset;
        ss_stream_feed(&stream, &wire[offset], n, record, &rec);
        CHECK(stream.length < SS_MAX_FRAME_SIZE);
        offset += n;
    }
    ss_stream_idle(&stream, record, &rec);
    CHECK(rec.count == 200u);
    for (i = 0; i < 200u; ++i) CHECK(same_frame(&rec, i, expected[i], expected_length[i]));
    CHECK(stream.stats.frames_ok == 200u);
    return 0;
}

typedef struct {
    sort_controller_t *controller;
    const sort_hal_t *hal;
    int last_result;
    unsigned calls;
} controller_sink_t;

static void to_controller(void *ctx, const uint8_t *raw, uint16_t length) {
    controller_sink_t *sink = (controller_sink_t *)ctx;
    sink->last_result = sort_controller_accept_frame(sink->controller, sink->hal, raw, length, 100u);
    sink->calls++;
}

static int test_fragmented_target_reaches_controller(void) {
    sort_controller_t controller;
    sort_hal_t hal;
    controller_sink_t sink;
    ss_stream_t stream;
    uint8_t payload[SS_VISION_RESULT_PAYLOAD_SIZE] = {2u, 4u, 3u, 0u};
    uint8_t wire[3u + SS_MAX_FRAME_SIZE];
    const float x = 120.5f, y = -35.25f, confidence = 0.875f;
    size_t length;
    size_t offset;
    memset(&hal, 0, sizeof(hal));
    memcpy(&payload[4], &x, sizeof(float));
    memcpy(&payload[8], &y, sizeof(float));
    memcpy(&payload[12], &confidence, sizeof(float));
    ss_write_u16_le(&payload[16], 0x1234u);
    wire[0] = 0x00u;
    wire[1] = 'S';
    wire[2] = 0xa5u;
    length = 3u + ss_encode_frame(&wire[3], SS_MAX_FRAME_SIZE, SS_MSG_VISION_RESULT,
                                  payload, sizeof(payload));
    sort_controller_init(&controller, &hal, 0u, 1000u, 20u, 0u);
    sort_controller_tick(&controller, &hal, 1u, false, false, false);
    sort_controller_tick(&controller, &hal, 2u, true, false, false);
    sort_controller_tick(&controller, &hal, 3u, false, true, false);
    CHECK(controller.state == SORT_WAIT_TARGET);
    memset(&sink, 0, sizeof(sink));
    sink.controller = &controller;
    sink.hal = &hal;
    ss_stream_init(&stream);
    for (offset = 0u; offset < length; offset += 3u) {
        size_t n = length - offset < 3u ? length - offset : 3u;
        ss_stream_feed(&stream, &wire[offset], n, to_controller, &sink);
    }
    CHECK(sink.calls == 1u);
    CHECK(sink.last_result == 0);
    CHECK(controller.state == SORT_MOVE_TO_TARGET);
    CHECK(controller.target.shape_id == 2u && controller.target.color_id == 4u);
    CHECK(controller.target.bin_id == 3u && controller.target.frame_id == 0x1234u);
    CHECK(controller.target.x_mm == x && controller.target.y_mm == y);
    CHECK(controller.target.confidence == confidence);
    CHECK(stream.stats.discarded_bytes == 3u);
    return 0;
}

int main(void) {
    if (test_every_chunk_size() != 0) return 1;
    if (test_back_to_back_frames() != 0) return 2;
    if (test_noise_and_bad_headers() != 0) return 3;
    if (test_crc_error_then_valid() != 0) return 4;
    if (test_false_header_hides_real_frame() != 0) return 5;
    if (test_reset_and_null_arguments() != 0) return 6;
    if (test_randomized_noise_and_chunking() != 0) return 7;
    if (test_fragmented_target_reaches_controller() != 0) return 8;
    puts("STM32 UART stream tests passed");
    return 0;
}
