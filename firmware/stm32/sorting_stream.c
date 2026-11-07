#include "sorting_stream.h"

#include <string.h>

static void drop_front(ss_stream_t *stream, uint16_t count) {
    if (count >= stream->length) {
        stream->length = 0u;
        return;
    }
    memmove(stream->buffer, &stream->buffer[count], (size_t)(stream->length - count));
    stream->length = (uint16_t)(stream->length - count);
}

static void discard_one(ss_stream_t *stream) {
    stream->stats.discarded_bytes++;
    drop_front(stream, 1u);
}

/* Consume as many frames / bad bytes as the buffer allows. On return the
 * buffer holds only a prefix that could still become a valid frame. */
static size_t process(ss_stream_t *stream, ss_frame_handler_t handler, void *ctx) {
    size_t delivered = 0u;
    for (;;) {
        uint16_t payload_len;
        uint16_t frame_len;
        if (stream->length == 0u) return delivered;
        if (stream->buffer[0] != SS_MAGIC_0) {
            discard_one(stream);
            continue;
        }
        if (stream->length < 2u) return delivered;
        if (stream->buffer[1] != SS_MAGIC_1) {
            discard_one(stream);
            continue;
        }
        if (stream->length < 3u) return delivered;
        if (stream->buffer[2] != SS_PROTOCOL_VERSION) {
            stream->stats.version_errors++;
            discard_one(stream);
            continue;
        }
        if (stream->length < 6u) return delivered;
        payload_len = ss_read_u16_le(&stream->buffer[4]);
        if (payload_len > SS_MAX_PAYLOAD) {
            stream->stats.length_errors++;
            discard_one(stream);
            continue;
        }
        frame_len = (uint16_t)(8u + payload_len);
        if (stream->length < frame_len) return delivered;
        if (ss_crc16(&stream->buffer[2], 4u + (size_t)payload_len) !=
            ss_read_u16_le(&stream->buffer[6u + payload_len])) {
            stream->stats.crc_errors++;
            discard_one(stream);
            continue;
        }
        stream->stats.frames_ok++;
        delivered++;
        if (handler != NULL) handler(ctx, stream->buffer, frame_len);
        drop_front(stream, frame_len);
    }
}

void ss_stream_init(ss_stream_t *stream) {
    memset(stream, 0, sizeof(*stream));
}

void ss_stream_reset(ss_stream_t *stream) {
    stream->stats.discarded_bytes += stream->length;
    stream->length = 0u;
}

size_t ss_stream_feed(ss_stream_t *stream, const uint8_t *data, size_t length,
                      ss_frame_handler_t handler, void *ctx) {
    size_t delivered = 0u;
    size_t i;
    if (stream == NULL || (data == NULL && length > 0u)) return 0u;
    for (i = 0; i < length; ++i) {
        /* process() always leaves fewer than SS_MAX_FRAME_SIZE bytes, so this
         * guard only protects against a corrupted stream object. */
        if (stream->length >= SS_MAX_FRAME_SIZE) discard_one(stream);
        stream->buffer[stream->length++] = data[i];
        delivered += process(stream, handler, ctx);
    }
    return delivered;
}

size_t ss_stream_idle(ss_stream_t *stream, ss_frame_handler_t handler, void *ctx) {
    size_t delivered = 0u;
    if (stream == NULL) return 0u;
    while (stream->length > 0u) {
        discard_one(stream);
        delivered += process(stream, handler, ctx);
    }
    return delivered;
}
