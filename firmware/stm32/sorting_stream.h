#ifndef SORTING_STREAM_H
#define SORTING_STREAM_H

#include <stddef.h>
#include <stdint.h>

#include "sorting_protocol.h"

/* Platform-independent UART byte-stream framer for the SS protocol.
 *
 * Feed it any chunking of received bytes (one byte from an RX interrupt, a
 * DMA half/full buffer, an idle-line burst). Every complete frame whose
 * header, length and CRC are valid is handed to the callback exactly once.
 *
 * Resynchronisation follows vision/runtime/protocol.py StreamDecoder: a bad
 * candidate (wrong version, oversized length or CRC mismatch) discards only
 * its first byte and the remaining buffered bytes are scanned again, so a
 * real frame hidden behind a false "SS" header is still recovered.
 *
 * The buffer is fixed at SS_MAX_FRAME_SIZE bytes; no input can make it grow.
 * The callback's raw pointer is valid only during the call, and the callback
 * must not feed or reset the same stream. Run the feed from one context only
 * (e.g. the main loop draining a DMA ring), not concurrently from an ISR.
 *
 * Unlike the Python decoder, a CRC-valid frame with an unknown message type
 * is still delivered; the consumer (sort_controller_accept_frame) replies to
 * it instead of letting the sender time out.
 */

typedef void (*ss_frame_handler_t)(void *ctx, const uint8_t *raw, uint16_t length);

typedef struct {
    uint32_t frames_ok;
    uint32_t crc_errors;
    uint32_t length_errors;
    uint32_t version_errors;
    uint32_t discarded_bytes;
} ss_stream_stats_t;

typedef struct {
    uint8_t buffer[SS_MAX_FRAME_SIZE];
    uint16_t length;
    ss_stream_stats_t stats;
} ss_stream_t;

void ss_stream_init(ss_stream_t *stream);

/* Drop any partial frame (e.g. after a UART overrun or a round reset).
 * The dropped bytes are counted in discarded_bytes; other stats are kept. */
void ss_stream_reset(ss_stream_t *stream);

/* Returns the number of frames delivered during this call. */
size_t ss_stream_feed(ss_stream_t *stream, const uint8_t *data, size_t length,
                      ss_frame_handler_t handler, void *ctx);

/* Call when the RX line has been idle longer than any intra-frame gap the
 * sender can produce (e.g. UART IDLE interrupt plus a few byte times).
 * A false "SS" header with a large length would otherwise hold a real frame
 * behind it until enough later bytes arrive. Idle flush treats the pending
 * candidate as failed, rescans the buffered bytes, delivers any complete
 * frame found and leaves the buffer empty. Returns frames delivered. */
size_t ss_stream_idle(ss_stream_t *stream, ss_frame_handler_t handler, void *ctx);

#endif
