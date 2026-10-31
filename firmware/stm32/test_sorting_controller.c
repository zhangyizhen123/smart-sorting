#include "sorting_controller.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    unsigned home_turntable_count;
    unsigned home_xy_count;
    unsigned move_count;
    unsigned descend_count;
    unsigned close_count;
    unsigned lift_count;
    unsigned move_to_drop_count;
    unsigned rotate_count;
    unsigned release_count;
    unsigned display_count;
    unsigned rescan_count;
    unsigned stop_count;
    unsigned status_count;
    uint8_t status_type[16];
    uint8_t status_code[16];
    uint32_t displayed_success_count;
} mock_hal_t;

static void start_turntable_home(void *ctx) { ((mock_hal_t *)ctx)->home_turntable_count++; }
static void start_xy_home(void *ctx) { ((mock_hal_t *)ctx)->home_xy_count++; }
static void move_xy(void *ctx, float x, float y) {
    mock_hal_t *mock = (mock_hal_t *)ctx;
    (void)x;
    (void)y;
    mock->move_count++;
}
static void gripper_descend(void *ctx) { ((mock_hal_t *)ctx)->descend_count++; }
static void gripper_close(void *ctx) { ((mock_hal_t *)ctx)->close_count++; }
static void gripper_lift(void *ctx) { ((mock_hal_t *)ctx)->lift_count++; }
static void move_to_drop(void *ctx) { ((mock_hal_t *)ctx)->move_to_drop_count++; }
static void rotate_to_bin(void *ctx, uint8_t bin_id) {
    mock_hal_t *mock = (mock_hal_t *)ctx;
    if (bin_id == 3u) mock->rotate_count++;
}
static void gripper_release(void *ctx) { ((mock_hal_t *)ctx)->release_count++; }
static void display_drop(void *ctx, uint8_t shape, uint8_t color,
                         uint8_t bin, uint32_t success_count) {
    mock_hal_t *mock = (mock_hal_t *)ctx;
    if (shape == 2u && color == 4u && bin == 3u) {
        mock->display_count++;
        mock->displayed_success_count = success_count;
    }
}
static void send_status(void *ctx, uint8_t type, uint8_t status, uint16_t sequence) {
    mock_hal_t *mock = (mock_hal_t *)ctx;
    unsigned index = mock->status_count;
    (void)sequence;
    if (index < 16u) {
        mock->status_type[index] = type;
        mock->status_code[index] = status;
    }
    mock->status_count++;
}
static void request_rescan(void *ctx) { ((mock_hal_t *)ctx)->rescan_count++; }
static void motion_stop(void *ctx) { ((mock_hal_t *)ctx)->stop_count++; }

static sort_hal_t make_hal(mock_hal_t *mock) {
    sort_hal_t hal;
    memset(&hal, 0, sizeof(hal));
    hal.start_turntable_home = start_turntable_home;
    hal.start_xy_home = start_xy_home;
    hal.move_xy = move_xy;
    hal.gripper_descend = gripper_descend;
    hal.gripper_close = gripper_close;
    hal.gripper_lift = gripper_lift;
    hal.move_to_drop = move_to_drop;
    hal.rotate_to_bin = rotate_to_bin;
    hal.gripper_release = gripper_release;
    hal.display_drop = display_drop;
    hal.send_status = send_status;
    hal.request_rescan = request_rescan;
    hal.motion_stop = motion_stop;
    hal.ctx = mock;
    return hal;
}

static size_t make_target_frame(uint8_t *raw, size_t capacity) {
    uint8_t payload[SS_VISION_RESULT_PAYLOAD_SIZE] = {0};
    float x_mm = 40.0f;
    float y_mm = 70.0f;
    float confidence = 0.9f;
    payload[0] = 2u;
    payload[1] = 4u;
    payload[2] = 3u;
    memcpy(&payload[4], &x_mm, sizeof(x_mm));
    memcpy(&payload[8], &y_mm, sizeof(y_mm));
    memcpy(&payload[12], &confidence, sizeof(confidence));
    ss_write_u16_le(&payload[16], 17u);
    return ss_encode_frame(raw, capacity, SS_MSG_VISION_RESULT,
                           payload, SS_VISION_RESULT_PAYLOAD_SIZE);
}

static void boot_to_wait_target(sort_controller_t *controller, const sort_hal_t *hal,
                                uint32_t motion_timeout_ms) {
    sort_controller_init(controller, hal, 100u, 100u, 20u, motion_timeout_ms);
    sort_controller_tick(controller, hal, 100u, false, false, false);
    sort_controller_tick(controller, hal, 105u, true, false, false);
    sort_controller_tick(controller, hal, 110u, false, true, false);
}

static int start_drop(sort_controller_t *controller, const sort_hal_t *hal,
                      uint32_t accepted_ms, uint32_t release_started_ms,
                      uint32_t release_completed_ms) {
    uint8_t raw[SS_MAX_FRAME_SIZE];
    size_t length = make_target_frame(raw, sizeof(raw));
    if (sort_controller_accept_frame(controller, hal, raw, (uint16_t)length, accepted_ms) != 0) return 0;
    sort_controller_motion_complete(controller, hal, accepted_ms + 10u); /* descend */
    sort_controller_motion_complete(controller, hal, accepted_ms + 20u); /* close */
    sort_controller_motion_complete(controller, hal, accepted_ms + 30u); /* lift */
    sort_controller_motion_complete(controller, hal, accepted_ms + 40u); /* move to drop */
    sort_controller_motion_complete(controller, hal, release_started_ms - 10u); /* align bin */
    if (controller->state != SORT_ROTATE_BIN) return 0;
    sort_controller_motion_complete(controller, hal, release_started_ms); /* start release */
    if (controller->state != SORT_RELEASE) return 0;
    sort_controller_motion_complete(controller, hal, release_completed_ms);
    return controller->state == SORT_WAIT_DROP_SENSOR &&
           controller->drop_started_ms == release_completed_ms;
}

static int test_debounced_drop_updates_display_once(void) {
    mock_hal_t mock = {0};
    sort_hal_t hal = make_hal(&mock);
    sort_controller_t controller;
    boot_to_wait_target(&controller, &hal, 2000u);
    if (!start_drop(&controller, &hal, 120u, 180u, 400u)) return 1;

    sort_controller_tick(&controller, &hal, 401u, false, false, true);
    sort_controller_tick(&controller, &hal, 410u, false, false, true);
    if (controller.state != SORT_WAIT_DROP_SENSOR || controller.drop_sensor_armed) return 2;
    sort_controller_tick(&controller, &hal, 415u, false, false, false);
    sort_controller_tick(&controller, &hal, 420u, false, false, true);
    sort_controller_tick(&controller, &hal, 430u, false, false, false); /* bounce */
    sort_controller_tick(&controller, &hal, 435u, false, false, true);
    sort_controller_tick(&controller, &hal, 454u, false, false, true);
    if (mock.display_count != 0u) return 3;
    sort_controller_tick(&controller, &hal, 455u, false, false, true);
    if (controller.state != SORT_WAIT_TARGET || controller.success_count != 1u) return 4;
    if (mock.display_count != 1u || mock.displayed_success_count != 1u) return 5;
    if (mock.rescan_count != 1u || mock.release_count != 1u || mock.rotate_count != 1u) return 6;
    if (mock.status_type[mock.status_count - 1u] != SS_MSG_DROP_SUCCESS) return 7;
    return 0;
}

static int test_drop_timeout_starts_after_release_and_rescans(void) {
    mock_hal_t mock = {0};
    sort_hal_t hal = make_hal(&mock);
    sort_controller_t controller;
    boot_to_wait_target(&controller, &hal, 2000u);
    if (!start_drop(&controller, &hal, 120u, 180u, 400u)) return 1;

    sort_controller_tick(&controller, &hal, 499u, false, false, false);
    if (controller.state != SORT_WAIT_DROP_SENSOR) return 2;
    sort_controller_tick(&controller, &hal, 500u, false, false, false);
    if (controller.state != SORT_WAIT_TARGET || controller.success_count != 0u) return 3;
    if (mock.display_count != 0u || mock.rescan_count != 1u) return 4;
    if (mock.status_type[mock.status_count - 1u] != SS_MSG_DROP_FAILED ||
        mock.status_code[mock.status_count - 1u] != SS_STATUS_SENSOR_TIMEOUT) return 5;
    return 0;
}

static int test_turntable_alignment_timeout_latches_fault(void) {
    mock_hal_t mock = {0};
    sort_hal_t hal = make_hal(&mock);
    sort_controller_t controller;
    uint8_t raw[SS_MAX_FRAME_SIZE];
    size_t length;
    boot_to_wait_target(&controller, &hal, 50u);
    length = make_target_frame(raw, sizeof(raw));
    if (sort_controller_accept_frame(&controller, &hal, raw, (uint16_t)length, 120u) != 0) return 1;
    sort_controller_motion_complete(&controller, &hal, 130u);
    sort_controller_motion_complete(&controller, &hal, 140u);
    sort_controller_motion_complete(&controller, &hal, 150u);
    sort_controller_motion_complete(&controller, &hal, 160u);
    sort_controller_motion_complete(&controller, &hal, 170u);
    if (controller.state != SORT_ROTATE_BIN) return 2;
    sort_controller_tick(&controller, &hal, 220u, false, false, false);
    if (controller.state != SORT_FAULT || !controller.fault_latched || mock.stop_count != 1u) return 3;
    if (mock.status_type[mock.status_count - 1u] != SS_MSG_BIN_POSITION_ERROR) return 4;
    return 0;
}

static int test_invalid_target_values_are_rejected(void) {
    mock_hal_t mock = {0};
    sort_hal_t hal = make_hal(&mock);
    sort_controller_t controller;
    uint8_t raw[SS_MAX_FRAME_SIZE];
    size_t length;
    float invalid_confidence = NAN;
    boot_to_wait_target(&controller, &hal, 2000u);
    length = make_target_frame(raw, sizeof(raw));
    memcpy(&raw[18], &invalid_confidence, sizeof(invalid_confidence));
    ss_write_u16_le(&raw[length - 2u], ss_crc16(&raw[2], 4u + SS_VISION_RESULT_PAYLOAD_SIZE));
    if (sort_controller_accept_frame(&controller, &hal, raw, (uint16_t)length, 120u) != -3) return 1;
    if (mock.move_count != 0u || mock.status_type[mock.status_count - 1u] != SS_MSG_ACK ||
        mock.status_code[mock.status_count - 1u] != SS_STATUS_BAD_LENGTH) return 2;

    length = make_target_frame(raw, sizeof(raw));
    raw[9] = 1u; /* reserved byte must stay zero */
    ss_write_u16_le(&raw[length - 2u], ss_crc16(&raw[2], 4u + SS_VISION_RESULT_PAYLOAD_SIZE));
    if (sort_controller_accept_frame(&controller, &hal, raw, (uint16_t)length, 121u) != -3) return 3;
    if (mock.move_count != 0u) return 4;
    return 0;
}

static int test_target_is_rejected_until_homing_finishes(void) {
    mock_hal_t mock = {0};
    sort_hal_t hal = make_hal(&mock);
    sort_controller_t controller;
    uint8_t raw[SS_MAX_FRAME_SIZE];
    size_t length;
    sort_controller_init(&controller, &hal, 100u, 100u, 20u, 2000u);
    length = make_target_frame(raw, sizeof(raw));
    if (sort_controller_accept_frame(&controller, &hal, raw, (uint16_t)length, 101u) != -4) return 1;
    if (controller.target_valid || mock.move_count != 0u) return 2;
    if (mock.status_type[0] != SS_MSG_ACK || mock.status_code[0] != SS_STATUS_BUSY) return 3;
    return 0;
}

int main(void) {
    int result;
    result = test_debounced_drop_updates_display_once();
    if (result != 0) return 10 + result;
    result = test_drop_timeout_starts_after_release_and_rescans();
    if (result != 0) return 30 + result;
    result = test_turntable_alignment_timeout_latches_fault();
    if (result != 0) return 50 + result;
    result = test_invalid_target_values_are_rejected();
    if (result != 0) return 70 + result;
    result = test_target_is_rejected_until_homing_finishes();
    if (result != 0) return 90 + result;
    puts("STM32 controller tests passed");
    return 0;
}
