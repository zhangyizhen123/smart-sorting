#include "sorting_controller.h"

#include <math.h>
#include <string.h>

static uint32_t elapsed(uint32_t now, uint32_t since) { return now - since; }

static void enter(sort_controller_t *controller, sort_state_t state, uint32_t now_ms) {
    controller->state = state;
    controller->state_started_ms = now_ms;
}

static void send_status(const sort_hal_t *hal, uint8_t type, uint8_t status,
                        uint16_t sequence) {
    if (hal->send_status != NULL) hal->send_status(hal->ctx, type, status, sequence);
}

static void request_rescan(const sort_hal_t *hal) {
    if (hal->request_rescan != NULL) hal->request_rescan(hal->ctx);
}

static void clear_drop_watch(sort_controller_t *controller) {
    controller->sensor_was_active = false;
    controller->drop_sensor_armed = false;
    controller->drop_confirmed = false;
}

/* Drop-sensor sampling shared by SORT_RELEASE and SORT_WAIT_DROP_SENSOR.
 * The watch is armed only by a clear sample taken after the bin started
 * aligning, so a beam that stays blocked (stuck object, previous part) can
 * never count as the current drop. A pulse may start, or even finish, while
 * the gripper is still releasing. */
static void sample_drop_sensor(sort_controller_t *controller, uint32_t now_ms,
                               bool drop_sensor_active) {
    if (controller->drop_confirmed) return;
    if (!drop_sensor_active) {
        controller->drop_sensor_armed = true;
        controller->sensor_was_active = false;
    } else if (controller->drop_sensor_armed) {
        if (!controller->sensor_was_active) {
            controller->sensor_was_active = true;
            controller->sensor_active_since_ms = now_ms;
        } else if (elapsed(now_ms, controller->sensor_active_since_ms) >= controller->sensor_debounce_ms) {
            controller->drop_confirmed = true;
        }
    }
}

static void report_drop_success(sort_controller_t *controller, const sort_hal_t *hal,
                                uint32_t now_ms) {
    controller->success_count++;
    if (hal->display_drop != NULL) {
        hal->display_drop(hal->ctx, controller->target.shape_id,
                          controller->target.color_id, controller->target.bin_id,
                          controller->success_count);
    }
    send_status(hal, SS_MSG_DROP_SUCCESS, SS_STATUS_OK, controller->operation_sequence);
    clear_drop_watch(controller);
    enter(controller, SORT_NOTIFY_SUCCESS, now_ms);
}

static void fail_drop(sort_controller_t *controller, const sort_hal_t *hal,
                      uint32_t now_ms) {
    controller->target_valid = false;
    clear_drop_watch(controller);
    send_status(hal, SS_MSG_DROP_FAILED, SS_STATUS_SENSOR_TIMEOUT,
                controller->operation_sequence);
    enter(controller, SORT_NOTIFY_FAILURE, now_ms);
}

void sort_controller_init(sort_controller_t *controller, const sort_hal_t *hal,
                          uint32_t now_ms, uint16_t drop_timeout_ms,
                          uint16_t sensor_debounce_ms, uint32_t motion_timeout_ms) {
    memset(controller, 0, sizeof(*controller));
    controller->drop_timeout_ms = drop_timeout_ms;
    controller->sensor_debounce_ms = sensor_debounce_ms;
    controller->motion_timeout_ms = motion_timeout_ms;
    enter(controller, SORT_BOOT, now_ms);
    (void)hal;
}

int sort_controller_accept_frame(sort_controller_t *controller,
                                 const sort_hal_t *hal,
                                 const uint8_t *raw, uint16_t length,
                                 uint32_t now_ms) {
    ss_frame_t frame;
    int decode_result;
    if (controller->fault_latched) {
        send_status(hal, SS_MSG_ACK, SS_STATUS_LIMIT_ERROR, controller->operation_sequence);
        return -4;
    }
    if (controller->state != SORT_WAIT_TARGET) {
        send_status(hal, SS_MSG_ACK, SS_STATUS_BUSY, controller->operation_sequence);
        return -4;
    }
    decode_result = ss_decode_frame(raw, length, &frame);
    if (decode_result != 0) {
        uint8_t status = decode_result == -3 ? SS_STATUS_BAD_CRC : SS_STATUS_BAD_LENGTH;
        send_status(hal, SS_MSG_ACK, status, controller->operation_sequence);
        return -1;
    }
    if (frame.type != SS_MSG_VISION_RESULT || frame.payload_len != SS_VISION_RESULT_PAYLOAD_SIZE) {
        send_status(hal, SS_MSG_ACK, SS_STATUS_BAD_LENGTH, controller->operation_sequence);
        return -2;
    }
    controller->target.shape_id = frame.payload[0];
    controller->target.color_id = frame.payload[1];
    controller->target.bin_id = frame.payload[2];
    memcpy(&controller->target.x_mm, &frame.payload[4], sizeof(float));
    memcpy(&controller->target.y_mm, &frame.payload[8], sizeof(float));
    memcpy(&controller->target.confidence, &frame.payload[12], sizeof(float));
    controller->target.frame_id = ss_read_u16_le(&frame.payload[16]);
    if (controller->target.shape_id > 8u || controller->target.color_id > 9u ||
        controller->target.bin_id < 1u || controller->target.bin_id > 6u || frame.payload[3] != 0u ||
        !isfinite(controller->target.x_mm) || !isfinite(controller->target.y_mm) ||
        !isfinite(controller->target.confidence) ||
        controller->target.confidence < 0.0f || controller->target.confidence > 1.0f) {
        send_status(hal, SS_MSG_ACK, SS_STATUS_BAD_LENGTH, controller->operation_sequence);
        return -3;
    }
    controller->target_valid = true;
    clear_drop_watch(controller);
    controller->operation_sequence++;
    send_status(hal, SS_MSG_ACK, SS_STATUS_OK, controller->operation_sequence);
    send_status(hal, SS_MSG_ACTION_STARTED, SS_STATUS_OK, controller->operation_sequence);
    if (hal->move_xy != NULL) hal->move_xy(hal->ctx, controller->target.x_mm, controller->target.y_mm);
    enter(controller, SORT_MOVE_TO_TARGET, now_ms);
    return 0;
}

void sort_controller_motion_complete(sort_controller_t *controller,
                                     const sort_hal_t *hal, uint32_t now_ms) {
    switch (controller->state) {
    case SORT_HOME_TURNTABLE:
        if (hal->start_xy_home != NULL) hal->start_xy_home(hal->ctx);
        enter(controller, SORT_HOME_XY, now_ms);
        break;
    case SORT_HOME_XY:
        send_status(hal, SS_MSG_ACK, SS_STATUS_OK, controller->operation_sequence);
        enter(controller, SORT_WAIT_TARGET, now_ms);
        break;
    case SORT_MOVE_TO_TARGET:
        if (hal->gripper_descend != NULL) hal->gripper_descend(hal->ctx);
        enter(controller, SORT_GRIP_DESCEND, now_ms);
        break;
    case SORT_GRIP_DESCEND:
        if (hal->gripper_close != NULL) hal->gripper_close(hal->ctx);
        enter(controller, SORT_GRIP_CLOSE, now_ms);
        break;
    case SORT_GRIP_CLOSE:
        if (hal->gripper_lift != NULL) hal->gripper_lift(hal->ctx);
        enter(controller, SORT_GRIP_LIFT, now_ms);
        break;
    case SORT_GRIP_LIFT:
        if (hal->move_to_drop != NULL) hal->move_to_drop(hal->ctx);
        enter(controller, SORT_MOVE_TO_DROP, now_ms);
        break;
    case SORT_MOVE_TO_DROP:
        clear_drop_watch(controller);
        if (hal->rotate_to_bin != NULL) hal->rotate_to_bin(hal->ctx, controller->target.bin_id);
        enter(controller, SORT_ROTATE_BIN, now_ms);
        break;
    case SORT_ROTATE_BIN:
        /* Keep the arm state sampled during alignment: a fast part can block
         * the beam before the first tick in SORT_RELEASE. */
        if (hal->gripper_release != NULL) hal->gripper_release(hal->ctx);
        enter(controller, SORT_RELEASE, now_ms);
        break;
    case SORT_RELEASE:
        /* The drop timeout still starts when the release finishes (O-02).
         * Sensor state carries over so a pulse that straddles the end of the
         * release is neither lost nor restarted. */
        controller->drop_started_ms = now_ms;
        if (controller->drop_confirmed) {
            report_drop_success(controller, hal, now_ms);
        } else {
            enter(controller, SORT_WAIT_DROP_SENSOR, now_ms);
        }
        break;
    default:
        break;
    }
}

void sort_controller_tick(sort_controller_t *controller, const sort_hal_t *hal,
                          uint32_t now_ms, bool turntable_home,
                          bool xy_home, bool drop_sensor_active) {
    switch (controller->state) {
    case SORT_BOOT:
        if (hal->start_turntable_home != NULL) hal->start_turntable_home(hal->ctx);
        enter(controller, SORT_HOME_TURNTABLE, now_ms);
        break;
    case SORT_HOME_TURNTABLE:
        if (turntable_home) sort_controller_motion_complete(controller, hal, now_ms);
        break;
    case SORT_HOME_XY:
        if (xy_home) sort_controller_motion_complete(controller, hal, now_ms);
        break;
    case SORT_ROTATE_BIN:
        /* The gripper still holds the part: track the latest beam state so
         * only a clear beam before the drop can arm the watch. */
        controller->drop_sensor_armed = !drop_sensor_active;
        controller->sensor_was_active = false;
        break;
    case SORT_RELEASE:
        /* Capture a drop during the release motion, but report it only after
         * the release completes so no new target can start mid-release. */
        sample_drop_sensor(controller, now_ms, drop_sensor_active);
        break;
    case SORT_WAIT_DROP_SENSOR:
        sample_drop_sensor(controller, now_ms, drop_sensor_active);
        if (controller->drop_confirmed) {
            report_drop_success(controller, hal, now_ms);
        } else if (elapsed(now_ms, controller->drop_started_ms) >= controller->drop_timeout_ms) {
            fail_drop(controller, hal, now_ms);
        }
        break;
    case SORT_FAULT:
        break;
    default:
        break;
    }
    if (!controller->fault_latched && controller->motion_timeout_ms > 0u &&
        controller->state != SORT_BOOT && controller->state != SORT_WAIT_TARGET &&
        controller->state != SORT_WAIT_DROP_SENSOR &&
        controller->state != SORT_NOTIFY_SUCCESS && controller->state != SORT_NOTIFY_FAILURE &&
        elapsed(now_ms, controller->state_started_ms) >= controller->motion_timeout_ms) {
        if (hal->motion_stop != NULL) hal->motion_stop(hal->ctx);
        controller->fault_latched = true;
        send_status(hal,
                    (controller->state == SORT_HOME_TURNTABLE || controller->state == SORT_ROTATE_BIN)
                        ? SS_MSG_BIN_POSITION_ERROR : SS_MSG_MOTION_ERROR,
                    SS_STATUS_LIMIT_ERROR, controller->operation_sequence);
        enter(controller, SORT_FAULT, now_ms);
    }
    if (controller->state == SORT_NOTIFY_SUCCESS || controller->state == SORT_NOTIFY_FAILURE) {
        controller->target_valid = false;
        request_rescan(hal);
        enter(controller, SORT_WAIT_TARGET, now_ms);
    }
}

void sort_controller_estop(sort_controller_t *controller,
                           const sort_hal_t *hal, uint32_t now_ms) {
    if (hal->motion_stop != NULL) hal->motion_stop(hal->ctx);
    controller->fault_latched = true;
    controller->target_valid = false;
    send_status(hal, SS_MSG_EMERGENCY_STOP, SS_STATUS_LIMIT_ERROR,
                controller->operation_sequence);
    enter(controller, SORT_FAULT, now_ms);
}
