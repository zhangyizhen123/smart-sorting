#ifndef SORTING_CONTROLLER_H
#define SORTING_CONTROLLER_H

#include <stdbool.h>
#include <stdint.h>

#include "sorting_protocol.h"

typedef enum {
    SORT_BOOT = 0,
    SORT_HOME_TURNTABLE,
    SORT_HOME_XY,
    SORT_WAIT_TARGET,
    SORT_MOVE_TO_TARGET,
    SORT_GRIP_DESCEND,
    SORT_GRIP_CLOSE,
    SORT_GRIP_LIFT,
    SORT_MOVE_TO_DROP,
    SORT_ROTATE_BIN,
    SORT_RELEASE,
    SORT_WAIT_DROP_SENSOR,
    SORT_NOTIFY_SUCCESS,
    SORT_NOTIFY_FAILURE,
    SORT_FAULT
} sort_state_t;

typedef struct {
    uint8_t shape_id;
    uint8_t color_id;
    uint8_t bin_id;
    float x_mm;
    float y_mm;
    float confidence;
    uint16_t frame_id;
} sort_target_t;

typedef struct {
    void (*start_turntable_home)(void *ctx);
    void (*start_xy_home)(void *ctx);
    void (*move_xy)(void *ctx, float x_mm, float y_mm);
    void (*gripper_descend)(void *ctx);
    void (*gripper_close)(void *ctx);
    void (*gripper_lift)(void *ctx);
    void (*move_to_drop)(void *ctx);
    void (*rotate_to_bin)(void *ctx, uint8_t bin_id);
    void (*gripper_release)(void *ctx);
    void (*display_drop)(void *ctx, uint8_t shape_id, uint8_t color_id,
                         uint8_t bin_id, uint32_t success_count);
    void (*send_status)(void *ctx, uint8_t message_type,
                        uint8_t status, uint16_t sequence);
    void (*request_rescan)(void *ctx);
    void (*motion_stop)(void *ctx);
    void *ctx;
} sort_hal_t;

typedef struct {
    sort_state_t state;
    sort_target_t target;
    uint32_t state_started_ms;
    uint32_t drop_started_ms;
    uint32_t sensor_active_since_ms;
    uint32_t success_count;
    uint16_t operation_sequence;
    uint16_t drop_timeout_ms;
    uint16_t sensor_debounce_ms;
    uint32_t motion_timeout_ms;
    bool target_valid;
    bool sensor_was_active;
    bool drop_sensor_armed;
    /* A debounced drop seen while the gripper is still releasing; reported
     * once the release motion completes (O-04). */
    bool drop_confirmed;
    bool fault_latched;
} sort_controller_t;

void sort_controller_init(sort_controller_t *controller, const sort_hal_t *hal,
                          uint32_t now_ms, uint16_t drop_timeout_ms,
                          uint16_t sensor_debounce_ms, uint32_t motion_timeout_ms);
void sort_controller_tick(sort_controller_t *controller, const sort_hal_t *hal,
                          uint32_t now_ms, bool turntable_home,
                          bool xy_home, bool drop_sensor_active);
int sort_controller_accept_frame(sort_controller_t *controller,
                                 const sort_hal_t *hal,
                                 const uint8_t *raw, uint16_t length,
                                 uint32_t now_ms);
void sort_controller_motion_complete(sort_controller_t *controller,
                                     const sort_hal_t *hal, uint32_t now_ms);
void sort_controller_estop(sort_controller_t *controller,
                           const sort_hal_t *hal, uint32_t now_ms);

#endif
