/* Declares observation/action shapes and the rr_* ctypes ABI. */
#include "roboracer.h"
#define OBS_SIZE 26          // odom(4) + lidar(20) + prev_action(2). UPDATE if obs layout changes.
#define NUM_ATNS 2           // steer, speed
#define ACT_SIZES {1,1}      // two continuous heads
#define OBS_TENSOR_T FloatTensor

#define Env Roboracer
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {  // Read plant and env kwargs.
    env->num_agents = 1;

    // limits (get_inputs + normalisation)
    env->limits.s_min = dict_get(kwargs, "s_min")->value;
    env->limits.s_max = dict_get(kwargs, "s_max")->value;
    env->limits.sv_min = dict_get(kwargs, "sv_min")->value;
    env->limits.sv_max = dict_get(kwargs, "sv_max")->value;
    env->limits.v_switch = dict_get(kwargs, "v_switch")->value;
    env->limits.a_max = dict_get(kwargs, "a_max")->value;
    env->limits.v_min = dict_get(kwargs, "v_min")->value;
    env->limits.v_max = dict_get(kwargs, "v_max")->value;
    env->limits.tau_v = dict_get(kwargs, "tau_v")->value;
    env->limits.tau_a = dict_get(kwargs, "tau_a")->value;
    env->limits.t_vd = dict_get(kwargs, "t_vd")->value;
    env->limits.t_sd = dict_get(kwargs, "t_sd")->value;
    env->limits.tau_delta = dict_get(kwargs, "tau_delta")->value;
    env->limits.tau_delta_dot = dict_get(kwargs, "tau_delta_dot")->value;
    env->limits.steer_gamma = dict_get(kwargs, "steer_gamma")->value;
    env->limits.delta_th = dict_get(kwargs, "delta_th")->value;
    env->limits.yaw_max = dict_get(kwargs, "yaw_max")->value;
    env->limits.r_max = dict_get(kwargs, "r_max")->value;

    // chassis (integrator, RK4, collision)
    env->chassis.m = dict_get(kwargs, "m")->value;
    env->chassis.inertia = dict_get(kwargs, "i")->value;
    env->chassis.lf = dict_get(kwargs, "lf")->value;
    env->chassis.lr = dict_get(kwargs, "lr")->value;
    env->chassis.h = dict_get(kwargs, "h")->value;
    env->chassis.ego_l = dict_get(kwargs, "ego_l")->value;
    env->chassis.ego_w = dict_get(kwargs, "ego_w")->value;
    env->chassis.r_circumscribed = dict_get(kwargs, "r_circumscribed")->value;
    env->chassis.r_inscribed = dict_get(kwargs, "r_inscribed")->value;
    env->chassis.lidar_mount_offset = dict_get(kwargs, "lidar_mount_offset")->value;
    DictItem* item;
    item = dict_get_unsafe(kwargs, "gravity");
    env->chassis.gravity = item ? (float)item->value : 9.81f;
    item = dict_get_unsafe(kwargs, "sigma_fy");
    env->chassis.sigma_fy = item ? (float)item->value : 0.03f;
    item = dict_get_unsafe(kwargs, "sigma_v_floor");
    env->chassis.sigma_v_floor = item ? (float)item->value : 0.5f;
    item = dict_get_unsafe(kwargs, "tau_beta_kin");
    env->chassis.tau_beta_kin = item ? (float)item->value : 0.15f;

    // tyre: read the params + peak scale, then derive the Pacejka curve shape
    env->tyre.mu = dict_get(kwargs, "mu")->value;
    env->tyre.c_sf = dict_get(kwargs, "c_sf")->value;
    env->tyre.c_sr = dict_get(kwargs, "c_sr")->value;
    env->tyre.s = dict_get(kwargs, "tyre_s")->value;
    float alpha_pk = dict_get(kwargs, "alpha_pk")->value;
    env->tyre.c_curve = 2.0f*((float)M_PI - cr_asinf(1.0f/env->tyre.s)) / (float)M_PI;
    env->tyre.b_r = cr_tanf((float)M_PI/(2.0f*env->tyre.c_curve)) / alpha_pk;

    // timing (integer 1 ms tick counts)
    env->time.control_period = (int)dict_get(kwargs, "control_period")->value;
    env->time.odom_period = (int)dict_get(kwargs, "odom_period")->value;
    env->time.compute_latency = (int)dict_get(kwargs, "compute_latency")->value;
    env->time.compute_latency_sd = (float)dict_get(kwargs, "compute_latency_sd")->value;
    env->sim2real = (int)dict_get(kwargs, "sim2real")->value; // master switch: jitter/DR/noise

    // perception: scalars, then build the full-resolution beam LUT
    env->perception.fov = dict_get(kwargs, "lidar_fov")->value;
    env->perception.max_range = dict_get(kwargs, "lidar_max_range")->value;
    env->perception.lidar_dist = dict_get(kwargs, "lidar_mount_offset")->value;
    env->perception.n_beams = dict_get(kwargs, "lidar_num_scans")->value;
    env->perception.angle_min = -env->perception.fov / 2.0f;
    env->perception.angle_inc = env->perception.fov / (env->perception.n_beams - 1);
    init_beam_tables(&env->perception);   // single source of truth (roboracer.h)

    // observation noise (sim2real): per-channel odom std (normalised units, obs[0..3]
    // order) + lidar bias/stochastic. Lidar specs are in METRES; precompute the
    // normalised additive terms here (raytrace emits 2*range/max-1 -> span factor 2/mr).
    env->noise.odom_std[0] = dict_get(kwargs, "obs_noise_s_norm")->value;   // obs[0] s_norm
    env->noise.odom_std[1] = dict_get(kwargs, "obs_noise_yaw_rate")->value; // obs[1] r
    env->noise.odom_std[2] = dict_get(kwargs, "obs_noise_vel")->value;      // obs[2] v
    env->noise.odom_std[3] = dict_get(kwargs, "obs_noise_yaw")->value;      // obs[3] yaw
    // Deploy pose/speed sigmas in SI: sensor specs, independent of the track.
    env->noise.odom_pos_m   = dict_get(kwargs, "odom_noise_pos_m")->value;
    env->noise.odom_vel_mps = dict_get(kwargs, "odom_noise_vel_mps")->value;
    float mr = env->perception.max_range;
    env->noise.lidar_bias_amp   = dict_get(kwargs, "lidar_bias")->value       * 2.0f / mr;
    env->noise.lidar_stoch_norm = dict_get(kwargs, "lidar_stochastic")->value * 2.0f / mr;

    // reward design knobs
    env->reward.progress = dict_get(kwargs, "progress_reward")->value;
    env->reward.collision = dict_get(kwargs, "collision_reward")->value;
    env->reward.lap_completion_bonus = dict_get(kwargs, "lap_completion_reward")->value;

    env->optimal_lap_time = dict_get(kwargs, "optimal_lap_time")->value;
    env->lap_cap = LAPS_TO_FINISH;
    // Failure-window defaults: lap-start spawn, lap-based completion, global tick cap.
    env->start_ndx = 0;
    env->end_s_norm = -1.0f;
    env->tick_cap = 0;

    init(env);     // actuator delay lengths, now that limits.t_sd/t_vd are set
    c_init(env);   // mmap the track blob (EDT + raceline)
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->lap_count > 0 ? log->optimal_lap_time / (log->lap_time_sum/(float)log->lap_count) : 0.0f);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
}

/* Swap the mmap'd track blob for one env. Returns 0 on success. */
int8_t rr_set_track(void* vec_p, int env_idx, const char* path){
    StaticVec* vec = (StaticVec*)vec_p;
    if (env_idx < 0 || env_idx >= vec->size) return -1;
    Env* envs = (Env*)vec->envs;
    return set_track(&envs[env_idx], path);
}

/* Per-env helpers live here so they can see sizeof(Roboracer). */
static Env* rr_env(void* vec_p, int env_idx){
    StaticVec* vec = (StaticVec*)vec_p;
    if (env_idx < 0 || env_idx >= vec->size) return NULL;
    return &((Env*)vec->envs)[env_idx];
}

void rr_seed(void* vec_p, int env_idx, unsigned int seed){
    Env* env = rr_env(vec_p, env_idx);
    if (!env) return;
    env->rng = seed | 1u;   // nonzero, same convention as init()
    env->odom_rng = env->rng;
    env->lidar_rng = env->rng;
    xorshift32(&env->odom_rng);
    xorshift32(&env->lidar_rng);
    xorshift32(&env->lidar_rng);
}

/* Live lap state (valid between steps; wiped by the in-step auto-reset). */
int rr_current_lap(void* vec_p, int env_idx){
    Env* env = rr_env(vec_p, env_idx);
    return env ? (int)env->current_lap : -1;
}
float rr_last_lap_time(void* vec_p, int env_idx){
    Env* env = rr_env(vec_p, env_idx);
    return env ? env->last_lap_time : -1.0f;
}
// Age [s] of the odom snapshot rr_measured_state returns, relative to the scan
// taken at the current tick (the car's scan stamp minus pose stamp).
float rr_odom_age(void* vec_p, int env_idx){
    Env* env = rr_env(vec_p, env_idx);
    return env ? (env->tick - env->odom_cache_tick) * DT_SIM : 0.0f;
}

/* Latched at terminal, before the auto-reset: the finished episode's outcome. */
int rr_done_laps(void* vec_p, int env_idx){
    Env* env = rr_env(vec_p, env_idx);
    return env ? (int)env->done_lap : -1;
}
float rr_done_lap_time(void* vec_p, int env_idx){
    Env* env = rr_env(vec_p, env_idx);
    return env ? env->done_lap_time : -1.0f;
}
int rr_done_kind(void* vec_p, int env_idx){
    Env* env = rr_env(vec_p, env_idx);
    return env ? (int)env->done_kind : -1;
}



int rr_pending_control_period_ms(void* vec_p, int env_idx){
    Env* env = rr_env(vec_p, env_idx);
    return env ? env->time.pending_control_period : -1;
}

int rr_pending_compute_latency_ms(void* vec_p, int env_idx){
    Env* env = rr_env(vec_p, env_idx);
    return env ? env->time.pending_compute_latency : -1;
}

int rr_last_compute_latency_ms(void* vec_p, int env_idx){
    Env* env = rr_env(vec_p, env_idx);
    return env ? env->time.last_compute_latency : -1;
}

/* Deterministic eval: sim2real gates exactly the DR set (tyre draw, timing
 * jitter, obs noise, lidar bias, odom phase), so nominal eval = sim2real off.
 * Tyre draw happens in c_reset, so call this BEFORE rr_reset. */
void rr_set_deterministic(void* vec_p, int env_idx, int flag){
    Env* env = rr_env(vec_p, env_idx);
    if (env) env->sim2real = !flag;
}

int rr_n_beams(void* vec_p, int env_idx){
    Env* env = rr_env(vec_p, env_idx);
    return env ? (int)env->perception.n_beams : -1;
}

void rr_vehicle_state(void* vec_p, int env_idx, float* out){
    Env* env = rr_env(vec_p, env_idx);
    if (!env) return;
    /* struct State { float x, y, delta, v, yaw, r, beta, fy_f, fy_r; } */
    out[0] = env->state.x;
    out[1] = env->state.y;
    out[2] = env->state.yaw;   /* RAW yaw — python wraps it */
    out[3] = env->state.delta;
    out[4] = env->state.v;
    out[5] = env->state.r;
    out[6] = env->state.beta;
    out[7] = env->act.accel;
    out[8] = env->act.delta_dot;
    out[9] = env->state.fy_f;
    out[10] = env->state.fy_r;
}

/* Read-only collision-geometry audit for an arbitrary pose. The returned
 * values come from the exact signed-EDT sampler and oriented-corner path used
 * by check_done(), so Python can compare safety-filter wall margins against
 * the simulator's actual terminal condition without duplicating geometry. */
void rr_collision_probe_pose(
        void* vec_p, int env_idx, float x, float y, float yaw, float* out){
    Env* env = rr_env(vec_p, env_idx);
    if (!env) return;

    State probe = env->state;
    probe.x = x;
    probe.y = y;
    probe.yaw = yaw;

    const float center_edt = bilinear_sample_edt(&env->track, x, y);
    const float cyaw = cosf(yaw), syaw = sinf(yaw);
    const float hl = env->chassis.ego_l / 2.0f;
    const float hw = env->chassis.ego_w / 2.0f;
    const float bx[4] = {+hl, +hl, -hl, -hl};
    const float by[4] = {-hw, +hw, +hw, -hw};
    float min_corner_edt = INFINITY;
    for (int i = 0; i < 4; ++i) {
        const float wx = x + bx[i]*cyaw - by[i]*syaw;
        const float wy = y + bx[i]*syaw + by[i]*cyaw;
        const float d = bilinear_sample_edt(&env->track, wx, wy);
        out[3 + i] = d;
        min_corner_edt = fminf(min_corner_edt, d);
    }
    out[0] = center_edt;
    out[1] = min_corner_edt;
    State saved = env->state;
    env->state = probe;
    out[2] = (float)check_done(env);
    env->state = saved;
}

/* Tyre parameters of the current episode (sim2real draws them per reset):
 * {mu, S, alpha_pk}. */
void rr_tyre_params(void* vec_p, int env_idx, float* out){
    Env* env = rr_env(vec_p, env_idx);
    if (!env) return;
    out[0] = env->tyre.mu;
    out[1] = env->tyre.s;
    out[2] = cr_tanf((float)M_PI / (2.0f * env->tyre.c_curve)) / env->tyre.b_r;
}

void rr_measured_state(void* vec_p, int env_idx, float* out){
    Env* env = rr_env(vec_p, env_idx);
    if (!env) return;

    float position_sigma = 0.0f;
    float yaw_sigma = 0.0f;
    float velocity_sigma = 0.0f;
    float yaw_rate_sigma = 0.0f;
    if (env->sim2real) {
        position_sigma = env->noise.odom_pos_m;
        yaw_rate_sigma = env->noise.odom_std[1] * env->limits.r_max;
        velocity_sigma = env->noise.odom_vel_mps;
        yaw_sigma = env->noise.odom_std[3] * env->limits.yaw_max;
    }

    /* Deploy observation: asynchronous odom snapshot plus configured sensor
     * noise. Beta is intentionally absent. Steering is assumed measurable. */
    out[0] = env->odom_cache_x + position_sigma*rng_normal(&env->odom_rng);
    out[1] = env->odom_cache_y + position_sigma*rng_normal(&env->odom_rng);
    out[2] = normalise_yaw(
        env->odom_cache_yaw + yaw_sigma*rng_normal(&env->odom_rng));
    out[3] = env->odom_cache_delta;
    out[4] = clampf(
        env->odom_cache_v + velocity_sigma*rng_normal(&env->odom_rng),
        env->limits.v_min, env->limits.v_max);
    out[5] = clampf(
        env->odom_cache_r + yaw_rate_sigma*rng_normal(&env->odom_rng),
        -env->limits.r_max, env->limits.r_max);
}

void rr_raw_scan(void* vec_p, int env_idx, float* out){
    Env* env = rr_env(vec_p, env_idx);
    if (!env) return;
    /* Full-resolution raytrace — writes normalised 2*(range/max)-1. */
    raytrace(env, out);
    /* Convert in-place back to metres. A beam that hit nothing within
     * max_range is a no-return: the sensor reports inf, not max_range. */
    int n = (int)env->perception.n_beams;
    float mr = env->perception.max_range;
    for (int i = 0; i < n; i++){
        out[i] = out[i] >= 1.0f ? INFINITY : (out[i] + 1.0f) * 0.5f * mr;
    }
}

void rr_measured_scan(void* vec_p, int env_idx, float* out){
    Env* env = rr_env(vec_p, env_idx);
    if (!env) return;
    /* Physical LiDAR originates at the true pose, but only noisy ranges cross
     * the deploy boundary. The dedicated RNG makes reads side-effect free for
     * tyre and timing randomisation. */
    raytrace(env, out);
    int n = (int)env->perception.n_beams;
    float mr = env->perception.max_range;
    for (int i = 0; i < n; i++){
        float measured = out[i];
        int no_return = measured >= 1.0f;
        if (env->sim2real) {
            /* Draw per beam, no-return included, so the stream is the same
             * whatever the beam hit. */
            float noise = env->noise.lidar_bias_norm
                + env->noise.lidar_stoch_norm*rng_normal(&env->lidar_rng);
            measured = clampf(measured + noise, -1.0f, 1.0f);
        }
        out[i] = no_return ? INFINITY : (measured + 1.0f) * 0.5f * mr;
    }
}

void rr_set_lap_cap(void* vec_p, int env_idx, int laps){
    Env* env = rr_env(vec_p, env_idx);
    if (env) env->lap_cap = laps < 1 ? 1 : laps;
}
/* Failure-window spawn: set the racing-line start index, the completion
 * s_norm (episode ends when the car passes the failure point; -1 = lap-based
 * completion) and the per-env tick cap (0 = global MAX_TICKS).
 * Call BEFORE rr_reset; persists across auto-resets. */
void rr_set_window(void* vec_p, int env_idx, int start_ndx, float end_s_norm, int tick_cap){
    Env* env = rr_env(vec_p, env_idx);
    if (!env) return;
    env->start_ndx = start_ndx < 0 ? 0 : start_ndx;
    env->end_s_norm = end_s_norm;
    env->tick_cap = tick_cap;
}

/* Force a terminal, latch the outcome, and reset that env. */
void rr_force_terminal(void* vec_p, int env_idx){
    Env* env = rr_env(vec_p, env_idx);
    if (!env) return;
    env->terminals[0] = 1;
    env->done_lap = env->current_lap;
    env->done_lap_time = env->last_lap_time;
    env->done_kind = COLLISION;

    add_log(env);
    env->log.episode_return += env->rewards[0];
    env->log.episode_length += 1;
    c_reset(env);
}

/* Per-env reset after rr_set_track. Does not latch a terminal — the caller
 * already consumed the finished episode. Pose/rl_ndx must be re-derived
 * from the new raceline before the next step. */
void rr_reset_env(void* vec_p, int env_idx){
    Env* env = rr_env(vec_p, env_idx);
    if (env) c_reset(env);
}

void rr_render(void* vec_p, int env_idx){
    Env* env = rr_env(vec_p, env_idx);
    if (env) renderer_render(env);
}

int rr_render_available(void){
    return renderer_available();
}

/* Overlay detector boxes packed as [x, y, yaw, length, width]. */
void rr_set_det_boxes(const float* data, int n){
    renderer_set_detection_boxes(data, n);
}

void rr_set_safety(int mode){
    renderer_set_safety(mode);
}

void rr_set_trace(int on){
    renderer_set_trace(on);
}

void rr_push_trace(float x, float y, int mode){
    renderer_push_trace(x, y, mode);
}

void rr_set_filter_path(const float* data, int n){
    renderer_set_filter_path(data, n);
}

/* Moving rectangles in the track frame. Packed per obstacle:
 * [s, n, v_s, half_length, half_width, 0, 0, 0]; the trailing slots are
 * ignored. Call after reset; the layout is owned by the indexed environment
 * and the policy observation at this pose is rebuilt so lidar includes them. */
#define OBSTACLE_PACK_FLOATS 8
void rr_set_obstacles(void* vec_p, int env_idx, const float* data, int n){
    Env* env = rr_env(vec_p, env_idx);
    if (!env) return;
    if (n < 0) n = 0;
    if (n > MAX_DYNAMIC_OBSTACLES) n = MAX_DYNAMIC_OBSTACLES;
    DynamicObstacle obstacles[MAX_DYNAMIC_OBSTACLES];
    for (int i = 0; i < n; i++){
        const float* row = data + i*OBSTACLE_PACK_FLOATS;
        obstacles[i] = (DynamicObstacle){
            .s = row[0], .n = row[1], .v_s = row[2],
            .half_length = row[3], .half_width = row[4],
        };
    }
    c_set_obstacles(env, obstacles, n);
}

/* Read the live obstacles back, packed per obstacle as
 * [cx, cy, yaw, half_length, half_width, vx, vy, s]; count via *out_n. */
void rr_get_obstacles(void* vec_p, int env_idx, float* out, int* out_n){
    Env* env = rr_env(vec_p, env_idx);
    if (!env) {
        if (out_n) *out_n = 0;
        return;
    }
    DynamicObstacle buf[MAX_DYNAMIC_OBSTACLES];
    int n = 0;
    c_get_obstacles(env, buf, &n);
    for (int i = 0; i < n; i++){
        float* row = out + i*OBSTACLE_PACK_FLOATS;
        obstacle_world_velocity(&buf[i], &row[5], &row[6]);
        row[0] = buf[i].cx;
        row[1] = buf[i].cy;
        row[2] = buf[i].yaw;
        row[3] = buf[i].half_length;
        row[4] = buf[i].half_width;
        row[7] = buf[i].s;
    }
    if (out_n) *out_n = n;
}
