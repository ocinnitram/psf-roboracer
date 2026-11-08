/* Raylib renderer and display-only telemetry for Roboracer. */

#define ROBORACER_DECLARATIONS_ONLY
#include "roboracer.h"
#include "renderer.h"
#include "raylib.h"

#define RENDER_WIN_W 1440
#define RENDER_WIN_H 900
#define TOP_H   52      // top strip: lap + last + safety pill
#define HUD_H   56      // bottom strip: steer + speed bars
#define MARGIN  24      // pixel border around the map
#define WHEEL_LEN 0.13f
#define WHEEL_WID 0.052f
#define WHEEL_R   0.048f
#define MAX_DET_BOXES 16 // display cap; matches MAX_DYNAMIC_OBSTACLES
#define SAFETY_HOLD_S 0.70f
#define MAX_TRACE 160
#define MAX_PLAN 40
#define HUD_AVG_TICKS 5

typedef struct {
    Texture2D track_tex;
    bool tex_ready;
    int win_w, win_h;
    float scale, off_x, off_y, map_w_m, map_h_m;
    float zoom, pan_x, pan_y;
    bool dragging;
    Vector2 drag_last;
    int n_dets;
    float det_x[MAX_DET_BOXES], det_y[MAX_DET_BOXES];
    float det_yaw[MAX_DET_BOXES], det_l[MAX_DET_BOXES], det_w[MAX_DET_BOXES];
    int safety_raw, safety_mode;
    float safety_hold;
    int trace_on, n_trace, trace_i;
    int8_t trace_lap;
    float trace_x[MAX_TRACE], trace_y[MAX_TRACE];
    unsigned char trace_mode[MAX_TRACE];
    int trace_pending;
    float pend_x, pend_y;
    int pend_mode;
    int n_plan;
    float plan_x[MAX_PLAN], plan_y[MAX_PLAN];
    float hud_steer_buf[HUD_AVG_TICKS], hud_speed_buf[HUD_AVG_TICKS];
    int hud_avg_n, hud_avg_i, hud_avg_tick;
    float lap_hist[8];
    int lap_hist_n;
    int8_t prev_lap;
    float wheel_spin;
    int shot_done, frame_i;
} RendererState;

static RendererState renderer = {
    .win_w = RENDER_WIN_W,
    .win_h = RENDER_WIN_H,
    .zoom = 1.0f,
    .trace_lap = -1,
    .hud_avg_tick = -1,
};

// Live pose trail (--trace). 0 PPO / 1 FILTER / 2 UNSAFE. Ring-capped and
// wiped on lap change so it stays a trail, not a full-run smear.
static void trace_clear(void){
    renderer.n_trace = 0;
    renderer.trace_i = 0;
}

static void trace_commit(float x, float y, int mode){
    if (!renderer.trace_on) return;
    if (mode < 0) mode = 0;
    if (mode > 2) mode = 2;
    renderer.trace_x[renderer.trace_i] = x;
    renderer.trace_y[renderer.trace_i] = y;
    renderer.trace_mode[renderer.trace_i] = (unsigned char)mode;
    renderer.trace_i = (renderer.trace_i + 1) % MAX_TRACE;
    if (renderer.n_trace < MAX_TRACE) renderer.n_trace++;
}

static void trace_push(float x, float y, int mode){
    renderer.pend_x = x;
    renderer.pend_y = y;
    renderer.pend_mode = mode;
    renderer.trace_pending = 1;
}

// Convert world metres with y-up into centred screen pixels with y-down.
static inline Vector2 world_to_screen(const Track* t, float wx, float wy){
    return (Vector2){
        (wx - t->origin_x) * renderer.scale + renderer.off_x,
        (renderer.map_h_m - (wy - t->origin_y)) * renderer.scale + renderer.off_y
    };
}

static void layout_fit(const Track* t){
    // Whole-map view: fit the full map to the content rect, then apply zoom and pan.
    renderer.map_w_m = t->W * t->res;
    renderer.map_h_m = t->H * t->res;
    float avail_w = (float)(renderer.win_w - 2 * MARGIN);
    float avail_h = (float)(renderer.win_h - TOP_H - HUD_H - 2 * MARGIN);
    float fit = avail_w / renderer.map_w_m < avail_h / renderer.map_h_m
        ? avail_w / renderer.map_w_m : avail_h / renderer.map_h_m;
    renderer.scale = fit * renderer.zoom;
    float content_cx = MARGIN + avail_w * 0.5f;
    float content_cy = TOP_H + MARGIN + avail_h * 0.5f;
    renderer.off_x = content_cx + renderer.pan_x - (renderer.map_w_m * 0.5f) * renderer.scale;
    renderer.off_y = content_cy + renderer.pan_y - (renderer.map_h_m * 0.5f) * renderer.scale;

    // Keep the map on the content rect: centered when it fits, clamped
    // pan once zoomed past the fit scale.
    float map_px_w = renderer.map_w_m * renderer.scale;
    float map_px_h = renderer.map_h_m * renderer.scale;
    if (map_px_w <= avail_w) {
        renderer.off_x = MARGIN + (avail_w - map_px_w) * 0.5f;
    } else {
        float x_min = MARGIN + avail_w - map_px_w;
        float x_max = (float)MARGIN;
        if (renderer.off_x > x_max) renderer.off_x = x_max;
        if (renderer.off_x < x_min) renderer.off_x = x_min;
    }
    if (map_px_h <= avail_h) {
        renderer.off_y = TOP_H + MARGIN + (avail_h - map_px_h) * 0.5f;
    } else {
        float y_min = TOP_H + MARGIN + avail_h - map_px_h;
        float y_max = (float)(TOP_H + MARGIN);
        if (renderer.off_y > y_max) renderer.off_y = y_max;
        if (renderer.off_y < y_min) renderer.off_y = y_min;
    }
}

static void oriented_box(const Track* t, float cx, float cy, float ang,
                         float length, float width, Color fill, Color outline,
                         float outline_w){
    float ca = cosf(ang), sa = sinf(ang);
    float hl = 0.5f * length, hw = 0.5f * width;
    const float bx[4] = { +hl, +hl, -hl, -hl };
    const float by[4] = { -hw, +hw, +hw, -hw };
    Vector2 p[4];
    for (int i = 0; i < 4; i++){
        p[i] = world_to_screen(t, cx + bx[i] * ca - by[i] * sa,
                                  cy + bx[i] * sa + by[i] * ca);
    }
    DrawTriangleFan(p, 4, fill);
    if (outline.a && outline_w > 0.0f){
        for (int i = 0; i < 4; i++)
            DrawLineEx(p[i], p[(i + 1) % 4], outline_w, outline);
    }
}

static void draw_wheel(const Track* t, float cx, float cy, float ang, float spin){
    Color rubber = (Color){22, 22, 26, 255};
    Color rim    = (Color){168, 170, 176, 255};
    Color spoke  = (Color){214, 216, 220, 230};
    oriented_box(t, cx, cy, ang, WHEEL_LEN, WHEEL_WID, rubber, rim, 1.4f);
    float ca = cosf(ang), sa = sinf(ang);
    float half = 0.42f * WHEEL_LEN * cosf(spin);
    Vector2 a = world_to_screen(t, cx - half * ca, cy - half * sa);
    Vector2 b = world_to_screen(t, cx + half * ca, cy + half * sa);
    DrawLineEx(a, b, 2.0f, spoke);
    DrawCircleV(world_to_screen(t, cx, cy), 2.0f, rim);
}

// Bake the occupancy corridor and collision boundary into the track texture.
static void build_track_texture(const Track* t){
    const Color outside = (Color){46, 58, 89, 255};
    const Color asphalt = (Color){58, 64, 74, 255};
    const Color wall    = (Color){236, 240, 246, 255};
    const int n = t->W * t->H;
    Color* px = (Color*)malloc((size_t)n * sizeof(Color));
    unsigned char* seen = (unsigned char*)calloc((size_t)n, 1);
    int* q = (int*)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) px[i] = outside;

    // Flood edt>0 from the raceline so asphalt covers the occupancy corridor.
    int qt = 0;
    for (int i = 0; i < t->n_rline; i++){
        int ix = (int)floorf((t->rline[i*10 + 0] - t->origin_x) / t->res);
        int iy = (int)floorf((t->rline[i*10 + 1] - t->origin_y) / t->res);
        if (ix < 0 || iy < 0 || ix >= t->W || iy >= t->H) continue;
        int id = iy * t->W + ix;
        if (seen[id] || t->edt[id] <= 0.0f) continue;
        seen[id] = 1;
        q[qt++] = id;
    }
    const int dx[4] = {1, -1, 0, 0};
    const int dy[4] = {0, 0, 1, -1};
    for (int qh = 0; qh < qt; qh++){
        int id = q[qh];
        int ix = id % t->W, iy = id / t->W;
        for (int k = 0; k < 4; k++){
            int xx = ix + dx[k], yy = iy + dy[k];
            if (xx < 0 || yy < 0 || xx >= t->W || yy >= t->H) continue;
            int nid = yy * t->W + xx;
            if (seen[nid] || t->edt[nid] <= 0.0f) continue;
            seen[nid] = 1;
            q[qt++] = nid;
        }
    }

    for (int iy = 0; iy < t->H; iy++){
        for (int ix = 0; ix < t->W; ix++){
            int id = iy * t->W + ix;
            int tex = (t->H - 1 - iy) * t->W + ix;
            if (t->edt[id] <= 0.0f) px[tex] = wall;
            else if (seen[id]) px[tex] = asphalt;
        }
    }
    free(seen);
    free(q);

    Image img = { .data = px, .width = t->W, .height = t->H, .mipmaps = 1,
                  .format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8 };
    renderer.track_tex = LoadTextureFromImage(img);
    SetTextureFilter(renderer.track_tex, TEXTURE_FILTER_POINT);
    free(px);
    renderer.tex_ready = true;
}

void renderer_render(Roboracer* env) {  // Draw one env into a raylib window.
    Track* t = &env->track;   // single source for the whole draw body

    if (!IsWindowReady()) {
#ifndef RAYLIB_STUB_H
        // InitWindow floods INFO logs on startup; the window is display-only,
        // so raise the threshold to warnings to keep the eval output clean.
        SetTraceLogLevel(LOG_WARNING);
#endif
        InitWindow(renderer.win_w, renderer.win_h, "Roboracer");
        SetTargetFPS(60);
        build_track_texture(t);
    } else if (!renderer.tex_ready) {
        build_track_texture(t);
    }
    layout_fit(t);

    // Mouse: wheel zoom about the cursor, left-drag pan.
    float wheel = GetMouseWheelMove();
    if (wheel != 0.0f){
        Vector2 m = GetMousePosition();
        float wx = (m.x - renderer.off_x) / renderer.scale + t->origin_x;
        float wy = t->origin_y + renderer.map_h_m - (m.y - renderer.off_y) / renderer.scale;
        renderer.zoom *= powf(1.2f, wheel);
        if (renderer.zoom < 0.5f) renderer.zoom = 0.5f;
        if (renderer.zoom > 60.0f) renderer.zoom = 60.0f;
        layout_fit(t);
        renderer.pan_x += m.x - ((wx - t->origin_x) * renderer.scale + renderer.off_x);
        renderer.pan_y += m.y - ((renderer.map_h_m - (wy - t->origin_y)) * renderer.scale + renderer.off_y);
        layout_fit(t);
    }
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)){
        renderer.dragging = true;
        renderer.drag_last = GetMousePosition();
    }
    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)){
        renderer.dragging = false;
    }
    if (renderer.dragging && IsMouseButtonDown(MOUSE_BUTTON_LEFT)){
        Vector2 m = GetMousePosition();
        renderer.pan_x += m.x - renderer.drag_last.x;
        renderer.pan_y += m.y - renderer.drag_last.y;
        renderer.drag_last = m;
        layout_fit(t);
    }

    if (WindowShouldClose() || IsKeyDown(KEY_ESCAPE)) {
        if (renderer.tex_ready) { UnloadTexture(renderer.track_tex); renderer.tex_ready = false; }
        CloseWindow();
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){46, 58, 89, 255});

    // 1) track: stretch the baked EDT texture to the centered map rect
    if (renderer.tex_ready){
        Rectangle src = {0, 0, (float)t->W, (float)t->H};
        Rectangle dst = {renderer.off_x, renderer.off_y, renderer.map_w_m * renderer.scale, renderer.map_h_m * renderer.scale};
        DrawTexturePro(renderer.track_tex, src, dst, (Vector2){0, 0}, 0.0f, WHITE);
    }

    // Pose trail: oldest faded, newest solid. Blue = PPO, amber = filter.
    if (renderer.trace_on){
        if (env->current_lap != renderer.trace_lap){
            trace_clear();
            renderer.trace_lap = env->current_lap;
        }
        if (renderer.trace_pending){
            trace_commit(renderer.pend_x, renderer.pend_y, renderer.pend_mode);
            renderer.trace_pending = 0;
        }
        int start = (renderer.trace_i - renderer.n_trace + MAX_TRACE) % MAX_TRACE;
        for (int k = 0; k < renderer.n_trace; k++){
            int i = (start + k) % MAX_TRACE;
            unsigned char a = (unsigned char)(70 + (185 * k) / (renderer.n_trace > 1 ? renderer.n_trace - 1 : 1));
            Color c;
            if (renderer.trace_mode[i] >= 2)      c = (Color){220, 70, 70, a};
            else if (renderer.trace_mode[i] == 1) c = (Color){255, 176, 64, a};
            else                           c = (Color){64, 156, 255, a};
            DrawCircleV(world_to_screen(t, renderer.trace_x[i], renderer.trace_y[i]), 3.4f, c);
        }
    }

    // Predicted horizon in front of the car: a green polyline with beads.
    if (renderer.n_plan > 1){
        for (int i = 0; i < renderer.n_plan - 1; i++){
            float tfrac = (float)i / (float)(renderer.n_plan - 1);
            unsigned char a = (unsigned char)(230 - (int)(110.0f * tfrac));
            Color c = (Color){72, 220, 128, a};
            Vector2 a_pt = world_to_screen(t, renderer.plan_x[i],     renderer.plan_y[i]);
            Vector2 b_pt = world_to_screen(t, renderer.plan_x[i + 1], renderer.plan_y[i + 1]);
            DrawLineEx(a_pt, b_pt, 2.6f, c);
        }
        for (int i = 0; i < renderer.n_plan; i++){
            float tfrac = (float)i / (float)(renderer.n_plan > 1 ? renderer.n_plan - 1 : 1);
            unsigned char a = (unsigned char)(240 - (int)(90.0f * tfrac));
            float r = (i == renderer.n_plan - 1) ? 3.6f : 2.2f;
            DrawCircleV(world_to_screen(t, renderer.plan_x[i], renderer.plan_y[i]),
                        r, (Color){72, 220, 128, a});
        }
    }

    // 2) car: rear wheels, body, front wheels steered by the plant delta
    //    (the command after delay + actuator). Spokes roll with speed.
    {
        float cyaw = cosf(env->state.yaw), syaw = sinf(env->state.yaw);
        float x = env->state.x, y = env->state.y;
        float delta = env->state.delta;
        float lf = env->chassis.lf, lr = env->chassis.lr;
        float hl = env->chassis.ego_l * 0.5f;
        float hw = env->chassis.ego_w * 0.5f;
        int period = env->time.last_control_period > 0
            ? env->time.last_control_period : env->time.control_period;
        renderer.wheel_spin += env->state.v * ((float)period * DT_SIM) / WHEEL_R;
        if (renderer.wheel_spin > 402.12389f) renderer.wheel_spin = fmodf(renderer.wheel_spin, 6.2831853f);

        float track = hw + 0.012f;
        float axle_x[4] = { +lf, +lf, -lr, -lr };   // FR, FL, RR, RL
        float axle_y[4] = { -track, +track, -track, +track };
        float wang[4] = {
            env->state.yaw + delta, env->state.yaw + delta,
            env->state.yaw,         env->state.yaw
        };
        float wx[4], wy[4];
        for (int i = 0; i < 4; i++){
            wx[i] = x + axle_x[i] * cyaw - axle_y[i] * syaw;
            wy[i] = y + axle_x[i] * syaw + axle_y[i] * cyaw;
        }

        // screen-space drop: body 3 px down as shadow, then the real car
        {
            Vector2 c[4];
            const float bx[4] = { +hl, +hl, -hl, -hl };
            const float by[4] = { -hw, +hw, +hw, -hw };
            for (int i = 0; i < 4; i++){
                c[i] = world_to_screen(t, x + bx[i]*cyaw - by[i]*syaw,
                                          y + bx[i]*syaw + by[i]*cyaw);
                c[i].y += 3.0f;
            }
            DrawTriangleFan(c, 4, (Color){0, 0, 0, 70});
        }

        // Rear wheels under the body, front wheels on top so steering reads.
        draw_wheel(t, wx[2], wy[2], wang[2], renderer.wheel_spin);
        draw_wheel(t, wx[3], wy[3], wang[3], renderer.wheel_spin);
        oriented_box(t, x, y, env->state.yaw, env->chassis.ego_l, env->chassis.ego_w,
                     (Color){227, 66, 52, 255}, (Color){90, 26, 20, 255}, 1.5f);
        draw_wheel(t, wx[0], wy[0], wang[0], renderer.wheel_spin);
        draw_wheel(t, wx[1], wy[1], wang[1], renderer.wheel_spin);
    }

    // Detector/tracker estimator boxes (LiDAR estimates, not occupancy).
    for (int i = 0; i < renderer.n_dets; i++){
        oriented_box(t, renderer.det_x[i], renderer.det_y[i], renderer.det_yaw[i],
                     renderer.det_l[i], renderer.det_w[i],
                     (Color){255, 40, 40, 50}, (Color){255, 48, 48, 255}, 2.4f);
    }

    // Dynamic obstacles (true pose, distinct colour from the
    // detector estimate). Drawn as a filled box with a velocity arrow.
    for (int i = 0; i < env->n_obstacles; i++){
        const DynamicObstacle *o = &env->obstacles[i];
        oriented_box(t, o->cx, o->cy, o->yaw,
                     2.0f * o->half_length, 2.0f * o->half_width,
                     (Color){40, 180, 80, 90}, (Color){30, 140, 60, 255}, 2.0f);
        float vx, vy;
        obstacle_world_velocity(o, &vx, &vy);
        Vector2 c = world_to_screen(t, o->cx, o->cy);
        Vector2 tip = world_to_screen(t, o->cx + vx * 0.5f, o->cy + vy * 0.5f);
        DrawLineEx(c, tip, 2.0f, (Color){30, 140, 60, 255});
    }

    // 3) TOP STRIP: lap (integer) + last lap (updates once) + latched safety.
    {
        if (env->current_lap > renderer.prev_lap && renderer.lap_hist_n < 8)
            renderer.lap_hist[renderer.lap_hist_n++] = env->last_lap_time;
        if (env->current_lap < renderer.prev_lap) {
            renderer.lap_hist_n = 0;
            renderer.wheel_spin = 0.0f;
            renderer.hud_avg_n = 0;
            renderer.hud_avg_i = 0;
            renderer.hud_avg_tick = -1;
        }
        renderer.prev_lap = env->current_lap;

        renderer.safety_hold -= GetFrameTime();
        if (renderer.safety_hold <= 0.0f) renderer.safety_mode = renderer.safety_raw;

        DrawRectangle(0, 0, renderer.win_w, TOP_H, (Color){22, 28, 42, 255});
        DrawLine(0, TOP_H, renderer.win_w, TOP_H, (Color){58, 70, 96, 255});

        Color mute = (Color){148, 162, 184, 255};
        Color ice  = (Color){236, 242, 250, 255};
        DrawText("LAP", 20, 8, 12, mute);
        DrawText(TextFormat("%d / %d", env->current_lap, env->lap_cap),
                 20, 22, 26, ice);

        DrawText("LAST", 160, 8, 12, mute);
        if (env->last_lap_time > 0.5f)
            DrawText(TextFormat("%.2f s", env->last_lap_time), 160, 22, 26, ice);
        else
            DrawText("--", 160, 22, 26, mute);

        const char* pill;
        Color pill_bg, pill_fg;
        if (renderer.safety_mode >= 2){
            pill = "UNSAFE"; pill_bg = (Color){150, 42, 48, 255}; pill_fg = (Color){255, 214, 214, 255};
        } else if (renderer.safety_mode == 1){
            pill = "FILTER"; pill_bg = (Color){168, 108, 28, 255}; pill_fg = (Color){255, 226, 160, 255};
        } else {
            pill = "CLEAR";  pill_bg = (Color){36, 110, 78, 255};  pill_fg = (Color){196, 240, 214, 255};
        }
        int pw = 132, ph = 32;
        int px = renderer.win_w - pw - 18, py = (TOP_H - ph) / 2;
        DrawRectangleRounded((Rectangle){(float)px, (float)py, (float)pw, (float)ph}, 0.5f, 6, pill_bg);
        int tw = MeasureText(pill, 20);
        DrawText(pill, px + (pw - tw) / 2, py + 6, 20, pill_fg);
    }

    // 4) BOTTOM STRIP: steer (centered, left/right) then speed.
    {
        int strip_top = renderer.win_h - HUD_H;
        DrawRectangle(0, strip_top, renderer.win_w, HUD_H, (Color){22, 28, 42, 255});
        DrawLine(0, strip_top, renderer.win_w, strip_top, (Color){58, 70, 96, 255});
        Color mute = (Color){148, 162, 184, 255};
        Color ice  = (Color){220, 228, 240, 255};
        Color track = (Color){36, 44, 62, 255};
        int bx = 90, bw = renderer.win_w - 110, bh = 10;

        DrawText("STEER", 20, strip_top + 8, 12, mute);
        int sty = strip_top + 10;
        DrawRectangleRounded((Rectangle){(float)bx, (float)sty, (float)bw, (float)bh},
                             0.5f, 4, track);
        float smax = env->limits.s_max > 0.05f ? env->limits.s_max : 0.42f;
        float sfrac = env->state.delta / smax;
        if (sfrac < -1.0f) sfrac = -1.0f;
        if (sfrac >  1.0f) sfrac =  1.0f;
        float vmax = env->limits.v_max > 0.1f ? env->limits.v_max : 8.0f;
        float frac = env->state.v / vmax;
        if (frac < 0.0f) frac = 0.0f;
        if (frac > 1.0f) frac = 1.0f;
        if (env->tick != renderer.hud_avg_tick){
            renderer.hud_steer_buf[renderer.hud_avg_i] = sfrac;
            renderer.hud_speed_buf[renderer.hud_avg_i] = frac;
            renderer.hud_avg_i = (renderer.hud_avg_i + 1) % HUD_AVG_TICKS;
            if (renderer.hud_avg_n < HUD_AVG_TICKS) renderer.hud_avg_n++;
            renderer.hud_avg_tick = env->tick;
        }
        float steer_sum = 0.0f, speed_sum = 0.0f;
        int n = renderer.hud_avg_n > 0 ? renderer.hud_avg_n : 1;
        for (int i = 0; i < renderer.hud_avg_n; i++){
            steer_sum += renderer.hud_steer_buf[i];
            speed_sum += renderer.hud_speed_buf[i];
        }
        sfrac = steer_sum / (float)n;
        frac  = speed_sum / (float)n;
        int mid = bx + bw / 2;
        int nx  = mid + (int)(sfrac * (bw * 0.5f));
        int fill_x = sfrac >= 0.0f ? mid : nx;
        int fill_w = sfrac >= 0.0f ? (nx - mid) : (mid - nx);
        if (fill_w < 2) fill_w = 2;
        DrawRectangle(fill_x, sty, fill_w, bh, ice);
        DrawRectangle(mid - 1, sty - 2, 2, bh + 4, (Color){90, 104, 128, 255});

        DrawText("SPEED", 20, strip_top + 32, 12, mute);
        int spy = strip_top + 34;
        DrawRectangleRounded((Rectangle){(float)bx, (float)spy, (float)bw, (float)bh},
                             0.5f, 4, track);
        if (frac > 0.01f)
            DrawRectangleRounded((Rectangle){(float)bx, (float)spy, bw * frac, (float)bh},
                                 0.5f, 4, ice);
    }

    EndDrawing();

#ifndef RAYLIB_STUB_H
    {
        const char* dir = getenv("ROBORACER_FRAME_DIR");
        if (dir && dir[0]){
            if ((renderer.frame_i % 5) == 0){
                TakeScreenshot(TextFormat("frame_%04d.png", renderer.frame_i));
            }
            renderer.frame_i++;
        } else if (!renderer.shot_done){
            const char* path = getenv("ROBORACER_SCREENSHOT");
            if (path && path[0]){
                TakeScreenshot(path);
                renderer.shot_done = 1;
            }
        }
    }
#endif
}


void renderer_close(void){
    if (renderer.tex_ready){
        UnloadTexture(renderer.track_tex);
        renderer.tex_ready = false;
    }
    if (IsWindowReady()) CloseWindow();
}

void renderer_invalidate_track(void){
    if (renderer.tex_ready){
        UnloadTexture(renderer.track_tex);
        renderer.tex_ready = false;
    }
}

int renderer_available(void){
#ifdef RAYLIB_STUB_H
    return 0;
#else
    return 1;
#endif
}

void renderer_set_detection_boxes(const float* data, int n){
    if (n < 0) n = 0;
    if (n > MAX_DET_BOXES) n = MAX_DET_BOXES;
    renderer.n_dets = n;
    for (int i = 0; i < n; i++){
        renderer.det_x[i] = data[i*5];
        renderer.det_y[i] = data[i*5 + 1];
        renderer.det_yaw[i] = data[i*5 + 2];
        renderer.det_l[i] = data[i*5 + 3];
        renderer.det_w[i] = data[i*5 + 4];
    }
}

void renderer_set_safety(int mode){
    if (mode < 0) mode = 0;
    if (mode > 2) mode = 2;
    renderer.safety_raw = mode;
    if (mode > renderer.safety_mode){
        renderer.safety_mode = mode;
        renderer.safety_hold = SAFETY_HOLD_S;
    } else if (mode == renderer.safety_mode && mode > 0){
        renderer.safety_hold = SAFETY_HOLD_S;
    }
}

void renderer_set_trace(int on){
    renderer.trace_on = on ? 1 : 0;
    if (!renderer.trace_on){
        trace_clear();
        renderer.trace_lap = -1;
    }
}

void renderer_push_trace(float x, float y, int mode){
    trace_push(x, y, mode);
}

void renderer_set_filter_path(const float* data, int n){
    if (n < 0) n = 0;
    if (n > MAX_PLAN) n = MAX_PLAN;
    renderer.n_plan = n;
    for (int i = 0; i < n; i++){
        renderer.plan_x[i] = data[i*2];
        renderer.plan_y[i] = data[i*2 + 1];
    }
}
