#pragma once

typedef struct Roboracer Roboracer;

void renderer_render(Roboracer* env);
void renderer_close(void);
void renderer_invalidate_track(void);
int renderer_available(void);
void renderer_set_detection_boxes(const float* data, int n);
void renderer_set_safety(int mode);
void renderer_set_trace(int on);
void renderer_push_trace(float x, float y, int mode);
void renderer_set_filter_path(const float* data, int n);
