/* No-op raylib so the plant compiles without a display.
 * IsWindowReady() returns false so the render setup path is never taken.
 */
#ifndef RAYLIB_STUB_H
#define RAYLIB_STUB_H

#include <stdbool.h>

typedef struct Color { unsigned char r, g, b, a; } Color;
typedef struct Vector2 { float x, y; } Vector2;
typedef struct Rectangle { float x, y, width, height; } Rectangle;
typedef struct Image { void *data; int width, height, mipmaps, format; } Image;
typedef struct Texture2D { unsigned int id; int width, height, mipmaps, format; } Texture2D;

#define KEY_ESCAPE     256
#define KEY_LEFT_SHIFT 340
#define KEY_SPACE       32
#define KEY_LEFT       263
#define KEY_RIGHT      262
#define KEY_UP         265
#define KEY_DOWN       264
#define KEY_A           65
#define KEY_D           68
#define KEY_W           87
#define KEY_S           83
#define MOUSE_BUTTON_LEFT 0

#define PIXELFORMAT_UNCOMPRESSED_R8G8B8A8 7
#define TEXTURE_FILTER_POINT 0

static const Color RAYWHITE = {245, 245, 245, 255};
static const Color WHITE    = {255, 255, 255, 255};
static const Color RED      = {230,  41,  55, 255};

static inline bool IsWindowReady(void) { return false; }
static inline void InitWindow(int w, int h, const char *t) { (void)w; (void)h; (void)t; }
static inline void SetTargetFPS(int f) { (void)f; }
static inline bool IsKeyDown(int k) { (void)k; return false; }
static inline bool WindowShouldClose(void) { return true; }
static inline float GetFrameTime(void) { return 0.0f; }
static inline float GetMouseWheelMove(void) { return 0.0f; }
static inline Vector2 GetMousePosition(void) { return (Vector2){0}; }
static inline bool IsMouseButtonPressed(int b) { (void)b; return false; }
static inline bool IsMouseButtonReleased(int b) { (void)b; return false; }
static inline bool IsMouseButtonDown(int b) { (void)b; return false; }
static inline void BeginDrawing(void) {}
static inline void ClearBackground(Color c) { (void)c; }
static inline void EndDrawing(void) {}
static inline void CloseWindow(void) {}

static inline void DrawRectangle(int x, int y, int w, int h, Color c) { (void)x; (void)y; (void)w; (void)h; (void)c; }
static inline void DrawRectangleLines(int x, int y, int w, int h, Color c) { (void)x; (void)y; (void)w; (void)h; (void)c; }
static inline void DrawRectangleRounded(Rectangle r, float roundness, int segments, Color c) {
    (void)r; (void)roundness; (void)segments; (void)c;
}
static inline void DrawText(const char *s, int x, int y, int sz, Color c) { (void)s; (void)x; (void)y; (void)sz; (void)c; }
static inline void DrawFPS(int x, int y) { (void)x; (void)y; }
static inline void DrawCircleV(Vector2 p, float r, Color c) { (void)p; (void)r; (void)c; }
static inline void DrawLineV(Vector2 a, Vector2 b, Color c) { (void)a; (void)b; (void)c; }
static inline void DrawLineEx(Vector2 a, Vector2 b, float th, Color c) { (void)a; (void)b; (void)th; (void)c; }
static inline void DrawTexturePro(Texture2D t, Rectangle s, Rectangle d, Vector2 o, float rot, Color c) {
    (void)t; (void)s; (void)d; (void)o; (void)rot; (void)c;
}
static inline Texture2D LoadTextureFromImage(Image img) { (void)img; Texture2D t = {0}; return t; }
static inline void SetTextureFilter(Texture2D t, int filter) { (void)t; (void)filter; }
static inline void UnloadTexture(Texture2D t) { (void)t; }
static inline void DrawTriangleFan(const Vector2 *pts, int n, Color c) { (void)pts; (void)n; (void)c; }
static inline void DrawLine(int x1, int y1, int x2, int y2, Color c) { (void)x1; (void)y1; (void)x2; (void)y2; (void)c; }
static inline int MeasureText(const char *s, int sz) { (void)s; (void)sz; return 0; }

static inline const char *TextFormat(const char *fmt, ...) { (void)fmt; return ""; }

#endif // RAYLIB_STUB_H
