/*
 * lumen.h - Lumen C API Header
 *
 * High-performance C interface for Lumen: Real-time 3D Gaussian Splatting
 * and Radiance Field Rasterization Engine.
 *
 * Suitable for foreign function interface (FFI) bindings in Python, Rust,
 * C++, C#, and WebAssembly embeddings.
 */

#ifndef LUMEN_H
#define LUMEN_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

/* Version definitions */
#define LUMEN_VERSION_MAJOR 0
#define LUMEN_VERSION_MINOR 1
#define LUMEN_VERSION_PATCH 0
#define LUMEN_VERSION_STRING "0.1.0"

/* Return status codes */
typedef enum {
    LUMEN_SUCCESS = 0,
    LUMEN_ERROR_INVALID_ARGUMENT = -1,
    LUMEN_ERROR_OUT_OF_MEMORY = -2,
    LUMEN_ERROR_UNINITIALIZED = -3,
    LUMEN_ERROR_IO_FAILURE = -4,
    LUMEN_ERROR_MATH_SINGULAR = -5
} LumenStatus;

/* Engine configuration */
typedef struct {
    uint32_t image_width;
    uint32_t image_height;
    uint32_t tile_size;
    float near_plane;
    float far_plane;
    float early_termination_alpha;
    float gaussian_radius_sigma;
} LumenConfig;

/* 3D Vector */
typedef struct {
    float x;
    float y;
    float z;
} LumenVec3;

/* 4D Vector / Quaternion */
typedef struct {
    float w;
    float x;
    float y;
    float z;
} LumenQuat;

/* Pinhole camera descriptor */
typedef struct {
    LumenVec3 position;
    LumenVec3 target;
    LumenVec3 up;
    float fov_x_degrees;
    float fov_y_degrees;
    float near_plane;
    float far_plane;
} LumenCamera;

/* 3D Gaussian Splat representation */
typedef struct {
    LumenVec3 position;
    LumenVec3 scale;
    LumenQuat rotation;
    float opacity;
    float sh_coeffs[48]; /* Degree 3 spherical harmonics: 16 basis * 3 RGB */
} LumenGaussian;

/* Frame buffer target */
typedef struct {
    uint32_t width;
    uint32_t height;
    uint8_t* rgb_data;    /* Array of size width * height * 3 */
    float* depth_data;    /* Array of size width * height */
    float* alpha_data;    /* Array of size width * height */
} LumenFrameBuffer;

/* Rasterization execution statistics */
typedef struct {
    size_t total_splats;
    size_t visible_splats;
    size_t culled_splats;
    size_t total_tiles;
    size_t touched_tiles;
    float rasterize_time_ms;
    float frames_per_second;
} LumenStats;

/* Opaque engine types */
typedef struct LumenCloud LumenCloud;
typedef struct LumenRasterizer LumenRasterizer;

/* ========================================================================= */
/* Library Metadata                                                          */
/* ========================================================================= */

const char* lumen_get_version(void);
int lumen_get_version_major(void);
int lumen_get_version_minor(void);
int lumen_get_version_patch(void);

/* Default configuration helper */
LumenConfig lumen_default_config(void);

/* ========================================================================= */
/* Gaussian Cloud Management                                                 */
/* ========================================================================= */

LumenCloud* lumen_cloud_create(size_t initial_capacity);
void lumen_cloud_destroy(LumenCloud* cloud);
LumenStatus lumen_cloud_reserve(LumenCloud* cloud, size_t new_capacity);
size_t lumen_cloud_get_count(const LumenCloud* cloud);
size_t lumen_cloud_get_capacity(const LumenCloud* cloud);
LumenStatus lumen_cloud_add_gaussian(LumenCloud* cloud, const LumenGaussian* gaussian);
LumenStatus lumen_cloud_get_gaussian(const LumenCloud* cloud, size_t index, LumenGaussian* out_gaussian);
void lumen_cloud_clear(LumenCloud* cloud);

/* Synthetic scene generators */
LumenCloud* lumen_create_synthetic_cube(size_t points_per_axis, float cube_size);
LumenCloud* lumen_create_synthetic_sphere(size_t point_count, float radius);

/* ========================================================================= */
/* Frame Buffer Allocation & Management                                      */
/* ========================================================================= */

LumenFrameBuffer* lumen_framebuffer_create(uint32_t width, uint32_t height);
void lumen_framebuffer_destroy(LumenFrameBuffer* fb);
void lumen_framebuffer_clear(LumenFrameBuffer* fb, uint8_t r, uint8_t g, uint8_t b);

/* ========================================================================= */
/* Rasterization Pipeline                                                    */
/* ========================================================================= */

LumenRasterizer* lumen_rasterizer_create(const LumenConfig* config);
void lumen_rasterizer_destroy(LumenRasterizer* rasterizer);
LumenStatus lumen_rasterizer_set_config(LumenRasterizer* rasterizer, const LumenConfig* config);

LumenStatus lumen_rasterize(
    LumenRasterizer* rasterizer,
    const LumenCloud* cloud,
    const LumenCamera* camera,
    LumenFrameBuffer* out_framebuffer,
    LumenStats* out_stats
);

#ifdef __cplusplus
}
#endif

#endif /* LUMEN_H */
