/*
 * lumen_c.c - Lumen C API Implementation
 *
 * High-performance C runtime implementation for Lumen 3D Gaussian Splatting
 * and Radiance Field Rasterization Engine.
 */

#include "lumen.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* 4x4 Row-major Matrix */
typedef struct {
    float m[16];
} Mat4x4;

/* Internal Gaussian Cloud Structure */
struct LumenCloud {
    LumenGaussian* data;
    size_t count;
    size_t capacity;
};

/* Internal Rasterizer Structure */
struct LumenRasterizer {
    LumenConfig config;
};

/* Internal 2D Projected Splat Representation */
typedef struct {
    float screen_x;
    float screen_y;
    float depth;
    float radius;
    float inv_cov_xx;
    float inv_cov_xy;
    float inv_cov_yy;
    float color_r;
    float color_g;
    float color_b;
    float opacity;
} ProjectedSplat;

/* Version functions */
const char* lumen_get_version(void) {
    return LUMEN_VERSION_STRING;
}

int lumen_get_version_major(void) {
    return LUMEN_VERSION_MAJOR;
}

int lumen_get_version_minor(void) {
    return LUMEN_VERSION_MINOR;
}

int lumen_get_version_patch(void) {
    return LUMEN_VERSION_PATCH;
}

LumenConfig lumen_default_config(void) {
    LumenConfig cfg;
    cfg.image_width = 800;
    cfg.image_height = 600;
    cfg.tile_size = 16;
    cfg.near_plane = 0.1f;
    cfg.far_plane = 100.0f;
    cfg.early_termination_alpha = 0.99f;
    cfg.gaussian_radius_sigma = 3.0f;
    return cfg;
}

/* Gaussian Cloud Implementation */
LumenCloud* lumen_cloud_create(size_t initial_capacity) {
    LumenCloud* cloud = (LumenCloud*)malloc(sizeof(LumenCloud));
    if (!cloud) return NULL;

    size_t cap = (initial_capacity > 0) ? initial_capacity : 16;
    cloud->data = (LumenGaussian*)malloc(cap * sizeof(LumenGaussian));
    if (!cloud->data) {
        free(cloud);
        return NULL;
    }

    cloud->count = 0;
    cloud->capacity = cap;
    return cloud;
}

void lumen_cloud_destroy(LumenCloud* cloud) {
    if (cloud) {
        if (cloud->data) {
            free(cloud->data);
            cloud->data = NULL;
        }
        free(cloud);
    }
}

LumenStatus lumen_cloud_reserve(LumenCloud* cloud, size_t new_capacity) {
    if (!cloud) return LUMEN_ERROR_INVALID_ARGUMENT;
    if (new_capacity <= cloud->capacity) return LUMEN_SUCCESS;

    LumenGaussian* new_data = (LumenGaussian*)realloc(cloud->data, new_capacity * sizeof(LumenGaussian));
    if (!new_data) return LUMEN_ERROR_OUT_OF_MEMORY;

    cloud->data = new_data;
    cloud->capacity = new_capacity;
    return LUMEN_SUCCESS;
}

size_t lumen_cloud_get_count(const LumenCloud* cloud) {
    return cloud ? cloud->count : 0;
}

size_t lumen_cloud_get_capacity(const LumenCloud* cloud) {
    return cloud ? cloud->capacity : 0;
}

LumenStatus lumen_cloud_add_gaussian(LumenCloud* cloud, const LumenGaussian* gaussian) {
    if (!cloud || !gaussian) return LUMEN_ERROR_INVALID_ARGUMENT;

    if (cloud->count >= cloud->capacity) {
        size_t next_cap = cloud->capacity * 2;
        if (next_cap < 16) next_cap = 16;
        LumenStatus status = lumen_cloud_reserve(cloud, next_cap);
        if (status != LUMEN_SUCCESS) return status;
    }

    cloud->data[cloud->count] = *gaussian;
    cloud->count++;
    return LUMEN_SUCCESS;
}

LumenStatus lumen_cloud_get_gaussian(const LumenCloud* cloud, size_t index, LumenGaussian* out_gaussian) {
    if (!cloud || !out_gaussian) return LUMEN_ERROR_INVALID_ARGUMENT;
    if (index >= cloud->count) return LUMEN_ERROR_INVALID_ARGUMENT;

    *out_gaussian = cloud->data[index];
    return LUMEN_SUCCESS;
}

void lumen_cloud_clear(LumenCloud* cloud) {
    if (cloud) {
        cloud->count = 0;
    }
}

/* Synthetic Scene Generators */
LumenCloud* lumen_create_synthetic_cube(size_t points_per_axis, float cube_size) {
    if (points_per_axis < 1) points_per_axis = 1;
    size_t total = points_per_axis * points_per_axis * points_per_axis;
    LumenCloud* cloud = lumen_cloud_create(total);
    if (!cloud) return NULL;

    float step = (points_per_axis > 1) ? (cube_size / (float)(points_per_axis - 1)) : 0.0f;
    float half = cube_size * 0.5f;
    float splat_scale = step * 0.6f;
    if (splat_scale < 0.01f) splat_scale = 0.05f;

    for (size_t x = 0; x < points_per_axis; ++x) {
        float px = -half + (float)x * step;
        for (size_t y = 0; y < points_per_axis; ++y) {
            float py = -half + (float)y * step;
            for (size_t z = 0; z < points_per_axis; ++z) {
                float pz = -half + (float)z * step;

                LumenGaussian g;
                memset(&g, 0, sizeof(LumenGaussian));
                g.position.x = px;
                g.position.y = py;
                g.position.z = pz;

                g.scale.x = splat_scale;
                g.scale.y = splat_scale;
                g.scale.z = splat_scale;

                g.rotation.w = 1.0f;
                g.rotation.x = 0.0f;
                g.rotation.y = 0.0f;
                g.rotation.z = 0.0f;

                g.opacity = 0.85f;

                /* RGB mapped to coordinate cube */
                float norm_r = (px + half) / cube_size;
                float norm_g = (py + half) / cube_size;
                float norm_b = (pz + half) / cube_size;

                /* Zero degree spherical harmonic coefficient is C0 = 0.2820947917 */
                const float c0 = 0.2820947917f;
                g.sh_coeffs[0] = (norm_r - 0.5f) / c0;
                g.sh_coeffs[1] = (norm_g - 0.5f) / c0;
                g.sh_coeffs[2] = (norm_b - 0.5f) / c0;

                lumen_cloud_add_gaussian(cloud, &g);
            }
        }
    }

    return cloud;
}

LumenCloud* lumen_create_synthetic_sphere(size_t point_count, float radius) {
    if (point_count < 1) point_count = 1;
    LumenCloud* cloud = lumen_cloud_create(point_count);
    if (!cloud) return NULL;

    const float golden_ratio = 1.6180339887f;
    float splat_scale = radius * 0.12f;

    for (size_t i = 0; i < point_count; ++i) {
        float theta = 2.0f * (float)M_PI * (float)i / golden_ratio;
        float phi = acosf(1.0f - 2.0f * ((float)i + 0.5f) / (float)point_count);

        float px = radius * sinf(phi) * cosf(theta);
        float py = radius * sinf(phi) * sinf(theta);
        float pz = radius * cosf(phi);

        LumenGaussian g;
        memset(&g, 0, sizeof(LumenGaussian));
        g.position.x = px;
        g.position.y = py;
        g.position.z = pz;

        g.scale.x = splat_scale;
        g.scale.y = splat_scale;
        g.scale.z = splat_scale;

        g.rotation.w = 1.0f;
        g.rotation.x = 0.0f;
        g.rotation.y = 0.0f;
        g.rotation.z = 0.0f;

        g.opacity = 0.90f;

        float norm_r = 0.5f + 0.5f * (px / radius);
        float norm_g = 0.5f + 0.5f * (py / radius);
        float norm_b = 0.5f + 0.5f * (pz / radius);

        const float c0 = 0.2820947917f;
        g.sh_coeffs[0] = (norm_r - 0.5f) / c0;
        g.sh_coeffs[1] = (norm_g - 0.5f) / c0;
        g.sh_coeffs[2] = (norm_b - 0.5f) / c0;

        lumen_cloud_add_gaussian(cloud, &g);
    }

    return cloud;
}

/* Frame Buffer Implementation */
LumenFrameBuffer* lumen_framebuffer_create(uint32_t width, uint32_t height) {
    if (width == 0 || height == 0) return NULL;

    LumenFrameBuffer* fb = (LumenFrameBuffer*)malloc(sizeof(LumenFrameBuffer));
    if (!fb) return NULL;

    fb->width = width;
    fb->height = height;
    size_t pixel_count = (size_t)width * (size_t)height;

    fb->rgb_data = (uint8_t*)malloc(pixel_count * 3 * sizeof(uint8_t));
    fb->depth_data = (float*)malloc(pixel_count * sizeof(float));
    fb->alpha_data = (float*)malloc(pixel_count * sizeof(float));

    if (!fb->rgb_data || !fb->depth_data || !fb->alpha_data) {
        lumen_framebuffer_destroy(fb);
        return NULL;
    }

    lumen_framebuffer_clear(fb, 0, 0, 0);
    return fb;
}

void lumen_framebuffer_destroy(LumenFrameBuffer* fb) {
    if (fb) {
        if (fb->rgb_data) free(fb->rgb_data);
        if (fb->depth_data) free(fb->depth_data);
        if (fb->alpha_data) free(fb->alpha_data);
        free(fb);
    }
}

void lumen_framebuffer_clear(LumenFrameBuffer* fb, uint8_t r, uint8_t g, uint8_t b) {
    if (!fb) return;
    size_t pixel_count = (size_t)fb->width * (size_t)fb->height;

    for (size_t i = 0; i < pixel_count; ++i) {
        fb->rgb_data[i * 3 + 0] = r;
        fb->rgb_data[i * 3 + 1] = g;
        fb->rgb_data[i * 3 + 2] = b;
        fb->depth_data[i] = 1000.0f;
        fb->alpha_data[i] = 0.0f;
    }
}

/* Rasterizer Implementation */
LumenRasterizer* lumen_rasterizer_create(const LumenConfig* config) {
    LumenRasterizer* r = (LumenRasterizer*)malloc(sizeof(LumenRasterizer));
    if (!r) return NULL;

    if (config) {
        r->config = *config;
    } else {
        r->config = lumen_default_config();
    }
    return r;
}

void lumen_rasterizer_destroy(LumenRasterizer* rasterizer) {
    if (rasterizer) {
        free(rasterizer);
    }
}

LumenStatus lumen_rasterizer_set_config(LumenRasterizer* rasterizer, const LumenConfig* config) {
    if (!rasterizer || !config) return LUMEN_ERROR_INVALID_ARGUMENT;
    rasterizer->config = *config;
    return LUMEN_SUCCESS;
}

/* Math Utilities for Rasterization */
static inline float vec3_dot(LumenVec3 a, LumenVec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline LumenVec3 vec3_cross(LumenVec3 a, LumenVec3 b) {
    LumenVec3 res;
    res.x = a.y * b.z - a.z * b.y;
    res.y = a.z * b.x - a.x * b.z;
    res.z = a.x * b.y - a.y * b.x;
    return res;
}

static inline LumenVec3 vec3_normalize(LumenVec3 v) {
    float len = sqrtf(vec3_dot(v, v));
    if (len < 1e-6f) return v;
    float inv_len = 1.0f / len;
    LumenVec3 res;
    res.x = v.x * inv_len;
    res.y = v.y * inv_len;
    res.z = v.z * inv_len;
    return res;
}

static inline Mat4x4 look_at(LumenVec3 eye, LumenVec3 target, LumenVec3 up) {
    LumenVec3 f;
    f.x = target.x - eye.x;
    f.y = target.y - eye.y;
    f.z = target.z - eye.z;
    f = vec3_normalize(f);

    LumenVec3 r = vec3_normalize(vec3_cross(f, up));
    LumenVec3 u = vec3_cross(r, f);

    Mat4x4 m;
    m.m[0]  = r.x;  m.m[1]  = r.y;  m.m[2]  = r.z;  m.m[3]  = -vec3_dot(r, eye);
    m.m[4]  = u.x;  m.m[5]  = u.y;  m.m[6]  = u.z;  m.m[7]  = -vec3_dot(u, eye);
    m.m[8]  = -f.x; m.m[9]  = -f.y; m.m[10] = -f.z; m.m[11] = vec3_dot(f, eye);
    m.m[12] = 0.0f; m.m[13] = 0.0f; m.m[14] = 0.0f; m.m[15] = 1.0f;
    return m;
}

static inline int compare_splats_depth(const void* a, const void* b) {
    const ProjectedSplat* sa = (const ProjectedSplat*)a;
    const ProjectedSplat* sb = (const ProjectedSplat*)b;
    if (sa->depth < sb->depth) return -1;
    if (sa->depth > sb->depth) return 1;
    return 0;
}

/* Rasterization Core */
LumenStatus lumen_rasterize(
    LumenRasterizer* rasterizer,
    const LumenCloud* cloud,
    const LumenCamera* camera,
    LumenFrameBuffer* out_fb,
    LumenStats* out_stats
) {
    if (!rasterizer || !cloud || !camera || !out_fb) return LUMEN_ERROR_INVALID_ARGUMENT;

    clock_t start_time = clock();

    uint32_t width = out_fb->width;
    uint32_t height = out_fb->height;
    float fov_y_rad = camera->fov_y_degrees * (float)M_PI / 180.0f;
    float focal_y = (float)height / (2.0f * tanf(fov_y_rad * 0.5f));
    float focal_x = focal_y;

    Mat4x4 view = look_at(camera->position, camera->target, camera->up);

    /* Allocate buffer for projected splats */
    size_t count = cloud->count;
    ProjectedSplat* projected = (ProjectedSplat*)malloc(count * sizeof(ProjectedSplat));
    if (!projected && count > 0) return LUMEN_ERROR_OUT_OF_MEMORY;

    size_t visible_count = 0;
    size_t culled_count = 0;

    for (size_t i = 0; i < count; ++i) {
        const LumenGaussian* g = &cloud->data[i];

        /* World to view space transform */
        float vx = view.m[0] * g->position.x + view.m[1] * g->position.y + view.m[2] * g->position.z + view.m[3];
        float vy = view.m[4] * g->position.x + view.m[5] * g->position.y + view.m[6] * g->position.z + view.m[7];
        float vz = -(view.m[8] * g->position.x + view.m[9] * g->position.y + view.m[10] * g->position.z + view.m[11]);

        /* Near and far plane culling */
        if (vz <= rasterizer->config.near_plane || vz >= rasterizer->config.far_plane) {
            culled_count++;
            continue;
        }

        /* Perspective projection */
        float screen_x = (vx / vz) * focal_x + (float)width * 0.5f;
        float screen_y = (vy / vz) * focal_y + (float)height * 0.5f;

        /* Approximate 2D footprint radius based on 3D scale and distance */
        float max_scale = g->scale.x;
        if (g->scale.y > max_scale) max_scale = g->scale.y;
        if (g->scale.z > max_scale) max_scale = g->scale.z;

        float radius = (max_scale * focal_y / vz) * rasterizer->config.gaussian_radius_sigma;
        if (radius < 1.0f) radius = 1.0f;

        /* Screen boundary culling */
        if (screen_x + radius < 0.0f || screen_x - radius >= (float)width ||
            screen_y + radius < 0.0f || screen_y - radius >= (float)height) {
            culled_count++;
            continue;
        }

        /* Compute 2D Gaussian inverse covariance */
        float var = (radius / rasterizer->config.gaussian_radius_sigma) * (radius / rasterizer->config.gaussian_radius_sigma);
        if (var < 0.5f) var = 0.5f;
        float inv_var = 1.0f / var;

        /* Decode zeroth degree SH color */
        const float c0 = 0.2820947917f;
        float cr = 0.5f + c0 * g->sh_coeffs[0];
        float cg = 0.5f + c0 * g->sh_coeffs[1];
        float cb = 0.5f + c0 * g->sh_coeffs[2];

        if (cr < 0.0f) cr = 0.0f; if (cr > 1.0f) cr = 1.0f;
        if (cg < 0.0f) cg = 0.0f; if (cg > 1.0f) cg = 1.0f;
        if (cb < 0.0f) cb = 0.0f; if (cb > 1.0f) cb = 1.0f;

        projected[visible_count].screen_x = screen_x;
        projected[visible_count].screen_y = screen_y;
        projected[visible_count].depth = vz;
        projected[visible_count].radius = radius;
        projected[visible_count].inv_cov_xx = inv_var;
        projected[visible_count].inv_cov_xy = 0.0f;
        projected[visible_count].inv_cov_yy = inv_var;
        projected[visible_count].color_r = cr;
        projected[visible_count].color_g = cg;
        projected[visible_count].color_b = cb;
        projected[visible_count].opacity = g->opacity;
        visible_count++;
    }

    /* Sort splats front-to-back by depth */
    if (visible_count > 1) {
        qsort(projected, visible_count, sizeof(ProjectedSplat), compare_splats_depth);
    }

    /* Rasterize onto framebuffer tiles */
    uint32_t tile_size = rasterizer->config.tile_size;
    uint32_t tiles_x = (width + tile_size - 1) / tile_size;
    uint32_t tiles_y = (height + tile_size - 1) / tile_size;
    size_t touched_tiles = 0;

    for (uint32_t ty = 0; ty < tiles_y; ++ty) {
        uint32_t y_start = ty * tile_size;
        uint32_t y_end = y_start + tile_size;
        if (y_end > height) y_end = height;

        for (uint32_t tx = 0; tx < tiles_x; ++tx) {
            uint32_t x_start = tx * tile_size;
            uint32_t x_end = x_start + tile_size;
            if (x_end > width) x_end = width;

            bool tile_touched = false;

            /* Iterate through splats intersecting this tile */
            for (size_t s = 0; s < visible_count; ++s) {
                const ProjectedSplat* p = &projected[s];

                /* Quick bounding box test against tile */
                if (p->screen_x + p->radius < (float)x_start || p->screen_x - p->radius >= (float)x_end ||
                    p->screen_y + p->radius < (float)y_start || p->screen_y - p->radius >= (float)y_end) {
                    continue;
                }

                tile_touched = true;

                /* Pixel evaluation inside tile */
                for (uint32_t py = y_start; py < y_end; ++py) {
                    float dy = (float)py - p->screen_y;
                    size_t row_idx = (size_t)py * (size_t)width;

                    for (uint32_t px = x_start; px < x_end; ++px) {
                        size_t pidx = row_idx + px;
                        float cur_alpha = out_fb->alpha_data[pidx];

                        /* Early ray termination */
                        if (cur_alpha >= rasterizer->config.early_termination_alpha) {
                            continue;
                        }

                        float dx = (float)px - p->screen_x;
                        float dist_sq = dx * dx + dy * dy;
                        if (dist_sq > p->radius * p->radius) continue;

                        float power = -0.5f * (p->inv_cov_xx * dx * dx + p->inv_cov_yy * dy * dy);
                        if (power > 0.0f) power = 0.0f;
                        float gaussian_weight = expf(power);
                        float alpha = p->opacity * gaussian_weight;

                        if (alpha < (1.0f / 255.0f)) continue;

                        float trans = 1.0f - cur_alpha;
                        float weight = alpha * trans;

                        /* Accumulate color */
                        float r_accum = (float)out_fb->rgb_data[pidx * 3 + 0] / 255.0f + weight * p->color_r;
                        float g_accum = (float)out_fb->rgb_data[pidx * 3 + 1] / 255.0f + weight * p->color_g;
                        float b_accum = (float)out_fb->rgb_data[pidx * 3 + 2] / 255.0f + weight * p->color_b;

                        if (r_accum > 1.0f) r_accum = 1.0f;
                        if (g_accum > 1.0f) g_accum = 1.0f;
                        if (b_accum > 1.0f) b_accum = 1.0f;

                        out_fb->rgb_data[pidx * 3 + 0] = (uint8_t)(r_accum * 255.0f);
                        out_fb->rgb_data[pidx * 3 + 1] = (uint8_t)(g_accum * 255.0f);
                        out_fb->rgb_data[pidx * 3 + 2] = (uint8_t)(b_accum * 255.0f);

                        /* Update depth if closest contribution */
                        if (cur_alpha < 0.5f && p->depth < out_fb->depth_data[pidx]) {
                            out_fb->depth_data[pidx] = p->depth;
                        }

                        out_fb->alpha_data[pidx] = cur_alpha + weight;
                    }
                }
            }

            if (tile_touched) touched_tiles++;
        }
    }

    if (projected) free(projected);

    clock_t end_time = clock();
    float elapsed_ms = (float)(end_time - start_time) * 1000.0f / (float)CLOCKS_PER_SEC;

    if (out_stats) {
        out_stats->total_splats = count;
        out_stats->visible_splats = visible_count;
        out_stats->culled_splats = culled_count;
        out_stats->total_tiles = (size_t)tiles_x * (size_t)tiles_y;
        out_stats->touched_tiles = touched_tiles;
        out_stats->rasterize_time_ms = elapsed_ms;
        out_stats->frames_per_second = (elapsed_ms > 0.001f) ? (1000.0f / elapsed_ms) : 10000.0f;
    }

    return LUMEN_SUCCESS;
}
