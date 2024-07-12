/*
 * test_c_api.c - Lumen C API Test Suite
 *
 * Comprehensive unit and integration verification for the Lumen C interface.
 */

#include "lumen.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

static int g_tests_run = 0;
static int g_tests_passed = 0;

#define TEST_ASSERT(cond, msg) do { \
    g_tests_run++; \
    if (!(cond)) { \
        printf("[FAIL] %s:%d: %s\n", __FILE__, __LINE__, msg); \
        exit(1); \
    } else { \
        g_tests_passed++; \
    } \
} while (0)

void test_version(void) {
    const char* ver = lumen_get_version();
    TEST_ASSERT(ver != NULL, "Version string must not be null");
    TEST_ASSERT(lumen_get_version_major() == 0, "Major version should be 0");
    TEST_ASSERT(lumen_get_version_minor() == 1, "Minor version should be 1");
    TEST_ASSERT(lumen_get_version_patch() == 0, "Patch version should be 0");
    printf("[PASS] test_version\n");
}

void test_config(void) {
    LumenConfig cfg = lumen_default_config();
    TEST_ASSERT(cfg.image_width == 800, "Default width should be 800");
    TEST_ASSERT(cfg.image_height == 600, "Default height should be 600");
    TEST_ASSERT(cfg.tile_size == 16, "Default tile size should be 16");
    TEST_ASSERT(cfg.near_plane > 0.0f, "Near plane must be positive");
    TEST_ASSERT(cfg.far_plane > cfg.near_plane, "Far plane must exceed near plane");
    TEST_ASSERT(cfg.early_termination_alpha > 0.9f, "Early termination alpha should be near 1.0");
    printf("[PASS] test_config\n");
}

void test_cloud_lifecycle(void) {
    LumenCloud* cloud = lumen_cloud_create(8);
    TEST_ASSERT(cloud != NULL, "Cloud allocation failed");
    TEST_ASSERT(lumen_cloud_get_count(cloud) == 0, "Initial count must be zero");
    TEST_ASSERT(lumen_cloud_get_capacity(cloud) >= 8, "Initial capacity must be at least 8");

    LumenGaussian g;
    g.position.x = 1.0f; g.position.y = 2.0f; g.position.z = 3.0f;
    g.scale.x = 0.1f; g.scale.y = 0.1f; g.scale.z = 0.1f;
    g.rotation.w = 1.0f; g.rotation.x = 0.0f; g.rotation.y = 0.0f; g.rotation.z = 0.0f;
    g.opacity = 0.9f;
    for (int i = 0; i < 48; ++i) g.sh_coeffs[i] = 0.0f;

    LumenStatus status = lumen_cloud_add_gaussian(cloud, &g);
    TEST_ASSERT(status == LUMEN_SUCCESS, "Add gaussian failed");
    TEST_ASSERT(lumen_cloud_get_count(cloud) == 1, "Count should be 1");

    LumenGaussian retrieved;
    status = lumen_cloud_get_gaussian(cloud, 0, &retrieved);
    TEST_ASSERT(status == LUMEN_SUCCESS, "Get gaussian failed");
    TEST_ASSERT(fabsf(retrieved.position.x - 1.0f) < 1e-5f, "Position X mismatch");
    TEST_ASSERT(fabsf(retrieved.position.y - 2.0f) < 1e-5f, "Position Y mismatch");
    TEST_ASSERT(fabsf(retrieved.position.z - 3.0f) < 1e-5f, "Position Z mismatch");

    /* Add more to test dynamic reallocation */
    for (int i = 0; i < 30; ++i) {
        g.position.x = (float)i;
        lumen_cloud_add_gaussian(cloud, &g);
    }
    TEST_ASSERT(lumen_cloud_get_count(cloud) == 31, "Count should be 31");
    TEST_ASSERT(lumen_cloud_get_capacity(cloud) >= 31, "Capacity should have expanded");

    lumen_cloud_clear(cloud);
    TEST_ASSERT(lumen_cloud_get_count(cloud) == 0, "Count after clear must be zero");

    lumen_cloud_destroy(cloud);
    printf("[PASS] test_cloud_lifecycle\n");
}

void test_synthetic_scenes(void) {
    /* Test synthetic cube: 4x4x4 = 64 points */
    LumenCloud* cube = lumen_create_synthetic_cube(4, 2.0f);
    TEST_ASSERT(cube != NULL, "Cube creation failed");
    TEST_ASSERT(lumen_cloud_get_count(cube) == 64, "Cube must have exactly 64 splats");

    LumenGaussian first_g;
    lumen_cloud_get_gaussian(cube, 0, &first_g);
    TEST_ASSERT(first_g.opacity > 0.0f, "Opacity should be positive");
    TEST_ASSERT(first_g.scale.x > 0.0f, "Scale should be positive");
    lumen_cloud_destroy(cube);

    /* Test synthetic sphere: 128 points */
    LumenCloud* sphere = lumen_create_synthetic_sphere(128, 1.5f);
    TEST_ASSERT(sphere != NULL, "Sphere creation failed");
    TEST_ASSERT(lumen_cloud_get_count(sphere) == 128, "Sphere must have 128 splats");
    lumen_cloud_destroy(sphere);

    printf("[PASS] test_synthetic_scenes\n");
}

void test_framebuffer(void) {
    LumenFrameBuffer* fb = lumen_framebuffer_create(128, 96);
    TEST_ASSERT(fb != NULL, "Framebuffer creation failed");
    TEST_ASSERT(fb->width == 128, "Width mismatch");
    TEST_ASSERT(fb->height == 96, "Height mismatch");
    TEST_ASSERT(fb->rgb_data != NULL, "RGB buffer null");
    TEST_ASSERT(fb->depth_data != NULL, "Depth buffer null");
    TEST_ASSERT(fb->alpha_data != NULL, "Alpha buffer null");

    /* Clear with specific color */
    lumen_framebuffer_clear(fb, 42, 84, 126);
    TEST_ASSERT(fb->rgb_data[0] == 42, "RGB[0] mismatch");
    TEST_ASSERT(fb->rgb_data[1] == 84, "RGB[1] mismatch");
    TEST_ASSERT(fb->rgb_data[2] == 126, "RGB[2] mismatch");
    TEST_ASSERT(fb->alpha_data[0] == 0.0f, "Alpha[0] should be 0");
    TEST_ASSERT(fb->depth_data[0] == 1000.0f, "Depth[0] should be 1000");

    lumen_framebuffer_destroy(fb);
    printf("[PASS] test_framebuffer\n");
}

void test_rasterization_pipeline(void) {
    LumenConfig cfg;
    cfg.image_width = 160;
    cfg.image_height = 120;
    cfg.tile_size = 16;
    cfg.near_plane = 0.1f;
    cfg.far_plane = 50.0f;
    cfg.early_termination_alpha = 0.99f;
    cfg.gaussian_radius_sigma = 3.0f;

    LumenRasterizer* r = lumen_rasterizer_create(&cfg);
    TEST_ASSERT(r != NULL, "Rasterizer creation failed");

    LumenCloud* cube = lumen_create_synthetic_cube(4, 1.0f); /* 64 splats */
    TEST_ASSERT(cube != NULL, "Cube creation failed");

    LumenCamera cam;
    cam.position.x = 0.0f; cam.position.y = 0.0f; cam.position.z = 3.0f;
    cam.target.x = 0.0f; cam.target.y = 0.0f; cam.target.z = 0.0f;
    cam.up.x = 0.0f; cam.up.y = 1.0f; cam.up.z = 0.0f;
    cam.fov_x_degrees = 60.0f;
    cam.fov_y_degrees = 45.0f;
    cam.near_plane = 0.1f;
    cam.far_plane = 50.0f;

    LumenFrameBuffer* fb = lumen_framebuffer_create(cfg.image_width, cfg.image_height);
    TEST_ASSERT(fb != NULL, "FB creation failed");

    LumenStats stats;
    LumenStatus status = lumen_rasterize(r, cube, &cam, fb, &stats);
    TEST_ASSERT(status == LUMEN_SUCCESS, "Rasterization failed");
    TEST_ASSERT(stats.total_splats == 64, "Total splats should be 64");
    TEST_ASSERT(stats.visible_splats > 0, "Visible splats should be greater than 0");
    TEST_ASSERT(stats.touched_tiles > 0, "Touched tiles should be greater than 0");
    TEST_ASSERT(stats.rasterize_time_ms >= 0.0f, "Rasterize time should be non-negative");

    /* Verify center pixel has received rendered radiance */
    size_t center_idx = (size_t)(cfg.image_height / 2) * (size_t)cfg.image_width + (size_t)(cfg.image_width / 2);
    float center_alpha = fb->alpha_data[center_idx];
    TEST_ASSERT(center_alpha > 0.0f, "Center alpha should be non-zero after rasterization");

    lumen_framebuffer_destroy(fb);
    lumen_cloud_destroy(cube);
    lumen_rasterizer_destroy(r);
    printf("[PASS] test_rasterization_pipeline\n");
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;

    printf("============================================================\n");
    printf(" Lumen C API Test Suite\n");
    printf("============================================================\n");

    test_version();
    test_config();
    test_cloud_lifecycle();
    test_synthetic_scenes();
    test_framebuffer();
    test_rasterization_pipeline();

    printf("============================================================\n");
    printf(" Results: %d / %d assertions passed. 100%% SUCCESS\n", g_tests_passed, g_tests_run);
    printf("============================================================\n");
    return 0;
}
