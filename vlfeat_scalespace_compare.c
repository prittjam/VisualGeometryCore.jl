/**
 * VLFeat Scale Space Comparison Tool
 *
 * This program builds a scale space using VLFeat and saves all levels
 * to TIFF files for comparison with our Julia implementation.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <tiffio.h>
#include <vl/scalespace.h>
#include <vl/covdet.h>
#include <vl/imopv.h>
#include <vl/generic.h>

/* Save float array to TIFF file */
int save_float_tiff(const char *filename, const float *data, int width, int height) {
    TIFF *tif = TIFFOpen(filename, "w");
    if (!tif) {
        fprintf(stderr, "Could not open %s for writing\n", filename);
        return -1;
    }

    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH, width);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH, height);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE, 32);
    TIFFSetField(tif, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
    TIFFSetField(tif, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP, 1);

    /* Write data row by row */
    for (int row = 0; row < height; row++) {
        if (TIFFWriteScanline(tif, (void *)(data + row * width), row, 0) < 0) {
            fprintf(stderr, "Error writing scanline %d\n", row);
            TIFFClose(tif);
            return -1;
        }
    }

    TIFFClose(tif);
    return 0;
}

/* Load float image from TIFF file */
float *load_float_tiff(const char *filename, int *width, int *height) {
    TIFF *tif = TIFFOpen(filename, "r");
    if (!tif) {
        fprintf(stderr, "Could not open %s for reading\n", filename);
        return NULL;
    }

    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, height);

    float *data = (float *)malloc((*width) * (*height) * sizeof(float));
    if (!data) {
        fprintf(stderr, "Could not allocate memory for image\n");
        TIFFClose(tif);
        return NULL;
    }

    /* Read data row by row */
    for (int row = 0; row < *height; row++) {
        if (TIFFReadScanline(tif, data + row * (*width), row, 0) < 0) {
            fprintf(stderr, "Error reading scanline %d\n", row);
            free(data);
            TIFFClose(tif);
            return NULL;
        }
    }

    TIFFClose(tif);
    return data;
}


int main(int argc, char **argv) {
    printf("=== VLFeat Scale Space Comparison ===\n\n");

    /* Create output directory structure */
    system("mkdir -p vlfeat_comparison/vlfeat_gaussian");
    system("mkdir -p vlfeat_comparison/vlfeat_hessian_det");

    /* Load input image */
    printf("Loading input image...\n");
    int width, height;
    float *input_image = load_float_tiff("vlfeat_comparison/input.tif", &width, &height);
    if (!input_image) {
        fprintf(stderr, "Failed to load input image. Run Julia script first!\n");
        return 1;
    }
    printf("✓ Loaded image: %dx%d\n", width, height);

    /* Create covariant detector with Hessian method */
    printf("\nCreating VLFeat Hessian detector...\n");
    VlCovDet *covdet = vl_covdet_new(VL_COVDET_METHOD_HESSIAN);
    if (!covdet) {
        fprintf(stderr, "Failed to create covdet\n");
        free(input_image);
        return 1;
    }

    /* Put image into detector (this creates both GSS and CSS) */
    printf("Processing image...\n");
    if (vl_covdet_put_image(covdet, input_image, width, height) != 0) {
        fprintf(stderr, "Failed to put image\n");
        vl_covdet_delete(covdet);
        free(input_image);
        return 1;
    }

    /* Trigger detection to compute CSS (Hessian determinant) */
    printf("Computing Hessian determinant (CSS)...\n");
    vl_covdet_detect(covdet);

    /* Get the Gaussian Scale Space (GSS) and Cornerness Scale Space (CSS) */
    VlScaleSpace *gss = vl_covdet_get_gss(covdet);
    VlScaleSpace *css = vl_covdet_get_css(covdet);

    if (!gss || !css) {
        fprintf(stderr, "Failed to get scale spaces\n");
        vl_covdet_delete(covdet);
        free(input_image);
        return 1;
    }

    VlScaleSpaceGeometry gss_geom = vl_scalespace_get_geometry(gss);
    VlScaleSpaceGeometry css_geom = vl_scalespace_get_geometry(css);

    /* Print geometry for verification */
    printf("\nGaussian Scale Space geometry:\n");
    printf("  First octave: %ld\n", gss_geom.firstOctave);
    printf("  Last octave: %ld\n", gss_geom.lastOctave);
    printf("  Octave resolution: %zu\n", gss_geom.octaveResolution);
    printf("  First subdivision: %ld\n", gss_geom.octaveFirstSubdivision);
    printf("  Last subdivision: %ld\n", gss_geom.octaveLastSubdivision);
    printf("  Base scale: %f\n", gss_geom.baseScale);

    printf("\nCornerness Scale Space (Hessian det) geometry:\n");
    printf("  First octave: %ld\n", css_geom.firstOctave);
    printf("  Last octave: %ld\n", css_geom.lastOctave);
    printf("  Octave resolution: %zu\n", css_geom.octaveResolution);
    printf("  First subdivision: %ld\n", css_geom.octaveFirstSubdivision);
    printf("  Last subdivision: %ld\n", css_geom.octaveLastSubdivision);

    /* Save all Gaussian levels */
    printf("\nSaving Gaussian scale space levels...\n");
    for (vl_index o = gss_geom.firstOctave; o <= gss_geom.lastOctave; o++) {
        for (vl_index s = gss_geom.octaveFirstSubdivision; s <= gss_geom.octaveLastSubdivision; s++) {
            float const *level = vl_scalespace_get_level_const(gss, o, s);
            double sigma = vl_scalespace_get_level_sigma(gss, o, s);
            VlScaleSpaceOctaveGeometry ogeom = vl_scalespace_get_octave_geometry(gss, o);

            char filename[256];
            snprintf(filename, sizeof(filename),
                    "vlfeat_comparison/vlfeat_gaussian/gaussian_o%ld_s%ld.tif",
                    o, s);

            if (save_float_tiff(filename, level, ogeom.width, ogeom.height) == 0) {
                printf("  Saved Gaussian: octave=%ld, subdivision=%ld, σ=%.3f\n", o, s, sigma);
            }
        }
    }

    /* Save all Hessian determinant levels from CSS */
    printf("\nSaving Hessian determinant (CSS) levels...\n");
    for (vl_index o = css_geom.firstOctave; o <= css_geom.lastOctave; o++) {
        for (vl_index s = css_geom.octaveFirstSubdivision; s <= css_geom.octaveLastSubdivision; s++) {
            float const *level = vl_scalespace_get_level_const(css, o, s);
            double sigma = vl_scalespace_get_level_sigma(css, o, s);
            VlScaleSpaceOctaveGeometry ogeom = vl_scalespace_get_octave_geometry(css, o);

            char filename[256];
            snprintf(filename, sizeof(filename),
                    "vlfeat_comparison/vlfeat_hessian_det/hessian_det_o%ld_s%ld.tif",
                    o, s);

            if (save_float_tiff(filename, level, ogeom.width, ogeom.height) == 0) {
                printf("  Saved Hessian det: octave=%ld, subdivision=%ld, σ=%.3f\n", o, s, sigma);
            }
        }
    }

    printf("\n✓ All VLFeat files saved to vlfeat_comparison/vlfeat_*/\n");
    printf("  - Gaussian: vlfeat_comparison/vlfeat_gaussian/\n");
    printf("  - Hessian determinant: vlfeat_comparison/vlfeat_hessian_det/\n");

    /* Get detected features from VLFeat */
    printf("\n=== VLFeat Feature Detection ===\n");
    vl_size num_features = vl_covdet_get_num_features(covdet);
    VlCovDetFeature *features = (VlCovDetFeature *)vl_covdet_get_features(covdet);

    printf("✓ VLFeat detected %zu features\n", num_features);

    /* Save features to JSON */
    FILE *fp = fopen("vlfeat_detections.json", "w");
    if (fp) {
        fprintf(fp, "{\n");
        fprintf(fp, "  \"num_features\": %zu,\n", num_features);
        fprintf(fp, "  \"detector\": \"VL_COVDET_METHOD_HESSIAN\",\n");
        fprintf(fp, "  \"parameters\": {\n");
        fprintf(fp, "    \"first_octave\": %ld,\n", gss_geom.firstOctave);
        fprintf(fp, "    \"octave_resolution\": %zu,\n", gss_geom.octaveResolution);
        fprintf(fp, "    \"base_scale\": %.6f,\n", gss_geom.baseScale);
        fprintf(fp, "    \"peak_threshold\": %.6f,\n", vl_covdet_get_peak_threshold(covdet));
        fprintf(fp, "    \"edge_threshold\": %.6f\n", vl_covdet_get_edge_threshold(covdet));
        fprintf(fp, "  },\n");
        fprintf(fp, "  \"features\": [\n");

        for (vl_size i = 0; i < num_features; i++) {
            VlCovDetFeature *f = &features[i];
            /* Compute scale from ellipse: sigma = sqrt(a11^2 + a21^2) */
            double a11 = f->frame.a11;
            double a21 = f->frame.a21;
            double sigma = sqrt(a11*a11 + a21*a21);

            fprintf(fp, "    {\n");
            fprintf(fp, "      \"x\": %.6f,\n", f->frame.x);
            fprintf(fp, "      \"y\": %.6f,\n", f->frame.y);
            fprintf(fp, "      \"sigma\": %.6f,\n", sigma);
            fprintf(fp, "      \"a11\": %.6f,\n", a11);
            fprintf(fp, "      \"a12\": %.6f,\n", f->frame.a12);
            fprintf(fp, "      \"a21\": %.6f,\n", a21);
            fprintf(fp, "      \"a22\": %.6f,\n", f->frame.a22);
            fprintf(fp, "      \"peakScore\": %.6e,\n", f->peakScore);
            fprintf(fp, "      \"edgeScore\": %.6e\n", f->edgeScore);
            fprintf(fp, "    }%s\n", (i < num_features - 1) ? "," : "");
        }

        fprintf(fp, "  ]\n");
        fprintf(fp, "}\n");
        fclose(fp);
        printf("✓ Saved VLFeat detections to vlfeat_detections.json\n");
    }

    /* Print statistics */
    if (num_features > 0) {
        printf("\nVLFeat Feature Statistics:\n");
        printf("  Total features: %zu\n", num_features);

        printf("\nFirst 10 features:\n");
        for (vl_size i = 0; i < (num_features < 10 ? num_features : 10); i++) {
            VlCovDetFeature *f = &features[i];
            /* Compute scale from ellipse */
            double a11 = f->frame.a11;
            double a21 = f->frame.a21;
            double sigma = sqrt(a11*a11 + a21*a21);

            printf("  %2zu: (%.2f, %.2f) σ=%.3f peak=%.6e edge=%.3f\n",
                   i+1, f->frame.x, f->frame.y, sigma,
                   f->peakScore, f->edgeScore);
        }
    }

    /* Cleanup */
    vl_covdet_delete(covdet);
    free(input_image);

    printf("\n=== Done ===\n");
    return 0;
}
