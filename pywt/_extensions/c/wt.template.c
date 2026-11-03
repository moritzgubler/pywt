/* Copyright (c) 2006-2012 Filip Wasilewski <http://en.ig.ma/>
 * Copyright (c) 2012-2016 The PyWavelets Developers
 *                         <https://github.com/PyWavelets/pywt>
 * See COPYING for license details.
 */

#include "templating.h"

#ifndef TYPE
#error TYPE must be defined here.
#else

#ifndef REAL_TYPE
#error REAL_TYPE must be defined here.
#else

#include "wt.h"
#include "timing.h"

#if defined _MSC_VER
#define restrict __restrict
#elif defined __GNUC__
#define restrict __restrict__
#endif

/* Lines copied together on a strided axis (speed work): neighbours along the contiguous last axis, so every strided
 * access moves PYWT_BLOCK consecutive values (32 floats = one 128-byte cache line on Apple Silicon). */
#ifndef PYWT_BLOCK
#define PYWT_BLOCK 32
#endif

/* Offsets (bytes) of line i: all indices except the one along axis, last axis fastest */
static inline void CAT(TYPE, _line_offsets)(size_t i, const ArrayInfo * const a, const ArrayInfo * const b,
                                            const ArrayInfo * const c, const size_t axis,
                                            pywt_index_t * oa, pywt_index_t * ob, pywt_index_t * oc){
    size_t j;
    *oa = *ob = *oc = 0;
    for (j = 0; j < c->ndim; ++j){
        size_t j_rev = c->ndim - 1 - j;
        if (j_rev != axis){
            size_t axis_idx = i % c->shape[j_rev];
            i /= c->shape[j_rev];
            if (a) *oa += axis_idx * a->strides[j_rev];
            if (b) *ob += axis_idx * b->strides[j_rev];
            *oc += axis_idx * c->strides[j_rev];
        }
    }
}

/* One line of a decomposition */
static inline void CAT(TYPE, _downcoef_row)(const TYPE * const restrict input_row, const size_t n_in,
                                            const DiscreteWavelet * const restrict wavelet,
                                            TYPE * const restrict output_row, const size_t n_out,
                                            const Coefficient coef, const MODE dwt_mode,
                                            const size_t swt_level, const DiscreteTransformType transform){
    switch (transform) {
        case DWT_TRANSFORM:
            if (coef == COEF_APPROX)
                CAT(TYPE, _dec_a)(input_row, n_in, wavelet, output_row, n_out, dwt_mode);
            else
                CAT(TYPE, _dec_d)(input_row, n_in, wavelet, output_row, n_out, dwt_mode);
            break;
        case SWT_TRANSFORM:
            if (coef == COEF_APPROX)
                CAT(TYPE, _swt_a)(input_row, n_in, wavelet, output_row, n_out, swt_level);
            else
                CAT(TYPE, _swt_d)(input_row, n_in, wavelet, output_row, n_out, swt_level);
            break;
    }
}

/* Decomposition of input with lowpass filter */

int CAT(TYPE, _downcoef_axis)(const TYPE * const restrict input, const ArrayInfo input_info,
                              TYPE * const restrict output, const ArrayInfo output_info,
                              const DiscreteWavelet * const restrict wavelet, const size_t axis,
                              const Coefficient coef, const MODE dwt_mode,
                              const size_t swt_level,
                              const DiscreteTransformType transform){
    size_t i;
    size_t num_loops = 1;
    TYPE * temp_input = NULL, * temp_output = NULL;

    // These are boolean values, but MSVC does not have <stdbool.h>
    int make_temp_input, make_temp_output;

    if (input_info.ndim != output_info.ndim)
        return 1;
    if (axis >= input_info.ndim)
        return 2;

    for (i = 0; i < input_info.ndim; ++i){
        if (i == axis){
            switch (transform) {
            case DWT_TRANSFORM:
                if (dwt_buffer_length(input_info.shape[i], wavelet->dec_len,
                                      dwt_mode) != output_info.shape[i])
                    return 3;
                break;
            case SWT_TRANSFORM:
                if (swt_buffer_length(input_info.shape[i])
                    != output_info.shape[i])
                    return 4;
                break;
            }
        } else {
            if (input_info.shape[i] != output_info.shape[i])
                return 5;
        }
    }

    make_temp_input = input_info.strides[axis] != sizeof(TYPE);
    make_temp_output = output_info.strides[axis] != sizeof(TYPE);
    if (make_temp_input)
        if ((temp_input = malloc(input_info.shape[axis] * sizeof(TYPE))) == NULL)
            goto cleanup;
    if (make_temp_output)
        if ((temp_output = malloc(output_info.shape[axis] * sizeof(TYPE))) == NULL)
            goto cleanup;

    for (i = 0; i < output_info.ndim; ++i){
        if (i != axis)
            num_loops *= output_info.shape[i];
    }

    const int timing = pywt_timing_on();
    double t0 = 0, t1 = 0, t2 = 0;
    pywt_calls[T_DEC] += timing;

    // Blocked path: strided axis, last axis contiguous in input and output
    const size_t last = input_info.ndim - 1;
    if (make_temp_input && make_temp_output && axis != last
            && input_info.strides[last] == sizeof(TYPE) && output_info.strides[last] == sizeof(TYPE)){
        const size_t n_in = input_info.shape[axis], n_out = output_info.shape[axis], n_last = input_info.shape[last];
        TYPE * const bin = malloc(PYWT_BLOCK * n_in * sizeof(TYPE));
        TYPE * const bout = malloc(PYWT_BLOCK * n_out * sizeof(TYPE));
        if (bin == NULL || bout == NULL){
            free(bin); free(bout);
            goto cleanup;
        }
        for (i = 0; i < num_loops; ){
            size_t j, b, g = n_last - i % n_last;          // lines i .. i + g - 1: neighbours along the last axis
            pywt_index_t input_offset, output_offset, unused;
            if (g > PYWT_BLOCK) g = PYWT_BLOCK;
            if (timing) t0 = pywt_now();
            CAT(TYPE, _line_offsets)(i, &input_info, NULL, &output_info, axis, &input_offset, &unused, &output_offset);
            for (j = 0; j < n_in; ++j){
                const TYPE * src = (const TYPE *)((const char *) input + input_offset + j * input_info.strides[axis]);
                for (b = 0; b < g; ++b)
                    bin[b * n_in + j] = src[b];
            }
            if (timing) t1 = pywt_now();
            for (b = 0; b < g; ++b)
                CAT(TYPE, _downcoef_row)(bin + b * n_in, n_in, wavelet, bout + b * n_out, n_out,
                                         coef, dwt_mode, swt_level, transform);
            if (timing) t2 = pywt_now();
            for (j = 0; j < n_out; ++j){
                TYPE * dst = (TYPE *)((char *) output + output_offset + j * output_info.strides[axis]);
                for (b = 0; b < g; ++b)
                    dst[b] = bout[b * n_out + j];
            }
            if (timing){
                pywt_t[T_DEC][T_IN] += t1 - t0;
                pywt_t[T_DEC][T_KERNEL] += t2 - t1;
                pywt_t[T_DEC][T_OUT] += pywt_now() - t2;
            }
            i += g;
        }
        free(bin); free(bout);
        free(temp_input); free(temp_output);
        return 0;
    }

    for (i = 0; i < num_loops; ++i){
        pywt_index_t j, input_offset, output_offset, unused;
        const TYPE * input_row;
        TYPE * output_row;
        if (timing) t0 = pywt_now();

        // Calculate offset into linear buffer
        CAT(TYPE, _line_offsets)(i, &input_info, NULL, &output_info, axis, &input_offset, &unused, &output_offset);

        // Copy to temporary input if necessary
        if (make_temp_input)
            for (j = 0; j < input_info.shape[axis]; ++j)
                // Offsets are byte offsets, to need to cast to char and back
                temp_input[j] = *(TYPE *)(((char *) input) + input_offset
                                          + j * input_info.strides[axis]);

        // Select temporary or direct output and input
        input_row = make_temp_input ? temp_input
            : (const TYPE *)((const char *) input + input_offset);
        output_row = make_temp_output ? temp_output
            : (TYPE *)((char *) output + output_offset);

        if (timing) t1 = pywt_now();
        CAT(TYPE, _downcoef_row)(input_row, input_info.shape[axis], wavelet, output_row, output_info.shape[axis],
                                 coef, dwt_mode, swt_level, transform);
        if (timing) t2 = pywt_now();

        // Copy from temporary output if necessary
        if (make_temp_output)
            for (j = 0; j < output_info.shape[axis]; ++j)
                // Offsets are byte offsets, to need to cast to char and back
                *(TYPE *)((char *) output + output_offset
                          + j * output_info.strides[axis]) = output_row[j];
        if (timing){
            pywt_t[T_DEC][T_IN] += t1 - t0;
            pywt_t[T_DEC][T_KERNEL] += t2 - t1;
            pywt_t[T_DEC][T_OUT] += pywt_now() - t2;
        }
    }

    free(temp_input);
    free(temp_output);
    return 0;

 cleanup:
    free(temp_input);
    free(temp_output);
    return 6;
}


/* One line of a reconstruction: output = rec_lo * a + rec_hi * d (upsampled); a_row or d_row may be NULL */
static inline void CAT(TYPE, _idwt_row)(const TYPE * const restrict a_row, const TYPE * const restrict d_row,
                                        const size_t n, const DiscreteWavelet * const restrict wavelet,
                                        TYPE * const restrict output_row, const size_t n_out, const MODE mode){
    memset(output_row, 0, n_out * sizeof(TYPE));    // upsampling_convolution adds to the output
    if (a_row)
        CAT(TYPE, _upsampling_convolution_valid_sf)(a_row, n, wavelet->CAT(rec_lo_, REAL_TYPE), wavelet->rec_len,
                                                     output_row, n_out, mode);
    if (d_row)
        CAT(TYPE, _upsampling_convolution_valid_sf)(d_row, n, wavelet->CAT(rec_hi_, REAL_TYPE), wavelet->rec_len,
                                                     output_row, n_out, mode);
}

int CAT(TYPE, _idwt_axis)(const TYPE * const restrict coefs_a, const ArrayInfo * const a_info,
                          const TYPE * const restrict coefs_d, const ArrayInfo * const d_info,
                          TYPE * const restrict output, const ArrayInfo output_info,
                          const DiscreteWavelet * const restrict wavelet,
                          const size_t axis, const MODE mode){
    size_t i;
    size_t num_loops = 1;
    TYPE * temp_coefs_a = NULL, * temp_coefs_d = NULL, * temp_output = NULL;

    // These are boolean values, but MSVC does not have <stdbool.h>
    int make_temp_coefs_a, make_temp_coefs_d, make_temp_output;
    int have_a = ((coefs_a != NULL) && (a_info != NULL));
    int have_d = ((coefs_d != NULL) && (d_info != NULL));


    if (!have_a && !have_d)
        return 3;

    if ((have_a && (a_info->ndim != output_info.ndim)) ||
        (have_d && (d_info->ndim != output_info.ndim)))
        return 1;
    if (axis >= output_info.ndim)
        return 1;

    for (i = 0; i < output_info.ndim; ++i){
        if (i == axis){
            size_t input_shape;
            if (have_a && have_d &&
                (d_info->shape[i] != a_info->shape[i]))
                return 1;
            input_shape = have_a ? a_info->shape[i] : d_info->shape[i];

            /* TODO: reconstruction_buffer_length should take a & d shapes
             *       - for odd output_len, d_len == (a_len - 1)
             */
            if (idwt_buffer_length(input_shape, wavelet->rec_len, mode)
                != output_info.shape[i])
                return 1;
        } else {
            if ((have_a && (a_info->shape[i] != output_info.shape[i])) ||
                (have_d && (d_info->shape[i] != output_info.shape[i])))
                return 1;
        }
    }

    make_temp_coefs_a = have_a && a_info->strides[axis] != sizeof(TYPE);
    make_temp_coefs_d = have_d && d_info->strides[axis] != sizeof(TYPE);
    make_temp_output = output_info.strides[axis] != sizeof(TYPE);
    if (make_temp_coefs_a)
        if ((temp_coefs_a = malloc(a_info->shape[axis] * sizeof(TYPE))) == NULL)
            goto cleanup;
    if (make_temp_coefs_d)
        if ((temp_coefs_d = malloc(d_info->shape[axis] * sizeof(TYPE))) == NULL)
            goto cleanup;
    if (make_temp_output)
        if ((temp_output = malloc(output_info.shape[axis] * sizeof(TYPE))) == NULL)
            goto cleanup;

    for (i = 0; i < output_info.ndim; ++i){
        if (i != axis)
            num_loops *= output_info.shape[i];
    }

    const int timing = pywt_timing_on();
    double t0 = 0, t1 = 0, t2 = 0;
    pywt_calls[T_REC] += timing;

    // Blocked path: strided axis, last axis contiguous in all present inputs and the output
    const size_t last = output_info.ndim - 1;
    if (make_temp_output && axis != last && output_info.strides[last] == sizeof(TYPE)
            && (!have_a || (make_temp_coefs_a && a_info->strides[last] == sizeof(TYPE)))
            && (!have_d || (make_temp_coefs_d && d_info->strides[last] == sizeof(TYPE)))){
        const size_t n = have_a ? a_info->shape[axis] : d_info->shape[axis];
        const size_t n_out = output_info.shape[axis], n_last = output_info.shape[last];
        TYPE * const ba = have_a ? malloc(PYWT_BLOCK * n * sizeof(TYPE)) : NULL;
        TYPE * const bd = have_d ? malloc(PYWT_BLOCK * n * sizeof(TYPE)) : NULL;
        TYPE * const bout = malloc(PYWT_BLOCK * n_out * sizeof(TYPE));
        if ((have_a && ba == NULL) || (have_d && bd == NULL) || bout == NULL){
            free(ba); free(bd); free(bout);
            goto cleanup;
        }
        for (i = 0; i < num_loops; ){
            size_t j, b, g = n_last - i % n_last;          // lines i .. i + g - 1: neighbours along the last axis
            pywt_index_t a_offset, d_offset, output_offset;
            if (g > PYWT_BLOCK) g = PYWT_BLOCK;
            if (timing) t0 = pywt_now();
            CAT(TYPE, _line_offsets)(i, have_a ? a_info : NULL, have_d ? d_info : NULL, &output_info, axis,
                                     &a_offset, &d_offset, &output_offset);
            for (j = 0; j < n; ++j){
                if (have_a){
                    const TYPE * src = (const TYPE *)((const char *) coefs_a + a_offset + j * a_info->strides[axis]);
                    for (b = 0; b < g; ++b)
                        ba[b * n + j] = src[b];
                }
                if (have_d){
                    const TYPE * src = (const TYPE *)((const char *) coefs_d + d_offset + j * d_info->strides[axis]);
                    for (b = 0; b < g; ++b)
                        bd[b * n + j] = src[b];
                }
            }
            if (timing) t1 = pywt_now();
            for (b = 0; b < g; ++b)
                CAT(TYPE, _idwt_row)(have_a ? ba + b * n : NULL, have_d ? bd + b * n : NULL, n, wavelet,
                                     bout + b * n_out, n_out, mode);
            if (timing) t2 = pywt_now();
            for (j = 0; j < n_out; ++j){
                TYPE * dst = (TYPE *)((char *) output + output_offset + j * output_info.strides[axis]);
                for (b = 0; b < g; ++b)
                    dst[b] = bout[b * n_out + j];
            }
            if (timing){
                pywt_t[T_REC][T_IN] += t1 - t0;
                pywt_t[T_REC][T_KERNEL] += t2 - t1;
                pywt_t[T_REC][T_OUT] += pywt_now() - t2;
            }
            i += g;
        }
        free(ba); free(bd); free(bout);
        free(temp_coefs_a); free(temp_coefs_d); free(temp_output);
        return 0;
    }

    for (i = 0; i < num_loops; ++i){
        pywt_index_t j, axis_idx, a_offset = 0, d_offset = 0, output_offset = 0;
        TYPE * output_row;
        if (timing) t0 = pywt_now();

        // Calculate offset into linear buffer
        {
            size_t reduced_idx = i;
            for (j = 0; j < output_info.ndim; ++j){
                size_t j_rev = output_info.ndim - 1 - j;
                if (j_rev != axis){
                    axis_idx = reduced_idx % output_info.shape[j_rev];
                    reduced_idx /= output_info.shape[j_rev];

                    if (have_a)
                        a_offset += (axis_idx * a_info->strides[j_rev]);
                    if (have_d)
                        d_offset += (axis_idx * d_info->strides[j_rev]);
                    output_offset += (axis_idx * output_info.strides[j_rev]);
                }
            }
        }

        // Copy to temporary input if necessary
        if (make_temp_coefs_a)
            for (j = 0; j < a_info->shape[axis]; ++j)
                // Offsets are byte offsets, to need to cast to char and back
                temp_coefs_a[j] = *(TYPE *)((char *) coefs_a + a_offset
                                            + j * a_info->strides[axis]);
        if (make_temp_coefs_d)
            for (j = 0; j < d_info->shape[axis]; ++j)
                // Offsets are byte offsets, to need to cast to char and back
                temp_coefs_d[j] = *(TYPE *)((char *) coefs_d + d_offset
                                            + j * d_info->strides[axis]);

        // Select temporary or direct output
        output_row = make_temp_output ? temp_output
            : (TYPE *)((char *) output + output_offset);

        if (timing) t1 = pywt_now();
        // upsampling_convolution adds to input, so zero
        memset(output_row, 0, output_info.shape[axis] * sizeof(TYPE));

        if (have_a){
            // Pointer arithmetic on NULL is undefined
            const TYPE * a_row = make_temp_coefs_a ? temp_coefs_a
                : (const TYPE *)((const char *) coefs_a + a_offset);
            CAT(TYPE, _upsampling_convolution_valid_sf)
                (a_row, a_info->shape[axis],
                 wavelet->CAT(rec_lo_, REAL_TYPE), wavelet->rec_len,
                 output_row, output_info.shape[axis],
                 mode);
        }
        if (have_d){
            // Pointer arithmetic on NULL is undefined
            const TYPE * d_row = make_temp_coefs_d ? temp_coefs_d
                : (const TYPE *)((const char *) coefs_d + d_offset);
            CAT(TYPE, _upsampling_convolution_valid_sf)
                (d_row, d_info->shape[axis],
                 wavelet->CAT(rec_hi_, REAL_TYPE), wavelet->rec_len,
                 output_row, output_info.shape[axis],
                 mode);
        }
        if (timing) t2 = pywt_now();

        // Copy from temporary output if necessary
        if (make_temp_output)
            for (j = 0; j < output_info.shape[axis]; ++j)
                // Offsets are byte offsets, to need to cast to char and back
                *(TYPE *)((char *) output + output_offset
                          + j * output_info.strides[axis]) = output_row[j];
        if (timing){
            pywt_t[T_REC][T_IN] += t1 - t0;
            pywt_t[T_REC][T_KERNEL] += t2 - t1;
            pywt_t[T_REC][T_OUT] += pywt_now() - t2;
        }
    }

    free(temp_coefs_a);
    free(temp_coefs_d);
    free(temp_output);
    return 0;

 cleanup:
    free(temp_coefs_a);
    free(temp_coefs_d);
    free(temp_output);
    return 2;
}


int CAT(TYPE, _dec_a)(const TYPE * const restrict input, const size_t input_len,
                      const DiscreteWavelet * const restrict wavelet,
                      TYPE * const restrict output, const size_t output_len,
                      const MODE mode){

    /* check output length */
    if(output_len != dwt_buffer_length(input_len, wavelet->dec_len, mode)){
        return -1;
    }

    return CAT(TYPE, _downsampling_convolution)(input, input_len,
                                                wavelet->CAT(dec_lo_, REAL_TYPE),
                                                wavelet->dec_len, output,
                                                2, mode);
}


/* Decomposition of input with highpass filter */

int CAT(TYPE, _dec_d)(const TYPE * const restrict input, const size_t input_len,
                      const DiscreteWavelet * const restrict wavelet,
                      TYPE * const restrict output, const size_t output_len,
                      const MODE mode){

    /* check output length */
    if(output_len != dwt_buffer_length(input_len, wavelet->dec_len, mode))
        return -1;

    return CAT(TYPE, _downsampling_convolution)(input, input_len,
                                                wavelet->CAT(dec_hi_, REAL_TYPE),
                                                wavelet->dec_len, output,
                                                2, mode);
}


/* Direct reconstruction with lowpass reconstruction filter */

int CAT(TYPE, _rec_a)(const TYPE * const restrict coeffs_a, const size_t coeffs_len,
                      const DiscreteWavelet * const restrict wavelet,
                      TYPE * const restrict output, const size_t output_len){

    /* check output length */
    if(output_len != reconstruction_buffer_length(coeffs_len, wavelet->rec_len))
        return -1;

    return CAT(TYPE, _upsampling_convolution_full)(coeffs_a, coeffs_len,
                                                   wavelet->CAT(rec_lo_, REAL_TYPE),
                                                   wavelet->rec_len, output,
                                                   output_len);
}


/* Direct reconstruction with highpass reconstruction filter */

int CAT(TYPE, _rec_d)(const TYPE * const restrict coeffs_d, const size_t coeffs_len,
                      const DiscreteWavelet * const restrict wavelet,
                      TYPE * const restrict output, const size_t output_len){

    /* check for output length */
    if(output_len != reconstruction_buffer_length(coeffs_len, wavelet->rec_len))
        return -1;

    return CAT(TYPE, _upsampling_convolution_full)(coeffs_d, coeffs_len,
                                                   wavelet->CAT(rec_hi_, REAL_TYPE),
                                                   wavelet->rec_len, output,
                                                   output_len);
}


/*
 * IDWT reconstruction from approximation and detail coeffs, either of which may
 * be NULL.
 *
 * Requires zero-filled output buffer.
 */
int CAT(TYPE, _idwt)(const TYPE * const restrict coeffs_a, const size_t coeffs_a_len,
                     const TYPE * const restrict coeffs_d, const size_t coeffs_d_len,
                     TYPE * const restrict output, const size_t output_len,
                     const DiscreteWavelet * const restrict wavelet, const MODE mode){
    size_t input_len;
    if(coeffs_a != NULL && coeffs_d != NULL){
        if(coeffs_a_len != coeffs_d_len)
            goto error;
        input_len = coeffs_a_len;
    } else if(coeffs_a != NULL){
        input_len  = coeffs_a_len;
    } else if (coeffs_d != NULL){
        input_len = coeffs_d_len;
    } else {
        goto error;
    }

    /* check output size */
    if(output_len != idwt_buffer_length(input_len, wavelet->rec_len, mode))
        goto error;

    /*
     * Set output to zero (this can be omitted if output array is already
     * cleared) memset(output, 0, output_len * sizeof(TYPE));
     */

    /* reconstruct approximation coeffs with lowpass reconstruction filter */
    if(coeffs_a){
        if(CAT(TYPE, _upsampling_convolution_valid_sf)(coeffs_a, input_len,
                                                  wavelet->CAT(rec_lo_, REAL_TYPE),
                                                  wavelet->rec_len, output,
                                                  output_len, mode) < 0){
            goto error;
        }
    }
    /*
     * Add reconstruction of details coeffs performed with highpass
     * reconstruction filter.
     */
    if(coeffs_d){
        if(CAT(TYPE, _upsampling_convolution_valid_sf)(coeffs_d, input_len,
                                                  wavelet->CAT(rec_hi_, REAL_TYPE),
                                                  wavelet->rec_len, output,
                                                  output_len, mode) < 0){
            goto error;
        }
    }

    return 0;

    error:
        return -1;
}

/* basic SWT step (TODO: optimize) */
int CAT(TYPE, _swt_)(const TYPE * const restrict input, pywt_index_t input_len,
                     const REAL_TYPE * const restrict filter, pywt_index_t filter_len,
                     TYPE * const restrict output, size_t output_len,
                     unsigned int level){

    REAL_TYPE * e_filter;
    pywt_index_t i, e_filter_len, fstep;
    int ret;

    if(level < 1)
        return -1;

    if(level > swt_max_level(input_len))
        return -2;

    if(output_len != swt_buffer_length(input_len))
        return -1;

    /* TODO: quick hack, optimize */
    if(level > 1){
        /* allocate filter first */
        e_filter_len = filter_len << (level-1);
        if ((e_filter = wtcalloc(e_filter_len, sizeof(TYPE))) == NULL)
            goto cleanup;
        if(e_filter == NULL)
            return -1;
        fstep = 1 << (level - 1);  // spacing between non-zero filter entries

        /* compute upsampled filter values */
        for(i = 0; i < filter_len; ++i){
            e_filter[i << (level-1)] = filter[i];
        }
        ret = CAT(TYPE, _downsampling_convolution_periodization)(input, input_len, e_filter,
                                                   e_filter_len, output, 1,
                                                   fstep);
        wtfree(e_filter);
        return ret;

    } else {
        return CAT(TYPE, _downsampling_convolution_periodization)(input, input_len, filter,
                                                    filter_len, output, 1,
                                                    1);
    }
 cleanup:
    wtfree(e_filter);
    return -3;
}

/*
 * Approximation at specified level
 * input - approximation coeffs from upper level or signal if level == 1
 */
int CAT(TYPE, _swt_a)(const TYPE * const restrict input, pywt_index_t input_len,
                      const DiscreteWavelet * const restrict wavelet,
                      TYPE * const restrict output, pywt_index_t output_len,
                      unsigned int level){
    return CAT(TYPE, _swt_)(input, input_len, wavelet->CAT(dec_lo_, REAL_TYPE),
                            wavelet->dec_len, output, output_len, level);
}

/* Details at specified level
 * input - approximation coeffs from upper level or signal if level == 1
 */
int CAT(TYPE, _swt_d)(const TYPE * const restrict input, pywt_index_t input_len,
                      const DiscreteWavelet * const restrict wavelet,
                      TYPE * const restrict output, pywt_index_t output_len,
                      unsigned int level){
    return CAT(TYPE, _swt_)(input, input_len, wavelet->CAT(dec_hi_, REAL_TYPE),
                            wavelet->dec_len, output, output_len, level);
}

#endif /* REAL_TYPE */
#endif /* TYPE */
#undef restrict
