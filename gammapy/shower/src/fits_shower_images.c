#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <fitsio.h>

#include "fits_shower_images.h"

void printerror(int status)
{
    if (status) {
        fits_report_error(stderr, status);
        exit(status);
    }
    return;
}

void import_from_fits(char *file_in)
{   
    Ppixel_info_HESS1 = pixel_info_HESS1;
    Ppixel_info_HESS2 = pixel_info_HESS2;

    uint8_t shower_header_is_pixel_list;
    char extname[] = "SHOWER_IMAGE_DATA";
    uint8_t buffer[BUFFER_MAX];

    fitsfile *fptr;
    int status = 0;
    
    if ( fits_open_file(&fptr, file_in, READONLY, &status) ) {
        return;
        printerror( status );
    }
    
    int n_hdu;
    if ( fits_get_num_hdus(fptr, &n_hdu, &status) ) {
        return;
        printerror( status );
    }
    
    if ( fits_read_key(fptr, TINT, "n_images", &n_images, NULL, &status) ) {
        return;
        printerror( status );
    }
        
    im = (Image *) malloc(n_images * sizeof(Image));
    
    if ( fits_movnam_hdu(fptr, BINARY_TBL, extname, 0, &status) ) {
        return;
        printerror( status );
    }
    
    long nrows;
    if ( fits_get_num_rows(fptr, &nrows, &status) ) {
        return;
        printerror( status );
    }
    printf("n_rows = %d\n", (int) nrows);
    
    int index = 0;
    int row;
    uint16_t k;
    for (row = 1; row <= nrows; row++) {
        if ( fits_read_tblbytes(fptr, row, 1, BUFFER_MAX, buffer, &status) ) {
            return;
            printerror( status );
        }

        int pos = 0;
        while ((buffer[pos] & 128) == 0) {   // while end of table cell not reached                                                                                                                                               
            uint8_t header = buffer[pos];
            pos++;
            shower_header_is_pixel_list = (uint8_t) ((header & 64) != 0);
            uint8_t high_intensities_exist = (uint8_t) ((header & 32) != 0);
            uint8_t shower_image_has_more_than_255_pixels = (uint8_t) ((header & 16) != 0);
            im[index].min = (float) ( (uint8_t) (header << 4) >> 4);
            if (shower_header_is_pixel_list == 1) {
                // read how many pixels there are in the shower
                if (shower_image_has_more_than_255_pixels == 1) {
                    im[index].n_pixels = ( ((uint16_t) buffer[pos]) << 8) | (uint16_t) buffer[pos+1];
                    pos += 2;
                } else {
                    im[index].n_pixels = buffer[pos];
                    pos++;
                }

                im[index].id = (uint16_t *) malloc(im[index].n_pixels * sizeof(uint16_t));
                im[index].intensity = (float *) malloc(im[index].n_pixels * sizeof(float));
                // read the pixel IDs:
                for (k = 0; k < im[index].n_pixels; k += 8) {
                    if (k+8 > im[index].n_pixels) {
                        im[index].id[k+0] = (((uint16_t) buffer[pos + 0])       <<  3) | (uint16_t) (buffer[pos + 1] >> 5);
                        if (k+8-im[index].n_pixels < 7) {
                            im[index].id[k+1] = (((uint16_t) buffer[pos + 1] &  31) <<  6) | (uint16_t) (buffer[pos + 2] >> 2);
                            if (k+8-im[index].n_pixels < 6) {
                                im[index].id[k+2] = (((uint16_t) buffer[pos + 2] &   3) <<  9) | (uint16_t) (buffer[pos + 3] << 1) | (uint16_t) (buffer[pos+4] >> 7);
                                if (k+8-im[index].n_pixels < 5) {
                                    im[index].id[k+3] = (((uint16_t) buffer[pos + 4] & 127) <<  4) | (uint16_t) (buffer[pos + 5] >> 4);
                                    if (k+8-im[index].n_pixels < 4) {
                                        im[index].id[k+4] = (((uint16_t) buffer[pos + 5] &  15) <<  7) | (uint16_t) (buffer[pos + 6] >> 1);
                                        if (k+8-im[index].n_pixels < 3) {
                                            im[index].id[k+5] = (((uint16_t) buffer[pos + 6] &   1) << 10) | (uint16_t) (buffer[pos + 7] << 2) | (uint16_t) (buffer[pos+8] >> 6);
                                            if (k+8-im[index].n_pixels < 2) {
                                                im[index].id[k+6] = (((uint16_t) buffer[pos + 8] &  63) <<  5) | (uint16_t) (buffer[pos + 9] >> 3);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        im[index].id[k+0] = (((uint16_t) buffer[pos + 0])       <<  3) | (uint16_t) (buffer[pos + 1] >> 5);
                        im[index].id[k+1] = (((uint16_t) buffer[pos + 1] &  31) <<  6) | (uint16_t) (buffer[pos + 2] >> 2);
                        im[index].id[k+2] = (((uint16_t) buffer[pos + 2] &   3) <<  9) | (uint16_t) (buffer[pos + 3] << 1) | (uint16_t) (buffer[pos + 4] >> 7);
                        im[index].id[k+3] = (((uint16_t) buffer[pos + 4] & 127) <<  4) | (uint16_t) (buffer[pos + 5] >> 4);
                        im[index].id[k+4] = (((uint16_t) buffer[pos + 5] &  15) <<  7) | (uint16_t) (buffer[pos + 6] >> 1);
                        im[index].id[k+5] = (((uint16_t) buffer[pos + 6] &   1) << 10) | (uint16_t) (buffer[pos + 7] << 2) | (uint16_t) (buffer[pos + 8] >> 6);
                        im[index].id[k+6] = (((uint16_t) buffer[pos + 8] &  63) <<  5) | (uint16_t) (buffer[pos + 9] >> 3);
                        im[index].id[k+7] = (((uint16_t) buffer[pos + 9] &   7) <<  8) | (uint16_t)  buffer[pos + 10];
                    }
                    pos += 11;
                }

                // store the pixel intensities in the simple case all pixels had an intensity <= MIN+25.5
                if (high_intensities_exist == 0) {
                    im[index].max = -1;
                    for (k = 0; k < im[index].n_pixels; k++) {
                        im[index].intensity[k] = im[index].min + buffer[pos]/10.;
                        pos++;
                        if (im[index].intensity[k] > im[index].max) {
                            im[index].max = im[index].intensity[k];
                        }
                    }
                } else { // otherwise, if some have an intensity > MIN+25.5, we have to use more difficult encoding
                    im[index].max = -1;
                    for (k = 0; k < im[index].n_pixels; k++) {
                        uint8_t a = buffer[pos];
                        pos++;
                        // low intensity pixel
                        if ((a & 128) == 0) {
                            im[index].intensity[k] = im[index].min + ((float) ( ((uint8_t) (a << 1)) >> 1)) / 10.;
                        } else {
                            uint16_t b = ( ((uint16_t) (((uint16_t) a) << 9)) >> 1) | buffer[pos];
                            pos++;
                            im[index].intensity[k] = im[index].min + ((float) b) / 10.;
                        }
                        if (im[index].intensity[k] > im[index].max) {
                            im[index].max = im[index].intensity[k];
                        }
                    }
                }
            } else {
                // read the pixel intensities in the simple case all pixels had an intensity <= MIN+25.5
                im[index].max = -1;
                im[index].n_pixels = 2048;
                im[index].id = (uint16_t *) malloc(im[index].n_pixels * sizeof(uint16_t));
                im[index].intensity = (float *) malloc(im[index].n_pixels * sizeof(float));
                if (high_intensities_exist == 0) {
                    for (k = 0; k < 2048; k++) {
                        im[index].id[k] = k;
                        im[index].intensity[k] = im[index].min + ((float) buffer[pos]) / 10.;
                        pos++;
                        if (im[index].intensity[k] > im[index].max) {
                            im[index].max = im[index].intensity[k];
                        }
                    }
                } else { // otherwise, if some have an intensity > 25.5, we have to use more difficult encoding                                                                                                                   
                    for (k = 0; k < 2048; k++) {
                        im[index].id[k] = k;
                        uint8_t a = buffer[pos];
                        pos++;
                        // if the intensity < 127                                                                                                                                                                                 
                        if ((a & 128) == 0) {
                            im[index].intensity[k] = im[index].min + ((float) ( ((uint8_t) (a << 1)) >> 1)) / 10.;
                        } else { // if two bytes are needed: 127 < x < 3200                                                                                                                                                       
                            uint16_t b = ( ((uint16_t) (((uint16_t) a) << 9)) >> 1) | buffer[pos];
                            pos++;
                            im[index].intensity[k] = im[index].min + ((float) b) / 10.;
                        }
                        if (im[index].intensity[k] > im[index].max) {
                            im[index].max = im[index].intensity[k];
                        }
                    }
                }
            }
            index++;
        }
    }
    if ( fits_close_file(fptr, &status) ) {
        return;
        printerror( status );
    }
}
