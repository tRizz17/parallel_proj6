3 Heads Dimensions:

1280 x 1280

bounding_box_image = a copy of the image

int count = 0
count_coint(image_file, coin(either head or tail)_file, bounding_box_image, image_height(1280), image_width(1280), coin_height(128), coin_width(128))

for (int y = 0; y < (1280 - 128 + 1); y += coin_height)
    for (int x = 0; x < (1280 - 128 + 1) x += coin_width)


    if (calculate_correlation(image_file + y * 1280 + x, coin, 1280, 128, 128 ) < CORRELATION_THRESHOLD)
        count ++ 


calculate_correlation(image_file)



