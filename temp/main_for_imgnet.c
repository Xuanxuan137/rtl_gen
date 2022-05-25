#include "call_lib.h"
#define TEST_NUM 999
#define CHANNEL 3
#define HEIGHT 224
#define WIDTH 224
typedef unsigned char uint8;
void read_data(unsigned char * test_images, int * test_labels)
{
    FILE * f_tei = fopen("../imgnet/imgnet_bin.bin", "rb");
    FILE * f_tel = fopen("../imgnet/imgnet_bin_label.bin", "rb");
    if(fread(test_images, sizeof(uint8), TEST_NUM*CHANNEL*HEIGHT*WIDTH, f_tei) != TEST_NUM*CHANNEL*HEIGHT*WIDTH) {
        printf("fread size error at line %d\n", __LINE__);
    }
    if(fread(test_labels, sizeof(int), TEST_NUM, f_tel) != TEST_NUM) {
        printf("fread size error at line %d\n", __LINE__);
    }
    fclose(f_tei);
    fclose(f_tel);
}
void copy_data(float * test_images, unsigned char * test_images_char)
{
    // for(int i = 0; i<TEST_NUM*CHANNEL*HEIGHT*WIDTH; i++) {
    //     test_images[i] = (float)test_images_char[i] / 255.0f;
    // }
	float mean[3] = {123.15, 115.90, 103.06};
	float var[3] = {58.395, 57.12, 57.375};
	for(int n = 0; n<TEST_NUM; n++) {
		for(int c = 0; c<CHANNEL; c++) {
			for(int h = 0; h<HEIGHT; h++) {
				for(int w = 0; w<WIDTH; w++) {
					test_images[
						n * CHANNEL * HEIGHT * WIDTH + 
						c * HEIGHT * WIDTH + 
						h * WIDTH + 
						w
					] = ((float)test_images_char[
						n * CHANNEL * HEIGHT * WIDTH + 
						c * HEIGHT * WIDTH + 
						h * WIDTH + 
						w
					] - mean[c]) / var[c];
				}
			}
		}
	}
}
int main(void)
{
	// load your dataset
    uint8 * test_images_char = (uint8*)malloc(sizeof(uint8)*TEST_NUM*CHANNEL*HEIGHT*WIDTH);
    int * test_labels_char = (int*)malloc(sizeof(int)*TEST_NUM);
    read_data(test_images_char, test_labels_char);
    float * test_images = (float*)malloc(sizeof(float)*TEST_NUM*CHANNEL*HEIGHT*WIDTH);
    copy_data(test_images, test_images_char);

	// init weight and result memory
	init();

	// alloc space for result
	float ** result = (float**)malloc(sizeof(float*)*1);

	// calculation
	int correct = 0;
    for(int i = 0; i<TEST_NUM; i++) {
        calc(result, &test_images[i*3*224*224]);
        int answer = argmax(result[0], 1000);
        if(answer == test_labels_char[i]) {
            correct ++;
        }
        printf("\r%d/%d", correct, i+1);
        fflush(stdout);
    }
    printf("\n");
}
