#include "call_lib.h"
#define TRAIN_NUM 60000
#define TEST_NUM 10000
#define CHANNEL 1
#define HEIGHT 28
#define WIDTH 28
void read_data(unsigned char * train_images, unsigned char * train_labels, 
               unsigned char * test_images, unsigned char * test_labels)
{
    FILE * f_tri = fopen("../../../../../dataset/mnist/train_images.bin", "rb");
    FILE * f_trl = fopen("../../../../../dataset/mnist/train_labels.bin", "rb");
    FILE * f_tei = fopen("../../../../../dataset/mnist/test_images.bin", "rb");
    FILE * f_tel = fopen("../../../../../dataset/mnist/test_labels.bin", "rb");
    if(fread(train_images, 1, TRAIN_NUM*HEIGHT*WIDTH, f_tri) != TRAIN_NUM*HEIGHT*WIDTH) {
        printf("fread size error at line %d\n", __LINE__);
    }
    if(fread(train_labels, 1, TRAIN_NUM, f_trl) != TRAIN_NUM) {
        printf("fread size error at line %d\n", __LINE__);
    }
    if(fread(test_images, 1, TEST_NUM*HEIGHT*WIDTH, f_tei) != TEST_NUM*HEIGHT*WIDTH) {
        printf("fread size error at line %d\n", __LINE__);
    }
    if(fread(test_labels, 1, TEST_NUM, f_tel) != TEST_NUM) {
        printf("fread size error at line %d\n", __LINE__);
    }
    fclose(f_tri);
    fclose(f_trl);
    fclose(f_tei);
    fclose(f_tel);
}

void copy_data(float * train_images, float * test_images, 
               unsigned char * train_images_char, unsigned char * test_images_char)
{
    for(int i = 0; i<TRAIN_NUM*CHANNEL*HEIGHT*WIDTH; i++) {
        train_images[i] = (float)train_images_char[i] / 255.0f;
    }
    for(int i = 0; i<TEST_NUM*CHANNEL*HEIGHT*WIDTH; i++) {
        test_images[i] = (float)test_images_char[i] / 255.0f;
    }
}
typedef unsigned char uint8;

int main(void)
{
    // load your dataset
     // 读取原始数据集
    uint8 * train_images_char = (uint8*)malloc(sizeof(uint8)*TRAIN_NUM*HEIGHT*WIDTH);
    uint8 * train_labels_char = (uint8*)malloc(sizeof(uint8)*TRAIN_NUM);
    uint8 * test_images_char = (uint8*)malloc(sizeof(uint8)*TEST_NUM*HEIGHT*WIDTH);
    uint8 * test_labels_char = (uint8*)malloc(sizeof(uint8)*TEST_NUM);
    read_data(train_images_char, train_labels_char, test_images_char, test_labels_char);

    // 数据集转换为float
    float * train_images = (float*)malloc(sizeof(float)*TRAIN_NUM*CHANNEL*HEIGHT*WIDTH);
    float * test_images = (float*)malloc(sizeof(float)*TEST_NUM*CHANNEL*HEIGHT*WIDTH);
    copy_data(train_images, test_images, train_images_char, test_images_char);

    // init weight and result memory
    init();

    // alloc space for result
    float ** result = (float**)malloc(sizeof(float*)*1);

    // calculation
    int correct = 0;
    for(int i = 0; i<TEST_NUM; i++) {
        calc(result, &test_images[i*28*28]);
        int answer = argmax(result[0], 10);
        if(answer == test_labels_char[i]) {
            correct ++;
        }
        printf("\r%d/%d", correct, i+1);
        fflush(stdout);
    }
    printf("\n");
}
